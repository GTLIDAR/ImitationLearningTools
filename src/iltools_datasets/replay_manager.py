from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch
from tensordict import TensorDict
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import Sampler, SamplerWithoutReplacement

from .replay_memmap import ExpertMemmapBuilder, Segment


@dataclass
class EnvAssignment:
    """Tracks which (task, traj) each env follows and its local step pointer."""

    task_id: int
    traj_id: int
    step: int = 0


class SequentialPerEnvSampler(Sampler):
    """Custom sampler that returns the next index for each env sequentially.

    - Each env is assigned a (task_id, traj_id) segment.
    - On every `sample(storage)` call, it returns indices of shape [num_envs],
      one per env, advancing each env's pointer by 1 (with wrap).
    - This keeps DeepMimic-style synchronized marching through reference clips.
    """

    def __init__(
        self, *, segments: Sequence[Segment], assignment: list[EnvAssignment]
    ) -> None:
        super().__init__()
        # Build lookup: (task_id, traj_id) -> Segment
        self._seg_by_key: dict[tuple[int, int], Segment] = {
            (s.task_id, s.traj_id): s for s in segments
        }
        self.assignment = assignment
        self.num_envs = len(self.assignment)

    def set_assignment(self, assignment: list[EnvAssignment]) -> None:
        if len(assignment) != self.num_envs:
            raise ValueError("Assignment length must match num_envs")
        self.assignment = assignment

    def sample(self, storage: LazyMemmapStorage, batch_size: Optional[int] = None) -> tuple[torch.Tensor, dict]:  # type: ignore[override]
        # TorchRL will pass batch_size, but we ignore it and return one index per env.
        idx = torch.empty((self.num_envs,), dtype=torch.int64)
        for i, a in enumerate(self.assignment):
            seg = self._seg_by_key[(a.task_id, a.traj_id)]
            idx[i] = seg.index_at(a.step, wrap=True)
            a.step += 1
        return idx, {}

    def _empty(self):
        """Empty method required by abstract base class."""
        pass

    def dumps(self, path):
        """Dumps method required by abstract base class."""
        # No-op for this sampler as it doesn't need to persist state
        pass

    def loads(self, path):
        """Loads method required by abstract base class."""
        # No-op for this sampler as it doesn't need to persist state
        pass

    def state_dict(self) -> dict[str, Any]:
        """Returns state dict for serialization."""
        return {
            "assignment": [
                {"task_id": a.task_id, "traj_id": a.traj_id, "step": a.step}
                for a in self.assignment
            ],
            "num_envs": self.num_envs,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Loads state dict for deserialization."""
        if "assignment" in state_dict:
            self.assignment = [
                EnvAssignment(
                    task_id=a["task_id"], traj_id=a["traj_id"], step=a["step"]
                )
                for a in state_dict["assignment"]
            ]
        if "num_envs" in state_dict:
            self.num_envs = state_dict["num_envs"]


class AssignedUniformSampler(Sampler):
    """Uniform sampler restricted to the segments referenced by an assignment.

    Supports both with- and without-replacement sampling. When sampling without
    replacement, it maintains an internal permutation of the allowed indices and
    iterates through it across successive calls, reshuffling upon exhaustion.

    This is useful when multiple environments are each bound to a distinct
    (task_id, traj_id), and training requires minibatches that only draw from
    those assigned expert trajectories, independent of the environments' current
    step pointers.
    """

    def __init__(
        self,
        *,
        segments: Sequence[Segment],
        assignment: Sequence[EnvAssignment],
        without_replacement: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        # Keep full mapping for all available segments
        self._all_seg_by_key: dict[tuple[int, int], Segment] = {
            (s.task_id, s.traj_id): s for s in segments
        }
        self._without_replacement = bool(without_replacement)
        self._device = device

        # Derived state for the currently allowed subset
        self._allowed_segments: list[Segment] = []
        self._cumlen: torch.Tensor | None = None
        self._starts: torch.Tensor | None = None
        self._total: int = 0
        self._perm: Optional[torch.Tensor] = None
        self._pos: int = 0

        # Initialize with given assignment
        self._init_allowed_from_assignment(assignment)

    def _reset_perm(self) -> None:
        if self._total <= 0:
            self._perm = torch.empty((0,), dtype=torch.int64, device=self._device)
            self._pos = 0
            return
        # Build flat list of all allowed indices and shuffle
        parts = [
            torch.arange(s.start, s.start + s.length, dtype=torch.int64)
            for s in self._allowed_segments
        ]
        if len(parts) == 0:
            self._perm = torch.empty((0,), dtype=torch.int64, device=self._device)
            self._pos = 0
            return
        flat = torch.cat(parts, dim=0)
        perm = torch.randperm(flat.numel())
        self._perm = flat[perm]
        self._pos = 0

    def set_assignment(self, assignment: Sequence[EnvAssignment]) -> None:
        # Reinitialize allowed subset from new assignment (can add/remove segments)
        self._init_allowed_from_assignment(assignment)

    def _init_allowed_from_assignment(
        self, assignment: Sequence[EnvAssignment]
    ) -> None:
        keys = []
        seen: set[tuple[int, int]] = set()
        for a in assignment:
            k = (a.task_id, a.traj_id)
            if k in self._all_seg_by_key and k not in seen:
                keys.append(k)
                seen.add(k)
        self._allowed_segments = [self._all_seg_by_key[k] for k in keys]
        if len(self._allowed_segments) == 0:
            self._cumlen = torch.zeros((0,), dtype=torch.int64)
            self._starts = torch.zeros((0,), dtype=torch.int64)
            self._total = 0
        else:
            lengths = torch.tensor(
                [s.length for s in self._allowed_segments], dtype=torch.int64
            )
            self._cumlen = lengths.cumsum(dim=0)
            self._starts = torch.tensor(
                [s.start for s in self._allowed_segments], dtype=torch.int64
            )
            self._total = int(self._cumlen[-1])
        # Reset without-replacement state
        self._perm = None
        self._pos = 0
        if self._without_replacement:
            self._reset_perm()

    def sample(  # type: ignore[override]
        self, storage: LazyMemmapStorage, batch_size: int
    ) -> tuple[torch.Tensor, dict]:
        bs = int(batch_size)
        if self._total <= 0 or bs <= 0:
            return torch.empty((0,), dtype=torch.int64), {}

        if self._without_replacement:
            assert self._perm is not None
            if self._pos + bs <= self._perm.numel():
                out = self._perm[self._pos : self._pos + bs]
                self._pos += bs
                return out, {}
            # Wrap across epoch boundary: finish current perm then reshuffle
            remain = self._perm.numel() - self._pos
            part1 = self._perm[self._pos :] if remain > 0 else None
            self._reset_perm()
            need = bs - remain
            part2 = self._perm[:need] if need > 0 else None
            if part1 is None:
                out = (
                    part2 if part2 is not None else torch.empty((0,), dtype=torch.int64)
                )
            elif part2 is None:
                out = part1
            else:
                out = torch.cat([part1, part2], dim=0)
            # Update position
            self._pos = int(need)
            return out, {}

        # With replacement: sample offsets in [0, total) uniformly, then map
        r = torch.randint(low=0, high=self._total, size=(bs,), dtype=torch.int64)
        if self._cumlen is None or self._starts is None:
            return torch.empty((0,), dtype=torch.int64), {}
        seg_idx = torch.bucketize(r, self._cumlen, right=True)
        prev_cum = torch.zeros_like(self._cumlen)
        prev_cum[1:] = self._cumlen[:-1]
        offsets = r - prev_cum[seg_idx]
        starts = self._starts[seg_idx]
        idx = starts + offsets
        return idx, {}

    def _empty(self):
        pass

    def dumps(self, path):
        pass

    def loads(self, path):
        pass

    def state_dict(self) -> dict[str, Any]:
        return {
            "without_replacement": self._without_replacement,
            "total": self._total,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # pragma: no cover
        # Minimal restore; full reconstruction requires segments/assignment context
        self._without_replacement = bool(state_dict.get("without_replacement", True))


@dataclass
class ExpertReplaySpec:
    """Specification for building an expert replay buffer.

    - tasks: map task_id -> list of trajectories, where each trajectory is a
      TensorDict with batch shape [T] and keys 'observation', 'action', ('next','observation').
    - scratch_dir: directory for memmap files (disk heavy, RAM light).
    - sample_batch_size: used only if you fall back to random sampling; sequential
      sampler ignores it and returns [num_envs].
    """

    tasks: Mapping[int, Sequence[TensorDict]]
    scratch_dir: str
    device: str = "cpu"
    sample_batch_size: int = 256


class ExpertReplayManager:
    """Builds and serves a disk-backed expert replay buffer with sequential sampling.

    Key features:
    - Multi-task: arbitrary number of tasks, each with 1+ trajectories.
    - Sequential sampling per env: DeepMimic-style alignment via `EnvAssignment`.
    - Disk-based storage via `LazyMemmapStorage` to minimize RAM.
    - Minimal training overhead: sampling is just integer indexing; no Python loops
      in hot path beyond per-env pointer increments.
    """

    def __init__(self, spec: ExpertReplaySpec) -> None:
        self.spec = spec
        # Count total transitions
        total_T = 0
        for t_id, trajs in spec.tasks.items():
            for td in trajs:
                if len(td.batch_size) != 1:
                    raise ValueError("Each trajectory must have batch shape [T]")
                total_T += td.batch_size[0]

        builder = ExpertMemmapBuilder(spec.scratch_dir, max_size=total_T, device="cpu")
        segments: list[Segment] = []
        for task_id, trajs in spec.tasks.items():
            for traj_id, td in enumerate(trajs):
                seg = builder.add_trajectory(
                    task_id=task_id, traj_id=traj_id, transitions=td
                )
                segments.append(seg)

        storage, segments = builder.finalize()

        # Default to uniform sampler; caller can switch to sequential via set_assignment()
        self._sampler: Sampler | None = None
        self._assignment: list[EnvAssignment] | None = None

        # Create the replay buffer (lives on CPU; consumer can move to device during .sample())
        self.buffer = TensorDictReplayBuffer(
            storage=storage,
            batch_size=spec.sample_batch_size,
            sampler=self._sampler,  # may be None; default sampler is uniform
            prefetch=0,  # Disable prefetch to avoid issues with assignment updates
            pin_memory=False,
            shared=False,
        )

        self._segments = segments
        self._transforms: list[Any] = (
            []
        )  # Store transforms to restore after buffer recreation

    @property
    def segments(self) -> list[Segment]:
        return list(self._segments)

    def _restore_transforms(self) -> None:
        """Restore transforms that were previously applied to the buffer."""
        for transform in self._transforms:
            self.buffer.append_transform(transform)

    def set_device_transform(
        self, device: torch.device, non_blocking: bool = True
    ) -> None:
        """Make the buffer yield batches already moved to `device`.

        This overlaps H2D copies with compute when prefetch>0.
        """

        def device_transform(td: TensorDict) -> TensorDict:
            return td.to(device, non_blocking=non_blocking)

        self._transforms.append(device_transform)
        self.buffer.append_transform(device_transform)  # type: ignore[arg-type]

    def set_assignment(self, assignment: Sequence[EnvAssignment]) -> None:
        """Enable per-env sequential sampling by setting an assignment.

        After this call, `self.buffer.sample()` will return a batch of indices
        of shape [num_envs], one per env, advancing each env's step pointer.
        """
        self._assignment = list(assignment)
        self._sampler = SequentialPerEnvSampler(
            segments=self._segments, assignment=self._assignment
        )
        # Recreate the buffer with the sequential sampler, batch_size ignored by sampler
        self.buffer = TensorDictReplayBuffer(
            storage=self.buffer._storage,  # reuse existing storage
            batch_size=len(self._assignment),
            sampler=self._sampler,
            prefetch=0,
            pin_memory=False,
            shared=False,
        )
        # Preserve any transforms that were previously applied
        self._restore_transforms()

    def clear_assignment(self) -> None:
        """Revert to default (uniform) sampling over all transitions."""
        self._assignment = None
        self._sampler = None
        self.buffer = TensorDictReplayBuffer(
            storage=self.buffer._storage,
            batch_size=self.spec.sample_batch_size,
            sampler=None,
            prefetch=0,
            pin_memory=False,
            shared=False,
        )
        # Preserve any transforms that were previously applied
        self._restore_transforms()

    def update_env_assignment(
        self, env_index: int, *, task_id: int, traj_id: int, step: Optional[int] = 0
    ) -> None:
        """Update the (task,traj[,step]) for a single env.

        Requires that an assignment has already been set via `set_assignment`.
        Works both for sequential sampler and assignment-aware uniform sampler.
        """
        if self._assignment is None:
            raise RuntimeError(
                "No assignment set. Call set_assignment([...]) once, then update per-env."
            )
        if self._assignment is None:
            raise RuntimeError(
                "No assignment set. Call set_assignment([...]) once, then update per-env."
            )
        if env_index < 0 or env_index >= len(self._assignment):
            raise IndexError(
                f"env_index {env_index} out of range [0,{len(self._assignment)-1}]"
            )
        current_step = self._assignment[env_index].step
        new_step = current_step if step is None else int(step)
        self._assignment[env_index] = EnvAssignment(
            task_id=int(task_id), traj_id=int(traj_id), step=new_step
        )
        # Notify current sampler if it cares about assignment
        if isinstance(self._sampler, SequentialPerEnvSampler):
            self._sampler.assignment[env_index] = self._assignment[env_index]
        elif isinstance(self._sampler, AssignedUniformSampler):
            self._sampler.set_assignment(self._assignment)

    def update_assignments(
        self,
        updates: Mapping[int, EnvAssignment | tuple[int, int] | tuple[int, int, int]],
        *,
        default_step: int = 0,
    ) -> None:
        """Bulk update of env assignments.

        `updates` maps `env_index -> (task_id, traj_id[, step])` or to an `EnvAssignment`.
        Missing `step` in a tuple defaults to `default_step`.
        """
        if self._assignment is None:
            raise RuntimeError(
                "No assignment set. Call set_assignment([...]) once, then update per-env."
            )
        for env_index, spec in updates.items():
            if self._assignment is None:
                raise RuntimeError("Assignment is None")
            if env_index < 0 or env_index >= len(self._assignment):
                raise IndexError(
                    f"env_index {env_index} out of range [0,{len(self._assignment)-1}]"
                )
            if isinstance(spec, EnvAssignment):
                new_a = spec
            else:
                if len(spec) == 2:
                    task_id, traj_id = spec  # type: ignore[misc]
                    step = default_step
                elif len(spec) == 3:
                    task_id, traj_id, step = spec  # type: ignore[misc]
                else:  # pragma: no cover - defensive
                    raise ValueError(
                        "Update tuples must be (task_id, traj_id) or (task_id, traj_id, step)"
                    )
                new_a = EnvAssignment(int(task_id), int(traj_id), int(step))
            self._assignment[env_index] = new_a
            if isinstance(self._sampler, SequentialPerEnvSampler):
                if self._sampler.assignment is not None:
                    self._sampler.assignment[env_index] = new_a
        # For assignment-aware uniform sampler, recompute allowed set
        if (
            isinstance(self._sampler, AssignedUniformSampler)
            and self._assignment is not None
        ):
            self._sampler.set_assignment(self._assignment)

    def set_uniform_sampler(
        self,
        *,
        batch_size: Optional[int] = None,
        without_replacement: bool = True,
        respect_assignment: bool = True,
    ) -> None:
        """Switch to uniform minibatch sampling suitable for IPMD.

        - If `without_replacement=True`, uses `SamplerWithoutReplacement`.
        - Otherwise falls back to default sampler behavior.
        - `batch_size` defaults to `spec.sample_batch_size`.
        - If `respect_assignment=True` and an assignment is set, restrict sampling
          to the assigned (task,traj) segments using a dedicated sampler.
        """
        bs = int(batch_size or self.spec.sample_batch_size)
        sampler: Sampler | None
        if respect_assignment and self._assignment:
            sampler = AssignedUniformSampler(
                segments=self._segments,
                assignment=self._assignment,
                without_replacement=without_replacement,
            )
            # Keep assignment so users can inspect/modify it later
        else:
            sampler = SamplerWithoutReplacement() if without_replacement else None
            # Clear assignment since we do global uniform sampling
            self._assignment = None
        self._sampler = sampler
        self.buffer = TensorDictReplayBuffer(
            storage=self.buffer._storage,
            batch_size=bs,
            sampler=self._sampler,
            prefetch=0,
            pin_memory=False,
            shared=False,
        )
        # Preserve any transforms that were previously applied
        self._restore_transforms()

    def set_custom_sampler(self, sampler: Sampler, *, batch_size: int) -> None:
        """Attach any TorchRL sampler (e.g., weighted) for flexible minibatching."""
        self._assignment = None
        self._sampler = sampler
        self.buffer = TensorDictReplayBuffer(
            storage=self.buffer._storage,
            batch_size=int(batch_size),
            sampler=self._sampler,
            prefetch=0,
            pin_memory=False,
            shared=False,
        )
        # Preserve any transforms that were previously applied
        self._restore_transforms()
