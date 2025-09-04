from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch
from tensordict import TensorDict
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerBase, SamplerWithoutReplacement

from .replay_memmap import ExpertMemmapBuilder, Segment


@dataclass
class EnvAssignment:
    """Tracks which (task, traj) each env follows and its local step pointer."""

    task_id: int
    traj_id: int
    step: int = 0


class SequentialPerEnvSampler(SamplerBase):
    """Custom sampler that returns the next index for each env sequentially.

    - Each env is assigned a (task_id, traj_id) segment.
    - On every `sample(storage)` call, it returns indices of shape [num_envs],
      one per env, advancing each env's pointer by 1 (with wrap).
    - This keeps DeepMimic-style synchronized marching through reference clips.
    """

    def __init__(self, *, segments: Sequence[Segment], assignment: list[EnvAssignment]) -> None:
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

    def sample(self, storage: LazyMemmapStorage, batch_size: Optional[int] = None) -> torch.Tensor:  # type: ignore[override]
        # TorchRL will pass batch_size, but we ignore it and return one index per env.
        idx = torch.empty((self.num_envs,), dtype=torch.int64)
        for i, a in enumerate(self.assignment):
            seg = self._seg_by_key[(a.task_id, a.traj_id)]
            idx[i] = seg.index_at(a.step, wrap=True)
            a.step += 1
        return idx


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
                if td.batch_ndim != 1:
                    raise ValueError("Each trajectory must have batch shape [T]")
                total_T += td.batch_size[0]

        builder = ExpertMemmapBuilder(spec.scratch_dir, max_size=total_T, device="cpu")
        segments: list[Segment] = []
        for task_id, trajs in spec.tasks.items():
            for traj_id, td in enumerate(trajs):
                seg = builder.add_trajectory(task_id=task_id, traj_id=traj_id, transitions=td)
                segments.append(seg)

        storage, segments = builder.finalize()

        # Default to uniform sampler; caller can switch to sequential via set_assignment()
        self._sampler: SamplerBase | None = None
        self._assignment: list[EnvAssignment] | None = None

        # Create the replay buffer (lives on CPU; consumer can move to device during .sample())
        self.buffer = TensorDictReplayBuffer(
            storage=storage,
            batch_size=spec.sample_batch_size,
            sampler=self._sampler,  # may be None; default sampler is uniform
            prefetch=3,
            pin_memory=False,
            device="cpu",
            shared=False,
        )

        self._segments = segments

    @property
    def segments(self) -> list[Segment]:
        return list(self._segments)

    def set_device_transform(self, device: torch.device, non_blocking: bool = True) -> None:
        """Make the buffer yield batches already moved to `device`.

        This overlaps H2D copies with compute when prefetch>0.
        """
        self.buffer.append_transform(lambda td: td.to(device, non_blocking=non_blocking))

    def set_assignment(self, assignment: Sequence[EnvAssignment]) -> None:
        """Enable per-env sequential sampling by setting an assignment.

        After this call, `self.buffer.sample()` will return a batch of indices
        of shape [num_envs], one per env, advancing each env's step pointer.
        """
        self._assignment = list(assignment)
        self._sampler = SequentialPerEnvSampler(segments=self._segments, assignment=self._assignment)
        # Recreate the buffer with the sequential sampler, batch_size ignored by sampler
        self.buffer = TensorDictReplayBuffer(
            storage=self.buffer._storage,  # reuse existing storage
            batch_size=len(self._assignment),
            sampler=self._sampler,
            prefetch=3,
            pin_memory=False,
            device="cpu",
            shared=False,
        )

    def clear_assignment(self) -> None:
        """Revert to default (uniform) sampling over all transitions."""
        self._assignment = None
        self._sampler = None
        self.buffer = TensorDictReplayBuffer(
            storage=self.buffer._storage,
            batch_size=self.spec.sample_batch_size,
            sampler=None,
            prefetch=3,
            pin_memory=False,
            device="cpu",
            shared=False,
        )

    def set_uniform_sampler(self, *, batch_size: Optional[int] = None, without_replacement: bool = True) -> None:
        """Switch to uniform minibatch sampling suitable for IPMD.

        - If `without_replacement=True`, uses `SamplerWithoutReplacement`.
        - Otherwise falls back to default sampler behavior.
        - `batch_size` defaults to `spec.sample_batch_size`.
        """
        bs = int(batch_size or self.spec.sample_batch_size)
        sampler = SamplerWithoutReplacement() if without_replacement else None
        self._assignment = None
        self._sampler = sampler
        self.buffer = TensorDictReplayBuffer(
            storage=self.buffer._storage,
            batch_size=bs,
            sampler=self._sampler,
            prefetch=3,
            pin_memory=False,
            device="cpu",
            shared=False,
        )

    def set_custom_sampler(self, sampler: SamplerBase, *, batch_size: int) -> None:
        """Attach any TorchRL sampler (e.g., weighted) for flexible minibatching."""
        self._assignment = None
        self._sampler = sampler
        self.buffer = TensorDictReplayBuffer(
            storage=self.buffer._storage,
            batch_size=int(batch_size),
            sampler=self._sampler,
            prefetch=3,
            pin_memory=False,
            device="cpu",
            shared=False,
        )

