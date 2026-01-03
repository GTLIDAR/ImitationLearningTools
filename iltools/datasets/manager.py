from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import TensorDictReplayBuffer

from iltools.datasets.utils import (
    _map_reference_to_target,
    get_ith_traj_info,
    get_traj_rank_from_info,
)


def get_global_index(
    rank: Tensor, start: Tensor, end: Tensor, local_step: Tensor
) -> Tensor:
    """Vectorized map (traj_rank, local_step) -> global index in replay storage."""
    return (start[rank] + local_step).clamp(min=start[rank], max=end[rank] - 1)


@dataclass(frozen=True)
class ResetSchedule:
    """How to pick the next trajectory rank when an env resets."""

    RANDOM: str = "random"
    SEQUENTIAL: str = "sequential"
    ROUND_ROBIN: str = "round_robin"
    CUSTOM: str = "custom"


class ParallelTrajectoryManager:
    """Manage `num_envs` logical environments, each bound to one expert trajectory.

    - Each env tracks a `(traj_rank, local_step)` cursor.
    - Sampling is done by direct indexing into the replay buffer storage with the
      computed global indices (one step per env).
    - On reset, an env is reassigned to a new trajectory according to
      `reset_schedule`, where trajectories are referenced by their **rank**
      in `traj_info["ordered_traj_list"]`.
    """

    def __init__(
        self,
        *,
        rb: TensorDictReplayBuffer,
        traj_info: dict,
        num_envs: int,
        reset_schedule: str = ResetSchedule.RANDOM,
        custom_reset_fn: Optional[Callable[[Tensor, int], Tensor]] = None,
        wrap_steps: bool = False,
        device: torch.device | str | None = None,
        target_joint_names: Optional[Sequence[str]] = None,
        reference_joint_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.rb = rb
        self.traj_info = traj_info
        self.reset_schedule = reset_schedule
        self.custom_reset_fn = custom_reset_fn
        self.wrap_steps = bool(wrap_steps)

        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)

        self._device = torch.device(device) if device is not None else None

        # Trajectory layout (by rank)
        try:
            start = torch.as_tensor(traj_info["start_index"], dtype=torch.int64)
            end = torch.as_tensor(traj_info["end_index"], dtype=torch.int64)
            ordered = traj_info["ordered_traj_list"]
        except KeyError as e:
            raise KeyError(
                "traj_info must contain keys: 'start_index', 'end_index', 'ordered_traj_list'"
            ) from e

        if start.ndim != 1 or end.ndim != 1 or start.numel() != end.numel():
            raise ValueError("traj_info start/end must be 1D and have the same length")
        if int(start.numel()) == 0:
            raise ValueError("No trajectories found in traj_info")

        self._start = start
        self._end = end
        self._length = (end - start).clamp(min=0)
        self._ordered_traj_list = list(ordered)

        # Per-env state
        self.env_traj_rank = torch.zeros((self.num_envs,), dtype=torch.int64)
        self.env_step = torch.zeros((self.num_envs,), dtype=torch.int64)

        # State for sequential scheduling
        self._next_rank = 0

        # Initialize all envs
        self.reset_envs(torch.arange(self.num_envs, dtype=torch.int64))

        # Get the mapping from reference to target joint names
        self.map_target_to_ref, self.map_ref_to_target = _map_reference_to_target(
            reference_joint_names, target_joint_names, self._device
        )
        self.target_mask = torch.zeros(
            len(target_joint_names), dtype=torch.bool, device=self._device
        )
        self.target_mask[self.map_target_to_ref] = True

        # Memory allocate for important data
        self.joint_pos = torch.empty(
            num_envs, len(target_joint_names), device=device, dtype=torch.float32
        )
        self.joint_vel = torch.empty(
            num_envs, len(target_joint_names), device=device, dtype=torch.float32
        )
        self.root_pos = torch.empty(num_envs, 3, device=device, dtype=torch.float32)
        self.root_quat = torch.empty(num_envs, 4, device=device, dtype=torch.float32)
        self.root_lin_vel = torch.empty(num_envs, 3, device=device, dtype=torch.float32)
        self.root_ang_vel = torch.empty(num_envs, 3, device=device, dtype=torch.float32)

    @property
    def num_trajectories(self) -> int:
        return int(self._start.numel())

    def get_env_traj_info(self, env_id: int) -> tuple[str, str, str]:
        """Return (dataset, motion, trajectory) tuple for the env's current rank."""
        r = int(self.env_traj_rank[int(env_id)])
        return get_ith_traj_info(r, self._ordered_traj_list)  # type: ignore[arg-type]

    def get_traj_rank(self, dataset: str, motion: str, trajectory: str) -> int:
        """Map (dataset,motion,trajectory) to trajectory rank using utils.py logic."""
        return int(
            get_traj_rank_from_info(
                dataset,
                motion,
                trajectory,
                self._ordered_traj_list,  # type: ignore[arg-type]
            )
        )

    def _choose_new_ranks(self, env_ids: Tensor) -> Tensor:
        env_ids = env_ids.to(dtype=torch.int64)
        n = int(env_ids.numel())
        if n == 0:
            return torch.empty((0,), dtype=torch.int64)

        if self.reset_schedule == ResetSchedule.RANDOM:
            return torch.randint(
                low=0, high=self.num_trajectories, size=(n,), dtype=torch.int64
            )

        if self.reset_schedule == ResetSchedule.SEQUENTIAL:
            base = self._next_rank
            ranks = (torch.arange(n, dtype=torch.int64) + base) % self.num_trajectories
            self._next_rank = int((base + n) % self.num_trajectories)
            return ranks

        if self.reset_schedule == ResetSchedule.ROUND_ROBIN:
            # Cycle each env independently: next = (current + 1) % N
            cur = self.env_traj_rank[env_ids]
            return (cur + 1) % self.num_trajectories

        if self.reset_schedule == ResetSchedule.CUSTOM:
            if self.custom_reset_fn is None:
                raise ValueError(
                    "reset_schedule=CUSTOM but custom_reset_fn is None. "
                    "Provide custom_reset_fn(env_ids, num_trajectories)->ranks."
                )
            ranks = self.custom_reset_fn(env_ids, self.num_trajectories)
            ranks = torch.as_tensor(ranks, dtype=torch.int64)
            if ranks.shape != (n,):
                raise ValueError(
                    "custom_reset_fn must return a 1D tensor of shape [len(env_ids)]"
                )
            return ranks % self.num_trajectories

        raise ValueError(f"Unknown reset_schedule: {self.reset_schedule}")

    def reset_envs(
        self, env_ids: Sequence[int] | Tensor, *, ranks: Optional[Tensor] = None
    ) -> None:
        """Reset selected envs: set step=0 and (optionally) reassign traj_rank."""
        env_ids_t = (
            env_ids
            if isinstance(env_ids, torch.Tensor)
            else torch.as_tensor(list(env_ids), dtype=torch.int64)
        ).to(dtype=torch.int64)
        if env_ids_t.numel() == 0:
            return

        if ranks is None:
            new_ranks = self._choose_new_ranks(env_ids_t)
        else:
            new_ranks = torch.as_tensor(ranks, dtype=torch.int64)
            if new_ranks.shape != env_ids_t.shape:
                raise ValueError("ranks must have the same shape as env_ids")
            new_ranks = new_ranks % self.num_trajectories

        self.env_traj_rank[env_ids_t] = new_ranks
        self.env_step[env_ids_t] = 0

    def set_env_cursor(
        self,
        *,
        env_ids: Sequence[int] | Tensor,
        ranks: Tensor,
        steps: Optional[Tensor] = None,
    ) -> None:
        """Set (rank, step) for a set of envs."""
        env_ids_t = (
            env_ids
            if isinstance(env_ids, torch.Tensor)
            else torch.as_tensor(list(env_ids), dtype=torch.int64)
        ).to(dtype=torch.int64)
        ranks_t = torch.as_tensor(ranks, dtype=torch.int64)
        if ranks_t.shape != env_ids_t.shape:
            raise ValueError("ranks must have the same shape as env_ids")
        self.env_traj_rank[env_ids_t] = ranks_t % self.num_trajectories

        if steps is None:
            self.env_step[env_ids_t] = 0
        else:
            steps_t = torch.as_tensor(steps, dtype=torch.int64)
            if steps_t.shape != env_ids_t.shape:
                raise ValueError("steps must have the same shape as env_ids")
            # Clamp into valid range for each env's trajectory
            r = self.env_traj_rank[env_ids_t]
            max_step = (self._length[r] - 1).clamp(min=0)
            self.env_step[env_ids_t] = torch.minimum(steps_t.clamp(min=0), max_step)

    def sample(
        self, env_ids: Sequence[int] | Tensor | None = None, *, advance: bool = True
    ) -> TensorDict:
        """Sample one transition per env (or subset) via direct storage indexing."""
        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, dtype=torch.int64)
        else:
            env_ids_t = (
                env_ids
                if isinstance(env_ids, torch.Tensor)
                else torch.as_tensor(list(env_ids), dtype=torch.int64)
            ).to(dtype=torch.int64)

        r = self.env_traj_rank[env_ids_t]
        step = self.env_step[env_ids_t]

        idx = get_global_index(r, self._start, self._end, step)
        # Direct indexing into storage (fast path)
        td = self.rb[idx]

        if self._device is not None:
            td = td.to(self._device)

        if advance:
            self._advance_steps(env_ids_t)

        # Append additional common tensors to current reference
        qpos = td.get("qpos")
        qvel = td.get("qvel")
        assert qpos is not None and qvel is not None, (
            "qpos and qvel must be present in the reference data"
        )
        self.root_qpos = qpos[..., 0:3]
        self.root_quat = qpos[..., 3:7]
        self.root_lin_vel = qvel[..., 0:3]
        self.root_ang_vel = qvel[..., 3:6]
        joint_pos = qpos[:, 7:]
        joint_vel = qvel[:, 6:]
        self.joint_pos[..., self.map_ref_to_target] = joint_pos[
            ..., self.map_target_to_ref
        ]
        self.joint_vel[..., self.map_ref_to_target] = joint_vel[
            ..., self.map_target_to_ref
        ]
        self.joint_pos[..., ~self.target_mask] = torch.nan
        self.joint_vel[..., ~self.target_mask] = torch.nan
        td.set(key="root_pos", item=self.root_pos)
        td.set(key="root_quat", item=self.root_quat)
        td.set(key="root_lin_vel", item=self.root_lin_vel)
        td.set(key="root_ang_vel", item=self.root_ang_vel)
        td.set(key="joint_pos", item=self.joint_pos)
        td.set(key="joint_vel", item=self.joint_vel)

        return td

    def _advance_steps(self, env_ids: Tensor) -> None:
        r = self.env_traj_rank[env_ids]
        length = self._length[r].clamp(min=1)  # avoid div-by-zero
        step = self.env_step[env_ids]

        if self.wrap_steps:
            self.env_step[env_ids] = (step + 1) % length
        else:
            self.env_step[env_ids] = torch.minimum(step + 1, length - 1)
