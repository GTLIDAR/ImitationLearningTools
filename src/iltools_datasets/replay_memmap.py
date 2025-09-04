from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from tensordict import TensorDict
from torchrl.data import LazyMemmapStorage


@dataclass(frozen=True)
class Segment:
    """Represents a contiguous [start, start+length) region in storage.

    Each segment corresponds to one (task_id, traj_id) pair and stores all
    its transitions in order. This gives O(1) mapping from (task, traj, t)
    to global storage index.
    """

    task_id: int
    traj_id: int
    start: int
    length: int

    def index_at(self, t: int, wrap: bool = True) -> int:
        if self.length <= 0:
            raise IndexError("Empty segment")
        if wrap:
            t = t % self.length
        elif t < 0 or t >= self.length:
            raise IndexError(f"t={t} out of range for length={self.length}")
        return self.start + t


class ExpertMemmapBuilder:
    """Incrementally builds a memmap-backed storage of expert transitions.

    - Uses `LazyMemmapStorage` to keep host RAM low; data is memory-mapped to disk.
    - Stores transitions as TensorDicts with keys: 'observation', 'action', ('next','observation').
    - Tracks `Segment`s per (task_id, traj_id) for O(1) sequential indexing.

    Workflow:
      builder = ExpertMemmapBuilder(scratch_dir, max_size)
      seg = builder.add_trajectory(task_id, traj_id, td_transitions)
      ... repeat ...
      storage, segments = builder.finalize()
    """

    def __init__(self, scratch_dir: str | os.PathLike, max_size: int, *, device: str = "cpu") -> None:
        self.scratch_dir = str(scratch_dir)
        Path(self.scratch_dir).mkdir(parents=True, exist_ok=True)
        self.storage = LazyMemmapStorage(max_size=max_size, device=device, scratch_dir=self.scratch_dir)
        self._size = 0
        self._segments: list[Segment] = []

    @property
    def size(self) -> int:
        return self._size

    @property
    def segments(self) -> list[Segment]:
        return list(self._segments)

    def add_trajectory(self, task_id: int, traj_id: int, transitions: TensorDict) -> Segment:
        """Append one trajectory's transitions to storage.

        transitions: TensorDict with batch shape [T] and keys:
          - 'observation': [...]
          - 'action': [...]
          - ('next','observation'): [...]
        """
        if transitions.batch_ndim != 1:
            raise ValueError("Expected 1-D batch of transitions with shape [T]")

        T = transitions.batch_size[0]
        start = self._size
        self.storage.extend(transitions)
        self._size += T
        seg = Segment(task_id=task_id, traj_id=traj_id, start=start, length=T)
        self._segments.append(seg)
        return seg

    def finalize(self) -> tuple[LazyMemmapStorage, list[Segment]]:
        return self.storage, list(self._segments)


# -----------------------------
# Helper builders for trajectories
# -----------------------------


def _ensure_1d_batch(t: torch.Tensor, name: str) -> None:
    if t.ndim < 2:
        raise ValueError(f"{name} must have a batch/time dimension: got shape {t.shape}")


def concat_components(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    """Concatenate a sequence of tensors along the last dimension.

    Assumes each tensor has the same leading shape `[T, ...]`. Returns a tensor
    of shape `[T, sum(last_dims)]` (broadcasting other middle dims if present).
    """
    if not parts:
        raise ValueError("concat_components received an empty sequence")
    for i, p in enumerate(parts):
        _ensure_1d_batch(p, f"parts[{i}]")
    return torch.cat(parts, dim=-1)


def build_trajectory_td(
    *,
    observation: torch.Tensor,
    action: torch.Tensor,
    next_observation: torch.Tensor,
) -> TensorDict:
    """Pack (s, a, s') into a TensorDict with batch shape [T].

    Shapes:
      - observation: [T, obs_dim]
      - action: [T, act_dim]
      - next_observation: [T, obs_dim]
    """
    _ensure_1d_batch(observation, "observation")
    _ensure_1d_batch(action, "action")
    _ensure_1d_batch(next_observation, "next_observation")

    T = observation.shape[0]
    if action.shape[0] != T or next_observation.shape[0] != T:
        raise ValueError("All inputs must share the same leading length [T]")

    td = TensorDict({}, batch_size=[T], device=observation.device)
    td.set("observation", observation)
    td.set("action", action)
    td.set(("next", "observation"), next_observation)
    return td


def build_trajectory_td_from_components(
    *,
    obs_parts: Sequence[torch.Tensor],
    action: torch.Tensor,
    next_obs_parts: Sequence[torch.Tensor],
) -> TensorDict:
    """Concatenate obs parts and next-obs parts, then pack into a TensorDict.

    Useful when your dataset stores qpos, qvel, etc. separately.
    """
    obs = concat_components(obs_parts)
    next_obs = concat_components(next_obs_parts)
    return build_trajectory_td(observation=obs, action=action, next_observation=next_obs)

