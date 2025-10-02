"""Zarr-backed, vectorized trajectory access utilities.

VectorizedTrajectoryDataset exposes lightweight, per-environment windowed fetching of
trajectory items (e.g., qpos/qvel) stored in a nested Zarr layout:

    <zarr_root>/<dataset_source>/<motion>/<trajectory>/<key>

Design goals and conventions:
  - Lazy I/O: keep only a small sliding window in memory per env to amortize random
    access (`buffer_size`). This is independent of any consumer `window_size`.
  - Vectorized per-env access: `update_references` assigns trajectory and step for
    each env; `fetch(env_ids, key)` returns the items for those envs at their current
    steps.
  - Consistent shapes and keys across trajectories are expected. Basic validation
    is performed at init to fail fast on mismatches.
"""

from typing import Optional, Iterable

import numpy as np
import zarr
from omegaconf import DictConfig

from iltools_datasets.base_loader import BaseDataset


class VectorizedTrajectoryDataset(BaseDataset):
    """Dataset that reads from per-trajectory Zarr format.

    Each trajectory is stored as a separate Zarr group at path
    `<dataset_source>/<motion>/<trajectory>/<key>`.

    Notes on configuration:
      - `window_size` is passed through but not used internally for fetching a single
        time-step. It is intended for downstream consumers that may form windows
        over time. The actual I/O batching is governed by `buffer_size`.
      - `buffer_size` controls the size of the per-env sliding window we cache.
      - `allow_wrap` (optional, bool, default False): if True, out-of-range steps
        are wrapped modulo trajectory length instead of raising.
    """

    def __init__(self, zarr_path: str, num_envs: int, cfg: DictConfig, **kwargs):
        self.zarr_dataset = zarr.open_group(zarr_path, mode="r")
        self.num_envs = num_envs
        self.cfg = cfg
        # window_size is kept for consumer semantics; not used internally for single-step fetch.
        self.window_size = cfg.window_size

        self.buffer_size = getattr(cfg, "buffer_size", 128)
        self.allow_wrap = bool(getattr(cfg, "allow_wrap", False))

        self.env_traj_ids = np.array([-1] * self.num_envs)
        self.env_steps = np.array([0] * self.num_envs)
        self.window_starts = np.array([0] * self.num_envs)
        self.buffers = np.array([{} for _ in range(self.num_envs)])
        self.handlers = {}

        # Build dataset cache first to avoid repeated filesystem traversal
        self._build_dataset_cache()

        # Gather trajectory lengths and validate key consistency
        self.traj_lengths = []
        self.all_keys = None
        for traj in self.available_trajectories:
            group = self.zarr_dataset[traj]
            assert isinstance(group, zarr.Group)
            if self.all_keys is None:
                self.all_keys = sorted(list(group.keys()))
            else:
                if sorted(list(group.keys())) != self.all_keys:
                    raise ValueError(
                        f"Inconsistent keys across trajectories: {traj} has {sorted(list(group.keys()))}, expected {self.all_keys}"
                    )
            if "qpos" not in group:
                raise KeyError(f"Trajectory {traj} missing required key 'qpos'")
            self.traj_lengths.append(group["qpos"].shape[0])
        self.traj_lengths = np.array(self.traj_lengths)

        # Make mypy happy
        if self.all_keys is None:
            self.all_keys = []

    def _build_dataset_cache(self):
        """Build cache for dataset structure to avoid repeated filesystem traversal."""
        self._cached_dataset_sources = list(self.zarr_dataset.keys())
        self._cached_motions_by_source = {}
        self._cached_trajectories_by_motion = {}
        self._cached_trajectories = []

        for dataset_source in self._cached_dataset_sources:
            dataset_source_group = self.zarr_dataset[dataset_source]
            assert isinstance(dataset_source_group, zarr.Group)
            motions = list(dataset_source_group.keys())
            self._cached_motions_by_source[dataset_source] = motions

            for motion in motions:
                motion_group = dataset_source_group[motion]
                assert isinstance(motion_group, zarr.Group)
                trajectories = sorted(list(motion_group.keys()))
                self._cached_trajectories_by_motion[(dataset_source, motion)] = (
                    trajectories
                )

                # Build full trajectory paths
                for trajectory in trajectories:
                    self._cached_trajectories.append(
                        f"{dataset_source}/{motion}/{trajectory}"
                    )

    @property
    def available_dataset_sources(self) -> list[str]:
        return self._cached_dataset_sources

    @property
    def available_motions(self) -> list[str]:
        return [
            motion
            for dataset_source in self._cached_dataset_sources
            for motion in self._cached_motions_by_source[dataset_source]
        ]

    def available_motions_in(self, dataset_source: str) -> list[str]:
        return self._cached_motions_by_source[dataset_source]

    def available_trajectories_in(self, dataset_source: str, motion: str) -> list[str]:
        return self._cached_trajectories_by_motion[(dataset_source, motion)]

    @property
    def available_trajectories(self) -> list[str]:
        return self._cached_trajectories

    def update_references(
        self,
        env_to_traj: Optional[dict[int, int]] = None,
        env_to_step: Optional[dict[int, int]] = None,
    ):
        if env_to_traj is not None:
            for env_id, traj_id in env_to_traj.items():
                if self.env_traj_ids[env_id] != traj_id:
                    self.env_traj_ids[env_id] = traj_id
                    self.buffers[env_id] = {}
                    self.window_starts[env_id] = 0
        if env_to_step is not None:
            for env_id, step in env_to_step.items():
                self.env_steps[env_id] = step

    def fetch(self, idx: Iterable[int], key: Optional[str] = None) -> np.ndarray:
        """Return values for `key` at current steps for the given env IDs.

        Args:
          idx: Iterable of environment IDs to fetch for (e.g., [0, 3, 7]).
          key: Zarr dataset key within each trajectory group (e.g., "qpos").

        Returns:
          Stacked numpy array of shape [len(idx), ...] with values read at
          each env's current step.
        """
        assert key is not None, "key must be provided (e.g., 'qpos')"

        data_list = []
        for env_id in idx:
            traj_id = self.env_traj_ids[env_id]
            if traj_id == -1:
                raise ValueError(f"No trajectory assigned to env {env_id}")
            step = self.env_steps[env_id]

            if traj_id not in self.handlers:
                self.handlers[traj_id] = {}
            if key not in self.handlers[traj_id]:
                traj_path = self.available_trajectories[traj_id]
                self.handlers[traj_id][key] = self.zarr_dataset[traj_path][key]
            arr = self.handlers[traj_id][key]

            traj_length = self.traj_lengths[traj_id]
            if step < 0 or step >= traj_length:
                if self.allow_wrap and traj_length > 0:
                    # Wrap step into valid range
                    step = step % traj_length
                    self.env_steps[env_id] = step
                else:
                    raise IndexError(
                        f"Step {step} out of range for traj length {traj_length} (env {env_id})"
                    )

            window_start = self.window_starts[env_id]
            buffer = self.buffers[env_id].get(key, None)
            buffer_len = len(buffer) if buffer is not None else 0

            if buffer is None or not (window_start <= step < window_start + buffer_len):
                new_start = max(0, step - self.buffer_size // 2)
                new_end = min(new_start + self.buffer_size, traj_length)
                if step < new_start or step >= new_end:
                    raise IndexError(
                        f"Cannot access step {step} in traj length {traj_length}"
                    )
                buffer_data = arr[new_start:new_end]
                self.buffers[env_id][key] = buffer_data
                self.window_starts[env_id] = new_start

            local_idx = step - self.window_starts[env_id]
            data_list.append(self.buffers[env_id][key][local_idx])

        return np.stack(data_list, axis=0)

    def fetch_window(
        self,
        idx: Iterable[int],
        key: Optional[str],
        window_size: int,
    ) -> np.ndarray:
        """Return a fixed-length window [num_envs, window_size, ...] for each env.

        This uses the same internal buffering as `fetch`, advancing a temporary
        step cursor for each env to collect a contiguous window starting at the
        env's current step. Steps are wrapped when `allow_wrap=True`.
        """
        assert key is not None, "key must be provided (e.g., 'qpos')"
        env_ids = list(idx)
        # Snapshot current steps
        original_steps = {env_id: self.env_steps[env_id] for env_id in env_ids}
        try:
            per_t = []
            for dt in range(window_size):
                # Set per-env temporary steps for this slice
                for env_id in env_ids:
                    traj_id = self.env_traj_ids[env_id]
                    traj_length = self.traj_lengths[traj_id]
                    step = original_steps[env_id] + dt
                    if self.allow_wrap and traj_length > 0:
                        step = step % traj_length
                    self.env_steps[env_id] = step
                # Fetch values at these steps
                per_t.append(self.fetch(env_ids, key))
            # Shape: [window_size, num_envs, ...] -> transpose to [num_envs, window_size, ...]
            stacked = np.stack(per_t, axis=0)
            return np.moveaxis(stacked, 0, 1)
        finally:
            # Restore original steps
            for env_id, step in original_steps.items():
                self.env_steps[env_id] = step
