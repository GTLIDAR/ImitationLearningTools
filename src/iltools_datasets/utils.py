import json
import os

import numpy as np
import torch
from typing import Optional


class ZarrTrajectoryWindowCache:
    """
    Efficient per-env rolling window cache for ZarrBackedTrajectoryDataset.
    For each env, keeps a window of data in memory and fetches a new window only when needed.
    """

    def __init__(
        self, zarr_dataset, window_size: int = 64, device: Optional[str] = None
    ):
        self.dataset = zarr_dataset
        self.window_size = window_size
        self.device = device or zarr_dataset.device
        # Per-env cache: env_idx -> (traj_idx, window_start, window_data)
        self.cache = {}

    def get(
        self,
        env_idx: int,
        traj_idx: int,
        step_idx: int,
        key: str = "qpos",
        data_type: str = "observations",
    ) -> torch.Tensor:
        """
        Get the observation/action for env_idx at (traj_idx, step_idx).
        If the current window does not cover step_idx, fetch a new window.
        Args:
            env_idx: Environment index
            traj_idx: Trajectory index
            step_idx: Step index within trajectory
            key: Which observation/action key to fetch (e.g., 'qpos')
            data_type: 'observations' or 'actions'
        Returns:
            torch.Tensor for the requested key
        """
        cache_entry = self.cache.get(env_idx, None)
        need_new_window = (
            cache_entry is None
            or cache_entry[0] != traj_idx
            or not (cache_entry[1] <= step_idx < cache_entry[1] + self.window_size)
        )
        if need_new_window:
            window_start = (step_idx // self.window_size) * self.window_size
            window_data = self.dataset.get_window(
                traj_idx, window_start, self.window_size
            )
            self.cache[env_idx] = (traj_idx, window_start, window_data)
        else:
            window_start = cache_entry[1]
            window_data = cache_entry[2]
        rel_idx = step_idx - window_start
        return window_data[data_type][key][rel_idx]

    def batch_get(
        self, traj_indices, step_indices, key: str, data_type: str = "observations"
    ) -> torch.Tensor:
        """
        Efficiently fetch a batch of (traj_idx, step_idx) pairs for a given key and data_type.
        Delegates to the underlying dataset's batch_get method.
        Args:
            traj_indices: 1D torch.Tensor or np.ndarray of trajectory indices [N]
            step_indices: 1D torch.Tensor or np.ndarray of step indices [N]
            key: observation or action key
            data_type: 'observations' or 'actions'
        Returns:
            torch.Tensor of shape [N, ...] on the correct device
        """
        return self.dataset.batch_get(traj_indices, step_indices, key, data_type)
