import json
import os

import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any


class ZarrTrajectoryWindowCache:
    """
    Efficient per-env rolling window cache for ZarrBackedTrajectoryDataset.
    For each env, keeps a window of data in memory and fetches a new window only when needed.

    This is designed for thousands of parallel environments that access different parts
    of the trajectory space. Each environment gets its own cached window that only
    updates when the environment steps outside the current window.

    Note: This cache works with the pre-computed windows stored in the zarr dataset.
    The dataset stores overlapping windows where window i starts at step i.
    """

    def __init__(
        self, zarr_dataset, window_size: int = 64, device: Optional[str] = None
    ):
        self.dataset = zarr_dataset
        self.window_size = window_size
        self.device = device or zarr_dataset.device
        # Per-env cache: env_idx -> (traj_idx, window_start, window_data)
        # window_data is a dict: {data_type: {key: tensor}}
        self.cache: Dict[int, Tuple[int, int, Dict[str, Dict[str, torch.Tensor]]]] = {}

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

        Uses per-environment caching to avoid repeated disk access for thousands
        of parallel environments.

        Args:
            env_idx: Environment index (for parallel environments)
            traj_idx: Trajectory index (must be 0 for single-trajectory datasets)
            step_idx: Step index within trajectory
            key: Which observation/action key to fetch (e.g., 'qpos')
            data_type: 'observations' or 'actions'
        Returns:
            torch.Tensor for the requested key
        """
        if step_idx >= len(self.dataset):
            raise IndexError(
                f"Step index {step_idx} out of bounds for dataset with {len(self.dataset)} windows"
            )

        # Check if we have cached data for this environment
        if env_idx in self.cache:
            cached_traj_idx, cached_window_start, cached_data = self.cache[env_idx]

            # Check if requested data is within the current cached window
            if (
                cached_traj_idx == traj_idx
                and cached_window_start
                <= step_idx
                < cached_window_start + self.window_size
            ):
                # Return from cache
                window_offset = step_idx - cached_window_start
                return cached_data[data_type][key][window_offset]

        # Cache miss or no cache for this env - fetch new window
        self._update_cache(env_idx, traj_idx, step_idx)

        # Now get from the updated cache
        cached_traj_idx, cached_window_start, cached_data = self.cache[env_idx]
        window_offset = step_idx - cached_window_start
        return cached_data[data_type][key][window_offset]

    def _update_cache(self, env_idx: int, traj_idx: int, step_idx: int):
        """
        Update the cache for the given environment with a new window.

        The window will be centered around step_idx to maximize cache hits
        for future requests.
        """
        # Calculate optimal window start to center around step_idx
        window_start = max(0, step_idx - self.window_size // 2)
        window_end = min(len(self.dataset), window_start + self.window_size)

        # Adjust window_start if we hit the end boundary
        if window_end - window_start < self.window_size and window_end == len(
            self.dataset
        ):
            window_start = max(0, window_end - self.window_size)

        # Fetch window data from dataset
        window_data = {"observations": {}, "actions": {}}

        # Pre-allocate tensors for the window
        for obs_key in self.dataset.observation_keys:
            # Get shape from first sample
            sample_shape = self.dataset.root[f"observations/{obs_key}"][0].shape
            window_shape = (self.window_size,) + sample_shape
            window_data["observations"][obs_key] = torch.zeros(
                window_shape, device=self.device, dtype=torch.float32
            )

        for act_key in self.dataset.action_keys:
            # Get shape from first sample
            sample_shape = self.dataset.root[f"actions/{act_key}"][0].shape
            window_shape = (self.window_size,) + sample_shape
            window_data["actions"][act_key] = torch.zeros(
                window_shape, device=self.device, dtype=torch.float32
            )

        # Fill the window with data
        for i, dataset_idx in enumerate(range(window_start, window_end)):
            dataset_sample = self.dataset[dataset_idx]

            # Copy observations (taking first element since dataset stores windows)
            for obs_key in self.dataset.observation_keys:
                window_data["observations"][obs_key][i] = dataset_sample[
                    "observations"
                ][obs_key][0]

            # Copy actions (taking first element since dataset stores windows)
            for act_key in self.dataset.action_keys:
                window_data["actions"][act_key][i] = dataset_sample["actions"][act_key][
                    0
                ]

        # Store in cache
        self.cache[env_idx] = (traj_idx, window_start, window_data)

    def batch_get(
        self, traj_indices, step_indices, key: str, data_type: str = "observations"
    ) -> torch.Tensor:
        """
        Efficiently fetch a batch of (traj_idx, step_idx) pairs for a given key and data_type.

        For now, delegates to the underlying dataset's batch_get method.
        Could be optimized to use cache for frequently accessed ranges.

        Args:
            traj_indices: 1D torch.Tensor or np.ndarray of trajectory indices [N]
            step_indices: 1D torch.Tensor or np.ndarray of step indices [N]
            key: observation or action key
            data_type: 'observations' or 'actions'
        Returns:
            torch.Tensor of shape [N, ...] on the correct device
        """
        return self.dataset.batch_get(traj_indices, step_indices, key, data_type)

    def clear_cache(self, env_idx: Optional[int] = None):
        """
        Clear cache for a specific environment or all environments.

        Args:
            env_idx: Environment to clear cache for. If None, clears all.
        """
        if env_idx is not None:
            self.cache.pop(env_idx, None)
        else:
            self.cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about cache usage.

        Returns:
            Dict with cache statistics
        """
        return {
            "num_cached_environments": len(self.cache),
            "cache_window_size": self.window_size,
            "cached_env_ids": list(self.cache.keys()),
        }
