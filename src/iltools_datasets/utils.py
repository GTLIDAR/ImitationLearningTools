import json
import os
import time

import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any, Union
from iltools_datasets import (
    ZarrBackedTrajectoryDataset,
    export_trajectories_to_zarr,
    LocoMuJoCoLoader,
)

try:
    from tensordict import TensorDict

    TENSORDICT_AVAILABLE = True
except ImportError:
    TENSORDICT_AVAILABLE = False
    TensorDict = type(None)  # Placeholder for type checking\
    raise ImportError("TensorDict not available")


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
        self,
        zarr_dataset: ZarrBackedTrajectoryDataset,
        window_size: int = 64,
        max_envs: int = 10000,
        device: Optional[str] = None,
    ):
        self.dataset = zarr_dataset
        self.window_size = window_size
        self.max_envs = max_envs
        self.device = device or zarr_dataset.device

        # Vectorized cache metadata
        self.cache_valid = torch.zeros(max_envs, dtype=torch.bool, device=self.device)
        self.cache_traj_indices = torch.full(
            (max_envs,), -1, dtype=torch.long, device=self.device
        )
        self.cache_window_starts = torch.zeros(
            max_envs, dtype=torch.long, device=self.device
        )

        # Initialize cache using TensorDict if available, otherwise fallback to dict
        if TENSORDICT_AVAILABLE:
            self.cache_data: Union[TensorDict, Dict[str, Dict[str, torch.Tensor]]]
            self._init_tensordict_cache()
        else:
            self.cache_data: Dict[str, Dict[str, torch.Tensor]]
            self._init_dict_cache()

    def _init_tensordict_cache(self) -> None:
        """Initialize cache using TensorDict for optimal performance."""
        if not TENSORDICT_AVAILABLE:
            raise RuntimeError("TensorDict not available")

        from tensordict import TensorDict

        # Create flat cache tensors - don't nest TensorDicts for now
        cache_tensors = {}

        for obs_key in self.dataset.observation_keys:
            sample_shape = self.dataset.root[f"observations/{obs_key}"][0].shape
            cache_shape = (self.max_envs, self.window_size) + sample_shape
            cache_tensors[f"observations.{obs_key}"] = torch.zeros(
                cache_shape, device=self.device, dtype=torch.float32
            )

        for act_key in self.dataset.action_keys:
            sample_shape = self.dataset.root[f"actions/{act_key}"][0].shape
            cache_shape = (self.max_envs, self.window_size) + sample_shape
            cache_tensors[f"actions.{act_key}"] = torch.zeros(
                cache_shape, device=self.device, dtype=torch.float32
            )

        self.cache_data = TensorDict(
            cache_tensors,
            batch_size=(self.max_envs, self.window_size),
            device=self.device,
        )

    def _init_dict_cache(self) -> None:
        """Fallback to dictionary-based cache if TensorDict not available."""
        self.cache_data = {"observations": {}, "actions": {}}

        for obs_key in self.dataset.observation_keys:
            sample_shape = self.dataset.root[f"observations/{obs_key}"][0].shape
            cache_shape = (self.max_envs, self.window_size) + sample_shape
            self.cache_data["observations"][obs_key] = torch.zeros(
                cache_shape, device=self.device, dtype=torch.float32
            )

        for act_key in self.dataset.action_keys:
            sample_shape = self.dataset.root[f"actions/{act_key}"][0].shape
            cache_shape = (self.max_envs, self.window_size) + sample_shape
            self.cache_data["actions"][act_key] = torch.zeros(
                cache_shape, device=self.device, dtype=torch.float32
            )

    def _get_cached_data(
        self, env_idx: int, window_offset: int, key: str, data_type: str
    ) -> torch.Tensor:
        """Helper method to get data from cache, handling both TensorDict and dict storage."""
        if TENSORDICT_AVAILABLE and hasattr(self.cache_data, "batch_size"):
            # TensorDict case - use flattened key
            flat_key = f"{data_type}.{key}"
            return self.cache_data[flat_key][env_idx, window_offset]
        else:
            # Regular dict case
            cache_data = self.cache_data
            return cache_data[data_type][key][env_idx, window_offset]

    def _set_cached_data(
        self,
        env_idx: int,
        window_offset: int,
        key: str,
        data_type: str,
        value: torch.Tensor,
    ) -> None:
        """Helper method to set data in cache, handling both TensorDict and dict storage."""
        if TENSORDICT_AVAILABLE and hasattr(self.cache_data, "batch_size"):
            # TensorDict case - use flattened key
            flat_key = f"{data_type}.{key}"
            self.cache_data[flat_key][env_idx, window_offset] = value
        else:
            # Regular dict case
            cache_data = self.cache_data
            cache_data[data_type][key][env_idx, window_offset] = value

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
        if self.cache_valid[env_idx]:
            cached_traj_idx = self.cache_traj_indices[env_idx]
            cached_window_start = self.cache_window_starts[env_idx]

            # Check if requested data is within the current cached window
            if (
                cached_traj_idx == traj_idx
                and cached_window_start
                <= step_idx
                < cached_window_start + self.window_size
            ):
                # Return from cache
                window_offset = step_idx - cached_window_start
                return self._get_cached_data(env_idx, window_offset, key, data_type)

        # Cache miss or no cache for this env - fetch new window
        self._update_cache(env_idx, traj_idx, step_idx)

        # Now get from the updated cache
        cached_window_start = self.cache_window_starts[env_idx]
        window_offset = step_idx - cached_window_start
        return self._get_cached_data(env_idx, window_offset, key, data_type)

    def _update_cache(self, env_idx: int, traj_idx: int, step_idx: int) -> None:
        """
        Update the cache for the given environment with a new window.

        The window will be centered around step_idx to maximize cache hits
        for future requests.
        """
        update_start = time.perf_counter()

        # Calculate optimal window start to center around step_idx
        window_start = max(0, step_idx - self.window_size // 2)
        window_end = min(len(self.dataset), window_start + self.window_size)

        # Adjust window_start if we hit the end boundary
        if window_end - window_start < self.window_size and window_end == len(
            self.dataset
        ):
            window_start = max(0, window_end - self.window_size)

        calc_time = time.perf_counter()

        # Fill the window with data directly into cache tensors
        for i, dataset_idx in enumerate(range(window_start, window_end)):
            io_start = time.perf_counter()
            dataset_sample = self.dataset[dataset_idx]
            io_end = time.perf_counter()

            # Copy observations (taking first element since dataset stores windows)
            for obs_key in self.dataset.observation_keys:
                value = dataset_sample["observations"][obs_key][0]
                self._set_cached_data(env_idx, i, obs_key, "observations", value)

            # Copy actions (taking first element since dataset stores windows)
            for act_key in self.dataset.action_keys:
                value = dataset_sample["actions"][act_key][0]
                self._set_cached_data(env_idx, i, act_key, "actions", value)

            if i == 0:  # Print timing for first sample only
                print(f"  Single dataset access: {(io_end - io_start) * 1000:.2f}ms")

        # Update cache metadata
        self.cache_valid[env_idx] = True
        self.cache_traj_indices[env_idx] = traj_idx
        self.cache_window_starts[env_idx] = window_start

        total_update_time = time.perf_counter()
        print(
            f"  _update_cache total: {(total_update_time - update_start) * 1000:.2f}ms"
        )
        print(f"  Window calc: {(calc_time - update_start) * 1000:.2f}ms")
        print(f"  Data loading: {(total_update_time - calc_time) * 1000:.2f}ms")

    def batch_get(
        self,
        traj_indices: Union[torch.Tensor, np.ndarray],
        step_indices: Union[torch.Tensor, np.ndarray],
        key: str,
        data_type: str = "observations",
        env_indices: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> torch.Tensor:
        """
        Fully vectorized batch fetch using cached windows.
        Uses all environments by default (env_indices = 0, 1, 2, ..., len(traj_indices)-1).
        Optimized for TensorDict when available.

        Args:
            traj_indices: 1D tensor/array of trajectory indices [N]
            step_indices: 1D tensor/array of step indices [N]
            key: observation or action key
            data_type: 'observations' or 'actions'
            env_indices: Optional 1D tensor/array of environment indices [N].
                        If None, uses torch.arange(len(traj_indices))
        Returns:
            torch.Tensor of shape [N, ...] on the correct device
        """
        start_time = time.perf_counter()

        # Ensure tensors are on the right device and dtype
        traj_indices = torch.as_tensor(
            traj_indices, dtype=torch.long, device=self.device
        )
        step_indices = torch.as_tensor(
            step_indices, dtype=torch.long, device=self.device
        )

        # Default to using sequential environment indices if not provided
        if env_indices is None:
            env_indices = torch.arange(
                len(traj_indices), dtype=torch.long, device=self.device
            )
        else:
            env_indices = torch.as_tensor(
                env_indices, dtype=torch.long, device=self.device
            )

        tensor_prep_time = time.perf_counter()
        print(f"Tensor preparation: {(tensor_prep_time - start_time) * 1000:.2f}ms")

        # Vectorized cache hit check
        cache_valid_for_envs = self.cache_valid[env_indices]  # [N]
        cached_traj_match = self.cache_traj_indices[env_indices] == traj_indices  # [N]
        cached_window_starts = self.cache_window_starts[env_indices]  # [N]

        step_in_window = (step_indices >= cached_window_starts) & (
            step_indices < cached_window_starts + self.window_size
        )  # [N]

        # Combined cache hit mask
        cache_hits = cache_valid_for_envs & cached_traj_match & step_in_window  # [N]

        cache_check_time = time.perf_counter()
        print(
            f"Cache hit detection: {(cache_check_time - tensor_prep_time) * 1000:.2f}ms"
        )
        print(
            f"Cache hit rate: {cache_hits.sum().item()}/{len(cache_hits)} ({cache_hits.float().mean().item() * 100:.1f}%)"
        )

        # For cache hits, compute window offsets
        window_offsets = step_indices - cached_window_starts  # [N]

        # Get cache hit data - optimized path for TensorDict
        hit_env_indices = env_indices[cache_hits]  # [num_hits]
        hit_window_offsets = window_offsets[cache_hits]  # [num_hits]

        # Prepare result tensor
        if TENSORDICT_AVAILABLE and hasattr(self.cache_data, "batch_size"):
            flat_key = f"{data_type}.{key}"
            sample_shape = self.cache_data[flat_key].shape[
                2:
            ]  # Shape after [env, window]
        else:
            cache_data = self.cache_data
            sample_shape = cache_data[data_type][key].shape[
                2:
            ]  # Shape after [env, window]

        result_shape = (len(env_indices),) + tuple(sample_shape)
        results = torch.zeros(result_shape, device=self.device, dtype=torch.float32)

        result_prep_time = time.perf_counter()
        print(
            f"Result tensor preparation: {(result_prep_time - cache_check_time) * 1000:.2f}ms"
        )

        # Fill cache hits - TensorDict optimization
        if len(hit_env_indices) > 0:
            if TENSORDICT_AVAILABLE and hasattr(self.cache_data, "batch_size"):
                # TensorDict advanced indexing - can get all data for all hits at once
                flat_key = f"{data_type}.{key}"
                results[cache_hits] = self.cache_data[flat_key][
                    hit_env_indices, hit_window_offsets
                ]
            else:
                # Fallback to regular tensor indexing
                cache_data = self.cache_data
                results[cache_hits] = cache_data[data_type][key][
                    hit_env_indices, hit_window_offsets
                ]

        cache_hit_fill_time = time.perf_counter()
        print(
            f"Cache hit data gathering: {(cache_hit_fill_time - result_prep_time) * 1000:.2f}ms"
        )

        # Handle cache misses
        miss_mask = ~cache_hits
        if miss_mask.any():
            print(f"Cache miss for {miss_mask.sum().item()} environments")
            miss_indices = torch.where(miss_mask)[0]

            miss_start_time = time.perf_counter()

            # For cache misses, update cache and get data
            for idx in miss_indices:
                env_idx = env_indices[idx].item()
                traj_idx = traj_indices[idx].item()
                step_idx = step_indices[idx].item()

                # Update cache for this environment
                cache_update_start = time.perf_counter()
                self._update_cache(env_idx, traj_idx, step_idx)
                cache_update_end = time.perf_counter()

                # Now get the data from the updated cache
                window_offset = step_idx - self.cache_window_starts[env_idx]
                results[idx] = self._get_cached_data(
                    env_idx, window_offset, key, data_type
                )

                if idx == miss_indices[0]:  # Print timing for first miss only
                    print(
                        f"Single cache update: {(cache_update_end - cache_update_start) * 1000:.2f}ms"
                    )

            miss_end_time = time.perf_counter()
            print(
                f"Total cache miss handling: {(miss_end_time - miss_start_time) * 1000:.2f}ms"
            )
            print(
                f"Average per miss: {(miss_end_time - miss_start_time) * 1000 / len(miss_indices):.2f}ms"
            )

        total_time = time.perf_counter()
        print(f"TOTAL batch_get time: {(total_time - start_time) * 1000:.2f}ms")
        print("=" * 50)

        return results

    def clear_cache(self, env_idx: Optional[int] = None) -> None:
        """
        Clear cache for a specific environment or all environments.

        Args:
            env_idx: Environment to clear cache for. If None, clears all.
        """
        if env_idx is not None:
            self.cache_valid[env_idx] = False
            self.cache_traj_indices[env_idx] = -1
            self.cache_window_starts[env_idx] = 0

            # Clear cached data
            if TENSORDICT_AVAILABLE and hasattr(self.cache_data, "batch_size"):
                # TensorDict: zero out the specific environment slice
                if hasattr(self.cache_data, "zero_"):
                    # Create a slice and zero it
                    env_slice = self.cache_data[env_idx]
                    if hasattr(env_slice, "zero_"):
                        env_slice.zero_()
            else:
                # Regular dict: zero out tensors for this environment
                cache_data = self.cache_data
                if isinstance(cache_data, dict):
                    for data_type in cache_data:
                        for cache_key in cache_data[data_type]:
                            cache_data[data_type][cache_key][env_idx] = (
                                torch.zeros_like(
                                    cache_data[data_type][cache_key][env_idx]
                                )
                            )
        else:
            self.cache_valid.fill_(False)
            self.cache_traj_indices.fill_(-1)
            self.cache_window_starts.fill_(0)

            # Clear all cached data
            if TENSORDICT_AVAILABLE and hasattr(self.cache_data, "batch_size"):
                # TensorDict: zero out entire cache
                if hasattr(self.cache_data, "zero_"):
                    self.cache_data.zero_()
            else:
                # Regular dict: zero out all tensors
                cache_data = self.cache_data
                if isinstance(cache_data, dict):
                    for data_type in cache_data:
                        for cache_key in cache_data[data_type]:
                            cache_data[data_type][cache_key].fill_(0)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about cache usage.

        Returns:
            Dict with cache statistics
        """
        return {
            "num_cached_environments": self.cache_valid.sum().item(),
            "cache_window_size": self.window_size,
            "cached_env_ids": [i for i in range(self.max_envs) if self.cache_valid[i]],
        }
