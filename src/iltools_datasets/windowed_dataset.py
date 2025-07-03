from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from typing import Dict, Tuple, Sequence, Any, Optional, List
from functools import lru_cache
import concurrent.futures
import os
import json
import logging
import numpy as np
import zarr
from functools import lru_cache

import torch
from tensordict import TensorDict

from .dataset_types import ZarrBackedTrajectoryDataset

logger = logging.getLogger("iltools_datasets.windowed_dataset")


class _AsyncLRUCache:
    """Thread-safe LRU cache with async prefetching capabilities."""

    def __init__(self, capacity: int = 2048):
        self.capacity = capacity
        self._lock = threading.RLock()
        self._store: OrderedDict[Tuple[int, int], TensorDict] = OrderedDict()

        # Async prefetching
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._prefetch_futures: Dict[Tuple[int, int], concurrent.futures.Future] = {}

    def get(self, key: Tuple[int, int]) -> Optional[TensorDict]:
        with self._lock:
            value = self._store.get(key)
            if value is not None:
                self._store.move_to_end(key)
                return value

            # Check if we have a pending prefetch
            future = self._prefetch_futures.get(key)
            if future is not None:
                try:
                    # Wait for prefetch to complete (should be ready)
                    value = future.result(timeout=0.1)
                    self._store[key] = value
                    self._store.move_to_end(key)
                    del self._prefetch_futures[key]
                    self._trim_cache()
                    return value
                except concurrent.futures.TimeoutError:
                    pass  # Fallback to synchronous loading

            return None

    def put(self, key: Tuple[int, int], value: TensorDict) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            self._trim_cache()

    def prefetch_async(self, key: Tuple[int, int], load_func) -> None:
        """Submit async prefetch request."""
        with self._lock:
            if key not in self._store and key not in self._prefetch_futures:
                future = self._executor.submit(load_func, *key)
                self._prefetch_futures[key] = future

    def _trim_cache(self) -> None:
        """Remove oldest entries to maintain capacity."""
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            # Cancel pending prefetches
            for future in self._prefetch_futures.values():
                future.cancel()
            self._prefetch_futures.clear()


class WindowedTrajectoryDataset(ZarrBackedTrajectoryDataset):
    """
    High-performance windowed trajectory dataset with:
    - Preallocated memory for batch operations
    - Async prefetching with guarantees
    - Vectorized zarr loading
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int = 64,
        device: str | torch.device = "cpu",
        cache_capacity: int = 2048,
        prefetch_ahead: int = 3,
        max_batch_size: int = 512,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            window_size=window_size,
            device=str(device),
        )

        self.window_size = window_size
        self.device = torch.device(device)
        self.prefetch_ahead = prefetch_ahead
        self.max_batch_size = max_batch_size

        # Enhanced cache with async capabilities
        self._cache = _AsyncLRUCache(capacity=cache_capacity)

        # Preallocated memory for batch operations
        self._preallocated_buffers = self._setup_preallocated_buffers()

        # Track data shapes for efficient allocation
        self._obs_shapes = self._get_observation_shapes()
        self._act_shapes = self._get_action_shapes()

    def _setup_preallocated_buffers(self) -> Dict[str, torch.Tensor]:
        """Setup preallocated tensors for batch operations."""
        buffers = {}

        # We'll initialize these lazily when we know the actual shapes
        # This is just a placeholder structure
        return buffers

    def _get_observation_shapes(self) -> Dict[str, tuple]:
        """Get shapes of observation tensors by sampling."""
        shapes = {}
        try:
            # Sample first trajectory to get shapes
            for key in self.observation_keys:
                sample_data = self.root[f"observations/{key}"][0, 0:1]  # First timestep
                shapes[key] = sample_data.shape[1:]  # Exclude time dimension
        except Exception:
            # Fallback shapes - will be corrected during first load
            shapes = {key: (1,) for key in self.observation_keys}
        return shapes

    def _get_action_shapes(self) -> Dict[str, tuple]:
        """Get shapes of action tensors by sampling."""
        shapes = {}
        try:
            for key in self.action_keys:
                sample_data = self.root[f"actions/{key}"][0, 0:1]
                shapes[key] = sample_data.shape[1:]
        except Exception:
            shapes = {key: (1,) for key in self.action_keys}
        return shapes

    def batch_get(self, traj_indices, step_indices, key=None, data_type="observations"):
        """
        Optimized batch get with preallocated memory and vectorized operations.
        """
        if len(traj_indices) != len(step_indices):
            raise ValueError("traj_indices and step_indices must have same length")

        # Convert to numpy for efficient indexing
        if isinstance(traj_indices, torch.Tensor):
            traj_indices = traj_indices.cpu().numpy()
        if isinstance(step_indices, torch.Tensor):
            step_indices = step_indices.cpu().numpy()

        batch_size = len(traj_indices)

        # For single key requests, use optimized path
        if key is not None:
            return self._batch_get_single_key(
                traj_indices, step_indices, key, data_type
            )

        # For small batches, use cached approach
        if batch_size <= 32:
            return self._batch_get_cached(traj_indices, step_indices)

        # For large batches, use vectorized approach
        return self._batch_get_vectorized(traj_indices, step_indices)

    def _batch_get_single_key(
        self,
        traj_indices: np.ndarray,
        step_indices: np.ndarray,
        key: str,
        data_type: str,
    ) -> torch.Tensor:
        """Optimized path for single key extraction."""
        zarr_key = f"{data_type}/{key}"
        if zarr_key not in self.root:
            raise KeyError(f"Key {zarr_key} not found in dataset")

        batch_size = len(traj_indices)

        # Get shape from first sample
        first_sample = self.root[zarr_key][
            traj_indices[0], step_indices[0] : step_indices[0] + 1
        ]
        sample_shape = first_sample.shape[1:]  # Exclude time dimension

        # Preallocate result tensor
        result_shape = [batch_size] + list(sample_shape)
        result = torch.zeros(result_shape, dtype=torch.float32, device=self.device)

        # Vectorized loading where possible
        for i in range(batch_size):
            traj_idx, step_idx = traj_indices[i], step_indices[i]

            # Clamp step_idx to valid range
            traj_len = self.lengths[traj_idx]
            step_idx = min(step_idx, traj_len - 1)

            data = self.root[zarr_key][traj_idx, step_idx : step_idx + 1]
            result[i] = (
                torch.from_numpy(data[0]).float().to(self.device, non_blocking=True)
            )

        return result

    def _batch_get_cached(
        self, traj_indices: np.ndarray, step_indices: np.ndarray
    ) -> TensorDict:
        """Use cached windows for small batches."""
        batch_windows = []

        # Trigger async prefetching for upcoming windows
        self._async_prefetch_batch(traj_indices, step_indices)

        for traj_idx, step_idx in zip(traj_indices, step_indices):
            window = self._fetch_window_with_cache(int(traj_idx), int(step_idx))
            batch_windows.append(window)

        return self._stack_windows_preallocated(batch_windows)

    def _batch_get_vectorized(
        self, traj_indices: np.ndarray, step_indices: np.ndarray
    ) -> TensorDict:
        """Vectorized loading for large batches - bypass cache."""
        batch_size = len(traj_indices)

        # Group by trajectory for efficient zarr access
        traj_groups = {}
        for i, (traj_idx, step_idx) in enumerate(zip(traj_indices, step_indices)):
            if traj_idx not in traj_groups:
                traj_groups[traj_idx] = []
            traj_groups[traj_idx].append((i, step_idx))

        # Preallocate result tensors
        result_dict = {}

        # Load observations
        for obs_key in self.observation_keys:
            zarr_key = f"observations/{obs_key}"
            obs_shape = self._obs_shapes[obs_key]
            result_dict[zarr_key] = torch.zeros(
                (batch_size, self.window_size, *obs_shape),
                dtype=torch.float32,
                device=self.device,
            )

        # Load actions
        for act_key in self.action_keys:
            zarr_key = f"actions/{act_key}"
            act_shape = self._act_shapes[act_key]
            result_dict[zarr_key] = torch.zeros(
                (batch_size, self.window_size, *act_shape),
                dtype=torch.float32,
                device=self.device,
            )

        # Fill data by trajectory groups
        for traj_idx, batch_items in traj_groups.items():
            traj_len = self.lengths[traj_idx]

            for batch_pos, step_idx in batch_items:
                # Clamp to valid window
                start_idx = (
                    min(step_idx, traj_len - self.window_size)
                    if traj_len >= self.window_size
                    else 0
                )
                end_idx = start_idx + self.window_size

                # Load all keys for this window
                for obs_key in self.observation_keys:
                    zarr_key = f"observations/{obs_key}"
                    data = self.root[zarr_key][traj_idx, start_idx:end_idx]

                    # Handle padding if needed
                    if data.shape[0] < self.window_size:
                        padded = np.zeros(
                            (self.window_size, *data.shape[1:]), dtype=data.dtype
                        )
                        padded[: data.shape[0]] = data
                        if data.shape[0] > 0:
                            # Repeat last frame for padding
                            padded[data.shape[0] :] = data[-1]
                        data = padded

                    result_dict[zarr_key][batch_pos] = (
                        torch.from_numpy(data)
                        .float()
                        .to(self.device, non_blocking=True)
                    )

                for act_key in self.action_keys:
                    zarr_key = f"actions/{act_key}"
                    data = self.root[zarr_key][traj_idx, start_idx:end_idx]

                    if data.shape[0] < self.window_size:
                        padded = np.zeros(
                            (self.window_size, *data.shape[1:]), dtype=data.dtype
                        )
                        padded[: data.shape[0]] = data
                        if data.shape[0] > 0:
                            padded[data.shape[0] :] = data[-1]
                        data = padded

                    result_dict[zarr_key][batch_pos] = (
                        torch.from_numpy(data)
                        .float()
                        .to(self.device, non_blocking=True)
                    )

        return TensorDict(result_dict, batch_size=[batch_size])

    def _stack_windows_preallocated(
        self, batch_windows: list[TensorDict]
    ) -> TensorDict:
        """Stack windows using preallocated memory."""
        if not batch_windows:
            raise ValueError("Empty batch_windows")

        batch_size = len(batch_windows)
        first_window = batch_windows[0]

        # Preallocate result dict
        result_dict = {}

        for key in first_window.keys():
            tensor_shape = first_window[key].shape
            stacked_shape = (batch_size, *tensor_shape)

            # Preallocate and fill
            stacked_tensor = torch.zeros(
                stacked_shape, dtype=torch.float32, device=self.device
            )
            for i, window in enumerate(batch_windows):
                stacked_tensor[i] = window[key].to(self.device, non_blocking=True)

            result_dict[key] = stacked_tensor

        return TensorDict(result_dict, batch_size=[batch_size])

    def _fetch_window_with_cache(self, traj_idx: int, start_idx: int) -> TensorDict:
        """Fetch window with cache, fallback to direct load."""
        # Clamp window to trajectory bounds
        traj_len = self.lengths[traj_idx]
        if start_idx >= traj_len:
            start_idx = max(0, traj_len - 1)
        if start_idx + self.window_size > traj_len:
            start_idx = max(0, traj_len - self.window_size)

        key = (traj_idx, start_idx)

        # Try cache first
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        # Load synchronously
        window = self._load_window_direct(traj_idx, start_idx)
        window["dt"] = torch.tensor(
            float(self.dt_list[traj_idx]), dtype=torch.float32, device=self.device
        )
        self._cache.put(key, window)
        return window

    def _load_window_direct(self, traj_idx: int, start_idx: int) -> TensorDict:
        """Direct window loading without cache."""
        end_idx = start_idx + self.window_size
        window_dict = {}

        # Load observations
        for key in self.observation_keys:
            data = self.root[f"observations/{key}"][traj_idx, start_idx:end_idx]
            tensor = torch.from_numpy(data).float().to(self.device, non_blocking=True)

            # Pad if needed
            if tensor.shape[0] < self.window_size:
                pad_len = self.window_size - tensor.shape[0]
                if tensor.shape[0] > 0:
                    pad = tensor[-1:].repeat(pad_len, *([1] * (tensor.ndim - 1)))
                    tensor = torch.cat([tensor, pad], dim=0)
                else:
                    # Zero tensor if no data
                    tensor = torch.zeros(
                        (self.window_size, *tensor.shape[1:]),
                        dtype=tensor.dtype,
                        device=self.device,
                    )

            window_dict[f"observations/{key}"] = tensor

        # Load actions
        for key in self.action_keys:
            data = self.root[f"actions/{key}"][traj_idx, start_idx:end_idx]
            tensor = torch.from_numpy(data).float().to(self.device, non_blocking=True)

            if tensor.shape[0] < self.window_size:
                pad_len = self.window_size - tensor.shape[0]
                if tensor.shape[0] > 0:
                    pad = tensor[-1:].repeat(pad_len, *([1] * (tensor.ndim - 1)))
                    tensor = torch.cat([tensor, pad], dim=0)
                else:
                    tensor = torch.zeros(
                        (self.window_size, *tensor.shape[1:]),
                        dtype=tensor.dtype,
                        device=self.device,
                    )

            window_dict[f"actions/{key}"] = tensor

        window = TensorDict(window_dict, batch_size=[self.window_size])
        window["dt"] = torch.tensor(
            float(self.dt_list[traj_idx]), dtype=torch.float32, device=self.device
        )
        return window

    def _async_prefetch_batch(
        self, traj_indices: np.ndarray, step_indices: np.ndarray
    ) -> None:
        """Async prefetch for upcoming windows."""
        for traj_idx, step_idx in zip(traj_indices, step_indices):
            # Prefetch next few windows
            for offset in range(1, self.prefetch_ahead + 1):
                future_step = step_idx + offset
                future_key = (int(traj_idx), int(future_step))

                # Submit async prefetch
                self._cache.prefetch_async(future_key, self._load_window_direct)

    def prefetch(
        self, traj_indices: Sequence[int], step_indices: Sequence[int]
    ) -> None:
        """Public prefetch interface."""
        if isinstance(traj_indices, torch.Tensor):
            traj_indices = traj_indices.cpu().numpy()
        if isinstance(step_indices, torch.Tensor):
            step_indices = step_indices.cpu().numpy()

        self._async_prefetch_batch(traj_indices, step_indices)


class WindowedTrajectoryDataset:
    """Efficient windowed dataset that works with per-trajectory Zarr format.

    Provides sliding windows over trajectories without loading entire trajectories into memory.
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int,
        stride: int = 1,
        device: str = "cpu",
        cache_size: int = 128,
    ):
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride
        self.device = device
        self.cache_size = cache_size

        # Load metadata
        with open(os.path.join(data_dir, "metadata.json"), "r") as f:
            self._metadata = json.load(f)

        # Check format
        if self._metadata.get("export_format") != "per_trajectory_zarr":
            raise ValueError(
                "WindowedTrajectoryDataset only supports per_trajectory_zarr format"
            )

        self.lengths = self._metadata["trajectory_lengths"]
        if isinstance(self.lengths, int):
            self.lengths = [self.lengths] * self._metadata["num_trajectories"]

        self.observation_keys = self._metadata["observation_keys"]
        self.action_keys = self._metadata["action_keys"] or []
        self.dt_list = self._metadata["dt"]

        # Open Zarr store
        zarr_path = os.path.join(data_dir, "trajectories.zarr")
        store = zarr.DirectoryStore(zarr_path)
        self.root = zarr.open_group(store=store, mode="r")

        # Precompute window indices
        self._build_window_index()

    def _build_window_index(self):
        """Build an index of all valid windows across all trajectories."""
        self.window_index = []  # List of (traj_idx, start_idx) tuples

        for traj_idx, traj_length in enumerate(self.lengths):
            # Calculate valid starting positions for this trajectory
            max_start = max(0, traj_length - self.window_size + 1)
            for start_idx in range(0, max_start, self.stride):
                self.window_index.append((traj_idx, start_idx))

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, idx: int) -> TensorDict:
        """Get a window by its global index."""
        traj_idx, start_idx = self.window_index[idx]
        return self._get_window(traj_idx, start_idx)

    @lru_cache(maxsize=None)  # Cache is bounded by cache_size via _get_window_cached
    def _get_window_cached(self, traj_idx: int, start_idx: int) -> TensorDict:
        """Get a window with caching."""
        return self._get_window_direct(traj_idx, start_idx)

    def _get_window(self, traj_idx: int, start_idx: int) -> TensorDict:
        """Get a window, using cache if cache_size > 0."""
        if self.cache_size > 0:
            # Manage cache size manually
            if len(self._get_window_cached.cache_info().currsize) >= self.cache_size:
                self._get_window_cached.cache_clear()
            return self._get_window_cached(traj_idx, start_idx)
        else:
            return self._get_window_direct(traj_idx, start_idx)

    def _get_window_direct(self, traj_idx: int, start_idx: int) -> TensorDict:
        """Load a window directly from Zarr (no caching)."""
        traj_group = self.root[f"traj_{traj_idx:04d}"]
        end_idx = start_idx + self.window_size

        # Load observations window
        obs_dict = {}
        obs_group = traj_group["observations"]
        for key in self.observation_keys:
            obs_dict[key] = torch.tensor(
                np.array(obs_group[key][start_idx:end_idx]),
                dtype=torch.float32,
                device=self.device,
            )

        # Load actions window (if present)
        act_dict = {}
        if self.action_keys and "actions" in traj_group:
            act_group = traj_group["actions"]
            for key in self.action_keys:
                act_dict[key] = torch.tensor(
                    np.array(act_group[key][start_idx:end_idx]),
                    dtype=torch.float32,
                    device=self.device,
                )

        return TensorDict(
            {
                **obs_dict,
                **act_dict,
                "dt": torch.tensor(
                    float(self.dt_list[traj_idx]),
                    dtype=torch.float32,
                    device=self.device,
                ),
                "traj_idx": torch.tensor(
                    traj_idx, dtype=torch.long, device=self.device
                ),
                "start_idx": torch.tensor(
                    start_idx, dtype=torch.long, device=self.device
                ),
            }
        )

    def get_trajectory_windows(self, traj_idx: int) -> List[TensorDict]:
        """Get all windows for a specific trajectory."""
        windows = []
        traj_length = self.lengths[traj_idx]
        max_start = max(0, traj_length - self.window_size + 1)

        for start_idx in range(0, max_start, self.stride):
            windows.append(self._get_window(traj_idx, start_idx))

        return windows

    def clear_cache(self):
        """Clear the window cache."""
        if hasattr(self, "_get_window_cached"):
            self._get_window_cached.cache_clear()

    def get_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return {
            "num_trajectories": len(self.lengths),
            "trajectory_lengths": self.lengths,
            "window_size": self.window_size,
            "stride": self.stride,
            "total_windows": len(self),
            "observation_keys": self.observation_keys,
            "action_keys": self.action_keys,
            "dt_list": self.dt_list,
        }


class LegacyWindowedTrajectoryDataset:
    """Legacy windowed dataset for backward compatibility with old Zarr format."""

    def __init__(
        self,
        data_dir: str,
        window_size: int,
        stride: int = 1,
        device: str = "cpu",
    ):
        raise NotImplementedError(
            "Legacy windowed dataset not implemented. Please export your data using "
            "export_trajectories_to_zarr_per_trajectory to use the new format."
        )
