import concurrent.futures
import json
import logging
import os
import queue
import threading
from functools import lru_cache
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np

import torch
import zarr
from loco_mujoco.task_factories import (
    DefaultDatasetConf,
    ImitationFactory,
    LAFAN1DatasetConf,
    AMASSDatasetConf,
)
from loco_mujoco.trajectory.dataclasses import interpolate_trajectories
from loco_mujoco.trajectory.handler import TrajectoryHandler
from torch.utils.data import Dataset as TorchDataset

from iltools_core.metadata_schema import DatasetMeta
from iltools_core.trajectory import Trajectory as ILTTrajectory
from iltools_datasets.base_loader import (
    BaseTrajectoryDataset,
    BaseTrajectoryLoader,
)

# --- Set up logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("loco_mujoco.loader")


class LocoMuJoCoLoader(BaseTrajectoryLoader):
    """
    Flexible loader for Loco-MuJoCo trajectories.
    """

    def __init__(self, env_name: str, task: str = "walk"):
        self.env_name = env_name
        self.task = task
        self._setup_cache()
        self.env = self._load_env()
        assert hasattr(self.env, "th") and isinstance(self.env.th, TrajectoryHandler), (
            "TrajectoryHandler not found in env"
        )
        self.th: TrajectoryHandler = self.env.th
        self._metadata = self._discover_metadata()

    def _setup_cache(self):
        cache_path = os.path.expanduser("~/.loco-mujoco-caches")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        os.environ["LOCO_MUJOCO_CACHE"] = cache_path

    def _load_env(self):
        try:
            return ImitationFactory.make(
                self.env_name,
                default_dataset_conf=DefaultDatasetConf([self.task]),
                n_substeps=20,
            )
        except Exception as e:
            if "MyoSkeleton" in self.env_name:
                print(
                    "Error loading MyoSkeleton environment. Did you run 'loco-mujoco-myomodel-init'?"
                )
            raise e

    def _discover_metadata(self) -> DatasetMeta:
        # Discover available observation/action keys from the first trajectory
        # (Assume all trajectories have the same keys for now, but can be extended)
        assert isinstance(self.th, TrajectoryHandler), "TrajectoryHandler not found"
        # Get available keys from the TrajectoryData fields
        data_fields = list(vars(self.th.traj.data).keys())
        obs_keys = [
            k
            for k in data_fields
            if not k.startswith("_") and getattr(self.th.traj.data, k).size > 0
        ]
        # Try to detect action keys (if present)
        action_keys = None
        if (
            hasattr(self.th.traj, "transitions")
            and self.th.traj.transitions is not None
        ):
            if getattr(self.th.traj.transitions, "actions", None) is not None:
                action_keys = ["actions"]

        return DatasetMeta(
            name=f"loco_mujoco_{self.env_name}_{self.task}",
            source="loco_mujoco",
            version="1.0.1",
            citation="TODO",
            num_trajectories=self.th.n_trajectories,
            observation_keys=obs_keys,
            action_keys=action_keys,
            trajectory_lengths=[
                self.th.len_trajectory(traj_ind)
                for traj_ind in range(self.th.n_trajectories)
            ],
        )

    @property
    def metadata(self) -> DatasetMeta:
        return self._metadata

    def __len__(self) -> int:
        return self.th.n_trajectories

    def __getitem__(
        self, idx: int, control_freq: Optional[float] = None
    ) -> ILTTrajectory:
        """
        Returns a single trajectory as an ImitationLearningTools Trajectory.
        Optionally interpolates to the desired control frequency.
        """
        # Get the slice for this trajectory
        start = self.th.traj.data.split_points[idx]
        end = self.th.traj.data.split_points[idx + 1]
        length = end - start

        # Optionally interpolate
        if control_freq is not None and control_freq != self.th.traj.info.frequency:
            # Interpolate using loco-mujoco utility
            new_data, new_info = interpolate_trajectories(
                self.th.traj.data, self.th.traj.info, control_freq
            )
            # Slice the interpolated data
            data = new_data
            info = new_info
        else:
            data = self.th.traj.data
            info = self.th.traj.info

        # Extract all available keys for this trajectory
        obs = {}
        for key in self.metadata.observation_keys:
            arr = getattr(data, key, None)
            if arr is not None and arr.size > 0:
                obs[key] = arr[start:end]

        # Actions (if available)
        actions = None
        if self.metadata.action_keys:
            actions = {}
            # Try to get actions from transitions if present
            if (
                hasattr(self.th.traj, "transitions")
                and self.th.traj.transitions is not None
            ):
                arr = getattr(self.th.traj.transitions, "actions", None)
                if arr is not None and arr.size > 0:
                    actions["actions"] = arr[start:end]

        # dt (time step)
        dt = 1.0 / (control_freq if control_freq is not None else info.frequency)

        return ILTTrajectory(
            observations=obs,
            actions=actions,
            dt=dt,
        )

    def iter_trajectories(
        self, control_freq: Optional[float] = None
    ) -> Iterator[ILTTrajectory]:
        for idx in range(len(self)):
            yield self.__getitem__(idx, control_freq=control_freq)

    def load(self) -> List[ILTTrajectory]:
        """
        Loads all trajectories (not recommended for large datasets).
        """
        return [self.__getitem__(i) for i in range(len(self))]

    def collate_batch(
        self, indices: List[int], control_freq: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Collate a batch of trajectories, padding as needed.
        Returns a dict with 'observations', 'actions', 'lengths', and 'mask'.
        """
        batch = [self.__getitem__(i, control_freq=control_freq) for i in indices]
        # Find max length
        max_len = int(
            np.max(
                [
                    traj.observations[self.metadata.observation_keys[0]].shape[0]
                    for traj in batch
                ]
            )
        )
        obs_keys = self.metadata.observation_keys
        action_keys = self.metadata.action_keys or []
        # Pad observations and actions
        obs_batch = {k: [] for k in obs_keys}
        act_batch = {k: [] for k in action_keys}
        lengths = np.array(
            [traj.observations[obs_keys[0]].shape[0] for traj in batch], dtype=np.int64
        )
        for traj in batch:
            l = traj.observations[obs_keys[0]].shape[0]
            for k in obs_keys:
                arr = traj.observations.get(k, np.zeros((l,)))
                arr = np.asarray(arr)
                # pad_width must be a list of tuples of Python ints
                pad_width = [(0, int(max_len - l))] + [
                    (0, 0) for _ in range(arr.ndim - 1)
                ]
                obs_batch[k].append(np.pad(arr, pad_width, mode="constant"))  # type: ignore
            for k in action_keys:
                arr = None
                if traj.actions is not None:
                    arr = traj.actions.get(k, np.zeros((l,)))
                if arr is None:
                    arr = np.zeros((l,))
                arr = np.asarray(arr)
                pad_width = [(0, int(max_len - l))] + [
                    (0, 0) for _ in range(arr.ndim - 1)
                ]
                act_batch[k].append(np.pad(arr, pad_width, mode="constant"))  # type: ignore
        # Stack
        obs_batch = {k: np.stack(v) for k, v in obs_batch.items()}
        act_batch = (
            {k: np.stack(v) for k, v in act_batch.items()} if action_keys else None
        )
        # Ensure mask creation uses compatible dtypes (int64) and use np.less for static analysis
        mask = np.less(
            np.arange(max_len, dtype=np.int64)[None, :], lengths[:, None]
        ).astype(bool)  # type: ignore
        return {
            "observations": obs_batch,
            "actions": act_batch,
            "lengths": lengths,
            "mask": mask,
        }

    def as_dataset(self, **kwargs) -> "LocoMuJoCoTrajectoryDataset":
        return LocoMuJoCoTrajectoryDataset(self, **kwargs)


class LocoMuJoCoTrajectoryIndexDataset(BaseTrajectoryDataset):
    """
    PyTorch Dataset that provides trajectory indices and metadata for sampling.
    """

    def __init__(self, loader: LocoMuJoCoLoader):
        self.loader = loader
        lengths = loader.metadata.trajectory_lengths
        if isinstance(lengths, int):
            # If a single int, repeat for all trajectories
            self.lengths = [lengths] * loader.metadata.num_trajectories
        else:
            self.lengths = list(lengths)

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        return {
            "traj_idx": idx,
            "length": self.lengths[idx],
            # Add more metadata if needed
        }

    @property
    def metadata(self):
        return self.loader.metadata

    def as_loader(self, **kwargs):
        return self.loader


class LocoMuJoCoTrajectoryAccessor:
    """
    Provides fast, cached access to trajectories as torch tensors.
    """

    def __init__(self, loader: LocoMuJoCoLoader, cache_size: int = 128):
        self.loader = loader
        self._get_traj = lru_cache(maxsize=cache_size)(self._get_traj_uncached)

    def _get_traj_uncached(self, idx):
        traj = self.loader[idx]
        obs = {k: torch.from_numpy(v).float() for k, v in traj.observations.items()}
        actions = {
            k: torch.from_numpy(v).float() for k, v in (traj.actions or {}).items()
        }
        return {
            "observations": obs,
            "actions": actions,
            "dt": torch.tensor(traj.dt, dtype=torch.float32),
        }

    def get(self, idx):
        return self._get_traj(idx)


# --- Export utility ---
def export_trajectories_to_disk(loader: LocoMuJoCoLoader, out_dir: str):
    """
    Export all trajectories from a LocoMuJoCoLoader to disk as .npz files, with a metadata.json file.
    """
    os.makedirs(out_dir, exist_ok=True)
    metadata = {
        "num_trajectories": len(loader),
        "trajectory_lengths": loader.metadata.trajectory_lengths,
        "observation_keys": loader.metadata.observation_keys,
        "action_keys": loader.metadata.action_keys,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    for idx in range(len(loader)):
        traj = loader[idx]
        # Ensure all values are numpy arrays with compatible dtype
        save_dict = {
            **{k: np.asarray(v) for k, v in traj.observations.items()},
            **{k: np.asarray(v) for k, v in (traj.actions or {}).items()},
            "dt": np.array(traj.dt, dtype=np.float32),
        }
        np.savez(
            os.path.join(out_dir, f"traj_{idx}.npz"),
            **save_dict,  # type: ignore
        )


# --- Disk-backed PyTorch Dataset ---
class DiskBackedTrajectoryDataset(BaseTrajectoryDataset):
    """
    PyTorch Dataset that loads only metadata into memory, and loads each trajectory from disk as needed.
    Uses an LRU cache for fast repeated access.
    """

    def __init__(self, data_dir, cache_size=128, device="cpu"):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
        self.lengths = self.metadata["trajectory_lengths"]
        if isinstance(self.lengths, int):
            self.lengths = [self.lengths] * self.metadata["num_trajectories"]
        self.observation_keys = self.metadata["observation_keys"]
        self.action_keys = self.metadata["action_keys"] or []
        self.device = device
        self._get_traj = lru_cache(maxsize=cache_size)(self._get_traj_uncached)

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        d = self._get_traj(idx)
        # Move to device
        d["observations"] = {k: v.to(self.device) for k, v in d["observations"].items()}
        d["actions"] = {k: v.to(self.device) for k, v in d["actions"].items()}
        d["dt"] = d["dt"].to(self.device)
        return d

    def _get_traj_uncached(self, idx):
        path = os.path.join(self.data_dir, f"traj_{idx}.npz")
        data = np.load(path)
        obs = {
            k: torch.from_numpy(data[k]).float()
            for k in self.observation_keys
            if k in data
        }
        actions = {
            k: torch.from_numpy(data[k]).float() for k in self.action_keys if k in data
        }
        dt = float(data["dt"])
        return {
            "observations": obs,
            "actions": actions,
            "dt": torch.tensor(dt, dtype=torch.float32),
        }

    @property
    def metadata(self):
        return self.metadata

    def as_loader(self, **kwargs):
        raise NotImplementedError(
            "DiskBackedTrajectoryDataset cannot be converted to a loader directly."
        )


# --- Parallel export utility with Zarr support ---
def export_trajectories_to_zarr(
    loader: LocoMuJoCoLoader,
    out_dir: str,
    num_workers: int = 8,
    window_size: Optional[int] = None,
):
    """
    Export all trajectories from a LocoMuJoCoLoader to a Zarr file, with parallel processing and robust error handling.
    If window_size is provided, store fixed-length windows instead of full trajectories.
    """
    os.makedirs(out_dir, exist_ok=True)
    zarr_path = os.path.join(out_dir, "trajectories.zarr")
    # Ensure trajectory_lengths is a list of ints
    traj_lengths = loader.metadata.trajectory_lengths
    if isinstance(traj_lengths, int):
        traj_lengths = [traj_lengths] * loader.metadata.num_trajectories
    else:
        traj_lengths = list(traj_lengths)
    metadata = {
        "num_trajectories": len(loader),
        "trajectory_lengths": traj_lengths,
        "observation_keys": loader.metadata.observation_keys,
        "action_keys": loader.metadata.action_keys,
        "window_size": window_size,
    }
    # Save metadata
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    # Prepare Zarr root as a group
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.open_group(store=store, mode="w")
    # Determine shapes and dtypes from the first trajectory
    try:
        first_traj = loader[0]
        first_traj_len = int(next(iter(first_traj.observations.values())).shape[0])
        obs_shapes = {
            k: first_traj.observations[k].shape[1:]
            for k in loader.metadata.observation_keys
        }
        obs_dtypes = {
            k: first_traj.observations[k].dtype
            for k in loader.metadata.observation_keys
        }
        if loader.metadata.action_keys:
            act_shapes = {
                k: first_traj.actions[k].shape[1:] for k in loader.metadata.action_keys
            }
            act_dtypes = {
                k: first_traj.actions[k].dtype for k in loader.metadata.action_keys
            }
        else:
            act_shapes, act_dtypes = {}, {}
    except Exception as e:
        logger.error(f"Failed to inspect first trajectory: {e}")
        raise
    # Compute total number of windows
    if window_size is not None:
        win_size = int(window_size)
        total_windows = sum(max(1, (int(l) - win_size + 1)) for l in traj_lengths)
    else:
        win_size = int(first_traj_len)
        total_windows = len(loader)
    # Create Zarr arrays in the group
    obs_arrays = {}
    for k in loader.metadata.observation_keys:
        obs_arrays[k] = root.create_dataset(
            f"observations/{k}",
            shape=(int(total_windows), win_size, *obs_shapes[k]),
            dtype=obs_dtypes[k],
            chunks=(min(1024, int(total_windows)), win_size, *obs_shapes[k]),
            compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2),
        )
    act_arrays = {}
    for k in loader.metadata.action_keys or []:
        act_arrays[k] = root.create_dataset(
            f"actions/{k}",
            shape=(int(total_windows), win_size, *act_shapes[k]),
            dtype=act_dtypes[k],
            chunks=(min(1024, int(total_windows)), win_size, *act_shapes[k]),
            compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2),
        )
    dt_array = root.create_dataset(
        "dt",
        shape=(int(total_windows),),
        dtype=np.float32,
        chunks=(min(1024, int(total_windows)),),
        compressor=zarr.Blosc(),
    )

    # Helper to extract windows
    def get_windows(traj, window_size):
        length = int(next(iter(traj.observations.values())).shape[0])
        win_size = window_size or length
        for start in range(0, max(1, length - win_size + 1)):
            obs_win = {
                k: v[start : start + win_size] for k, v in traj.observations.items()
            }
            act_win = {
                k: v[start : start + win_size] for k, v in (traj.actions or {}).items()
            }
            yield obs_win, act_win, traj.dt

    # Worker function
    def process_traj(idx):
        try:
            traj = loader[int(idx)]
            if window_size is not None:
                return list(get_windows(traj, window_size))
            else:
                return [(traj.observations, traj.actions or {}, traj.dt)]
        except Exception as e:
            logger.error(f"Failed to process trajectory {idx}: {e}")
            return []

    # Parallel export
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        fut_to_idx = {
            executor.submit(process_traj, int(idx)): int(idx)
            for idx in range(len(loader))
        }
        for fut in concurrent.futures.as_completed(fut_to_idx):
            idx = int(fut_to_idx[fut])
            try:
                res = fut.result()
                results.append((idx, res))
            except Exception as e:
                logger.error(f"Export failed for trajectory {idx}: {e}")
    # Write to Zarr
    flat_results = []
    for idx, winlist in sorted(results):
        flat_results.extend(winlist)
    for i, (obs, act, dt) in enumerate(flat_results):
        for k, arr in obs.items():
            obs_arrays[k][int(i), ...] = np.asarray(arr)
        for k, arr in act.items():
            act_arrays[k][int(i), ...] = np.asarray(arr)
        dt_array[int(i)] = np.float32(dt)
    logger.info(f"Exported {len(flat_results)} windows/trajectories to {zarr_path}")


# --- Zarr-backed async, batched, device-aware dataset ---
class ZarrBackedTrajectoryDataset(BaseTrajectoryDataset):
    """
    PyTorch Dataset for Zarr-backed trajectory data, supporting windowed/batched access, async prefetch, device-aware loading, and robust error handling.
    """

    def __init__(
        self,
        data_dir,
        window_size=None,
        device="cpu",
        pin_memory=False,
        prefetch_batches=4,
        batch_size=1,
    ):
        self.data_dir = data_dir
        self.zarr_path = os.path.join(data_dir, "trajectories.zarr")
        with open(os.path.join(data_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
        self.window_size = (
            int(window_size)
            if window_size is not None
            else int(self.metadata.get("window_size") or 1)
        )
        self.device = device
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.prefetch_batches = prefetch_batches
        # Ensure self.zarr_path is a string and not None
        if self.zarr_path is None or not isinstance(self.zarr_path, str):
            raise RuntimeError(
                f"self.zarr_path must be a non-None string, got {self.zarr_path}"
            )
        self.root = zarr.open(self.zarr_path, mode="r")
        # Ensure self.length is a Python int
        dt_arr = self.root["dt"]
        shape0 = dt_arr.shape[0]
        if isinstance(shape0, int):
            self.length = shape0
        else:
            try:
                self.length = int(np.asarray(shape0))
            except Exception:
                raise RuntimeError(
                    f"Could not convert dt_arr.shape[0] to int: {shape0}"
                )
        self.observation_keys = self.metadata["observation_keys"]
        self.action_keys = self.metadata["action_keys"] or []
        # Async prefetch queue
        self._queue = queue.Queue(maxsize=prefetch_batches)
        self._stop_event = threading.Event()
        self._next_idx = 0
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_loop, daemon=True
        )
        self._prefetch_thread.start()
        logger.info(
            f"Initialized ZarrBackedTrajectoryDataset with {self.length} windows/trajectories."
        )

    def __len__(self):
        return self.length

    def _prefetch_loop(self):
        while not self._stop_event.is_set():
            if self._queue.full():
                self._stop_event.wait(0.01)
                continue
            batch = []
            for _ in range(self.batch_size):
                # Ensure self._next_idx and self.length are Python ints
                if self._next_idx >= self.length:
                    self._next_idx = 0
                try:
                    sample = self._get_item(self._next_idx)
                    batch.append(sample)
                except Exception as e:
                    logger.error(f"Failed to prefetch idx {self._next_idx}: {e}")
                self._next_idx += 1
            if batch:
                self._queue.put(batch)

    def __getitem__(self, idx):
        # Synchronous fallback if async prefetch is not used
        return self._get_item(idx)

    def get_batch(self):
        # Async batch prefetch
        try:
            batch = self._queue.get(timeout=5)
            return self._collate_batch(batch)
        except queue.Empty:
            logger.warning(
                "Prefetch queue empty, falling back to synchronous batch loading."
            )
            batch = [
                self._get_item((self._next_idx + i) % self.length)
                for i in range(self.batch_size)
            ]
            self._next_idx = (self._next_idx + self.batch_size) % self.length
            return self._collate_batch(batch)

    def _get_item(self, idx):
        obs = {
            k: torch.from_numpy(self.root[f"observations/{k}"][int(idx)]).float()
            for k in self.observation_keys
        }
        actions = {
            k: torch.from_numpy(self.root[f"actions/{k}"][int(idx)]).float()
            for k in self.action_keys
        }
        # Ensure dt is a float
        dt_val = self.root["dt"][int(idx)]
        dt = float(np.asarray(dt_val))
        # Device and pin_memory
        for d in [obs, actions]:
            for k, v in d.items():
                if self.pin_memory:
                    v = v.pin_memory()
                v = v.to(self.device)
                d[k] = v
        return {
            "observations": obs,
            "actions": actions,
            "dt": torch.tensor(dt, dtype=torch.float32, device=self.device),
        }

    def _collate_batch(self, batch):
        obs_keys = self.observation_keys
        act_keys = self.action_keys
        obs_batch = {
            k: torch.stack([sample["observations"][k] for sample in batch])
            for k in obs_keys
        }
        act_batch = (
            {
                k: torch.stack([sample["actions"][k] for sample in batch])
                for k in act_keys
            }
            if act_keys
            else None
        )
        dt_batch = torch.stack([sample["dt"] for sample in batch])
        return {"observations": obs_batch, "actions": act_batch, "dt": dt_batch}

    def shutdown(self):
        self._stop_event.set()
        self._prefetch_thread.join()
        logger.info("ZarrBackedTrajectoryDataset prefetch thread stopped.")

    def as_loader(self, **kwargs):
        raise NotImplementedError(
            "ZarrBackedTrajectoryDataset cannot be converted to a loader directly."
        )
