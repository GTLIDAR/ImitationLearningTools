import os
import json
import queue
import threading
from functools import lru_cache

import numpy as np
import torch
import zarr
from iltools_datasets.base_loader import BaseTrajectoryDataset


class DiskBackedTrajectoryDataset(BaseTrajectoryDataset):
    def __init__(self, data_dir, cache_size=128, device="cpu"):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, "metadata.json"), "r") as f:
            self._metadata = json.load(f)
        self.lengths = self._metadata["trajectory_lengths"]
        if isinstance(self.lengths, int):
            self.lengths = [self.lengths] * self._metadata["num_trajectories"]
        self.observation_keys = self._metadata["observation_keys"]
        self.action_keys = self._metadata["action_keys"] or []
        self.device = device
        self._get_traj = lru_cache(maxsize=cache_size)(self._get_traj_uncached)

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        d = self._get_traj(idx)
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
        from iltools_core.metadata_schema import DatasetMeta

        return DatasetMeta(**self._metadata)

    def as_loader(self, **kwargs):
        raise NotImplementedError(
            "DiskBackedTrajectoryDataset cannot be converted to a loader directly."
        )


class ZarrBackedTrajectoryDataset(BaseTrajectoryDataset):
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
            self._metadata = json.load(f)
        self.lengths = self._metadata["trajectory_lengths"]
        if isinstance(self.lengths, int):
            self.lengths = [self.lengths] * self._metadata["num_trajectories"]
        self.observation_keys = self._metadata["observation_keys"]
        self.action_keys = self._metadata["action_keys"] or []
        self.device = device
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.prefetch_batches = prefetch_batches
        if self.zarr_path is None or not isinstance(self.zarr_path, str):
            raise RuntimeError(
                f"self.zarr_path must be a non-None string, got {self.zarr_path}"
            )
        self.root = zarr.open(self.zarr_path, mode="r")
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
        self._queue = queue.Queue(maxsize=prefetch_batches)
        self._stop_event = threading.Event()
        self._next_idx = 0
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_loop, daemon=True
        )
        self._prefetch_thread.start()

    def __len__(self):
        return self.length

    def _prefetch_loop(self):
        while not self._stop_event.is_set():
            if self._queue.full():
                self._stop_event.wait(0.01)
                continue
            batch = []
            for _ in range(self.batch_size):
                if self._next_idx >= self.length:
                    self._next_idx = 0
                try:
                    sample = self._get_item(self._next_idx)
                    batch.append(sample)
                except Exception as e:
                    pass
                self._next_idx += 1
            if batch:
                self._queue.put(batch)

    def __getitem__(self, idx):
        return self._get_item(idx)

    def get_batch(self):
        try:
            batch = self._queue.get(timeout=5)
            return self._collate_batch(batch)
        except queue.Empty:
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
        dt_val = self.root["dt"][int(idx)]
        dt = float(np.asarray(dt_val))
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

    def as_loader(self, **kwargs):
        raise NotImplementedError(
            "ZarrBackedTrajectoryDataset cannot be converted to a loader directly."
        )

    @property
    def metadata(self):
        from iltools_core.metadata_schema import DatasetMeta

        return DatasetMeta(**self._metadata)
