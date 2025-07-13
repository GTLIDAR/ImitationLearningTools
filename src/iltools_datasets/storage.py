from typing import Optional

import numpy as np
import zarr
from omegaconf import DictConfig

from iltools_datasets.base_loader import BaseDataset


class VectorizedTrajectoryDataset(BaseDataset):
    """Dataset that reads from per-trajectory Zarr format.

    Each trajectory is stored as a separate Zarr group:

    """

    def __init__(self, zarr_path: str, num_envs: int, cfg: DictConfig, **kwargs):
        self.zarr_dataset = zarr.open_group(zarr_path, mode="r")
        self.num_envs = num_envs
        self.cfg = cfg
        self.window_size = cfg.window_size

        self.buffer_size = getattr(cfg, "buffer_size", 128)

        self.env_traj_ids = [-1] * self.num_envs
        self.env_steps = [0] * self.num_envs
        self.window_starts = [0] * self.num_envs
        self.buffers = [{} for _ in range(self.num_envs)]
        self.handlers = {}
        self.traj_lengths = [
            self.zarr_dataset[traj]["qpos"].shape[0]
            for traj in self.available_trajectories
        ]
        self.all_keys = list(self.zarr_dataset[self.available_trajectories[0]].keys())

    @property
    def available_dataset_sources(self) -> list[str]:
        return list(self.zarr_dataset.keys())

    @property
    def available_motions(self) -> list[str]:
        return [
            motion
            for dataset_source in self.available_dataset_sources
            for motion in self.available_motions_in(dataset_source)
        ]

    def available_motions_in(self, dataset_source: str) -> list[str]:
        dataset_source_group = self.zarr_dataset[dataset_source]
        assert isinstance(dataset_source_group, zarr.Group)
        return list(dataset_source_group.keys())

    def available_trajectories_in(self, dataset_source: str, motion: str) -> list[str]:
        dataset_source_group = self.zarr_dataset[dataset_source]
        assert isinstance(dataset_source_group, zarr.Group)
        motion_group = dataset_source_group[motion]
        assert isinstance(motion_group, zarr.Group)
        return list(motion_group.keys())

    @property
    def available_trajectories(self) -> list[str]:
        return [
            f"{dataset_source}/{motion}/{trajectory}"
            for dataset_source in self.available_dataset_sources
            for motion in self.available_motions_in(dataset_source)
            for trajectory in self.available_trajectories_in(dataset_source, motion)
        ]

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

    def fetch(self, idx: list[int], key: Optional[str] = None) -> np.ndarray:
        assert key is not None
        assert len(idx) == self.num_envs

        data_list = []
        for env_id in range(self.num_envs):
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
            if step >= traj_length:
                raise IndexError(
                    f"Step {step} >= traj length {traj_length} for env {env_id}"
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
