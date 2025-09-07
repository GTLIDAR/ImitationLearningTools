import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any
from iltools_core.trajectory import Trajectory
from iltools_core.metadata_schema import DatasetMeta
from torch.utils.data import Dataset as TorchDataset
from iltools_datasets.base_loader import BaseLoader, BaseDataset


class TrajoptLoader(BaseLoader):
    """
    Loads trajectories from a directory of trajectory optimization results.
    """

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self._trajectory_lengths = []
        self._file_list = sorted(list(self.data_path.glob("*.npz")))
        self._dt = 0.02
        self._joint_names = ["joint1", "joint2", "joint3"]
        self._body_names = ["body1", "body2"]
        self._site_names = ["site1", "site2"]
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> DatasetMeta:
        """
        Loads the metadata for the dataset.
        """
        return DatasetMeta(
            name="trajopt_dataset",
            source="trajopt",
            version="1.0.0",
            citation="TODO",
            num_trajectories=len(self._file_list),
            trajectory_lengths=self._trajectory_lengths,
            dt=self._dt,
            keys=["qpos", "qvel", "action"],
            joint_names=self._joint_names,
            body_names=self._body_names,
            site_names=self._site_names,
            metadata={"trajopt": True},
        )

    def __len__(self):
        return len(self._file_list)

    def __getitem__(self, idx: int) -> Trajectory:
        file_path = self._file_list[idx]
        data = np.load(file_path)
        length = data["qpos"].shape[0]
        if len(self._trajectory_lengths) < len(self._file_list):
            self._trajectory_lengths.append(length)
            self._metadata = self._load_metadata()
        return Trajectory(
            observations={"qpos": data["qpos"], "qvel": data["qvel"]},
            actions={"action": data["actions"]},
            rewards=data.get("rewards"),
            dt=data.get("dt").item(),
        )

    def as_dataset(self, **kwargs) -> "TrajoptTrajectoryDataset":
        return TrajoptTrajectoryDataset(self, **kwargs)

    def save(self, path: str, **kwargs) -> None:
        """
        Saves the dataset to a directory using zarr format.
        """
        import zarr
        from zarr.storage import LocalStore
        import os

        # Create zarr store
        store = LocalStore(path)
        root = zarr.group(store=store, overwrite=True)

        # Create dataset group
        ds_group = root.create_group("ds1")
        motion_group = ds_group.create_group("trajopt")

        # Save each trajectory
        for i, file_path in enumerate(self._file_list):
            data = np.load(file_path)
            traj_group = motion_group.create_group(f"traj{i}")

            # Save trajectory data
            qpos_ds = traj_group.create_dataset(
                "qpos", shape=data["qpos"].shape, dtype=np.float32
            )
            qpos_ds[:] = data["qpos"]
            qvel_ds = traj_group.create_dataset(
                "qvel", shape=data["qvel"].shape, dtype=np.float32
            )
            qvel_ds[:] = data["qvel"]
            actions_ds = traj_group.create_dataset(
                "actions", shape=data["actions"].shape, dtype=np.float32
            )
            actions_ds[:] = data["actions"]

            # Save metadata
            if "dt" in data:
                dt_ds = traj_group.create_dataset(
                    "dt", shape=data["dt"].shape, dtype=np.float32
                )
                dt_ds[:] = data["dt"]

        # Save metadata.json
        metadata_path = os.path.join(os.path.dirname(path), "metadata.json")
        metadata_dict = {
            "num_trajectories": len(self._file_list),
            "trajectory_lengths": self._trajectory_lengths,
            "observation_keys": ["qpos", "qvel"],
            "action_keys": ["actions"],
            "window_size": 64,
            "export_control_freq": 50.0,
            "original_frequency": 100.0,
            "effective_frequency": 50.0,
            "dt": [self._dt for _ in range(len(self._file_list))],
        }

        import json

        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f)

    @property
    def metadata(self) -> DatasetMeta:
        """
        Returns the metadata for the dataset.
        """
        return self._metadata


# PyTorch Dataset for TrajoptLoader
class TrajoptTrajectoryDataset(BaseDataset):
    def __init__(self, loader: TrajoptLoader, device="cpu"):
        self.loader = loader
        self.device = device

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, idx):
        traj = self.loader[idx]
        obs = {
            k: torch.from_numpy(v).float().to(self.device)
            for k, v in traj.observations.items()
        }
        actions = {
            k: torch.from_numpy(v).float().to(self.device)
            for k, v in (traj.actions or {}).items()
        }
        return {
            "observations": obs,
            "actions": actions,
            "dt": torch.tensor(traj.dt, dtype=torch.float32, device=self.device),
        }

    @property
    def metadata(self):
        return self.loader.metadata

    def as_loader(self, **kwargs) -> TrajoptLoader:
        return self.loader
