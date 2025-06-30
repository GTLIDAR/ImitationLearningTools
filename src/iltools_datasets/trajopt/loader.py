import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from iltools_core.trajectory import Trajectory
from iltools_core.metadata_schema import DatasetMeta
from torch.utils.data import Dataset as TorchDataset
from iltools_datasets.base_loader import BaseTrajectoryLoader, BaseTrajectoryDataset


class TrajoptLoader(BaseTrajectoryLoader):
    """
    Loads trajectories from a directory of trajectory optimization results.
    """

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self._trajectory_lengths = []
        self._metadata = self._load_metadata()
        self._file_list = sorted(list(self.data_path.glob("*.npz")))

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
            observation_keys=["qpos", "qvel"],
            action_keys=["action"],
            trajectory_lengths=self._trajectory_lengths,
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

    @property
    def metadata(self) -> DatasetMeta:
        """
        Returns the metadata for the dataset.
        """
        return self._metadata


# PyTorch Dataset for TrajoptLoader
class TrajoptTrajectoryDataset(BaseTrajectoryDataset):
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
