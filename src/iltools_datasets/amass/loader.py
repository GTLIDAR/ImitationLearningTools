import smplx
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from iltools_core.trajectory import Trajectory
from iltools_core.metadata_schema import DatasetMeta
from torch.utils.data import Dataset as TorchDataset
from iltools_datasets.base_loader import BaseLoader


class AmassLoader(BaseLoader):
    """
    Loads trajectories from the AMASS dataset.
    """

    def __init__(self, data_path: str, model_path: str):
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.smplx_model = self._load_smplx_model()
        self._trajectory_lengths = []
        self._metadata = self._load_metadata()
        self._file_list = sorted(list(self.data_path.glob("**/*.npz")))

    def _load_smplx_model(self):
        """
        Loads the SMPL-X model.
        """
        return smplx.create(
            self.model_path,
            model_type="smplx",
            gender="neutral",
            ext="npz",
            num_pca_comps=12,
            create_global_orient=True,
            create_body_pose=True,
            create_betas=True,
            create_left_hand_pose=True,
            create_right_hand_pose=True,
            create_expression=True,
            create_jaw_pose=True,
            create_leye_pose=True,
            create_reye_pose=True,
            create_transl=True,
        )

    def _load_metadata(self) -> DatasetMeta:
        """
        Loads the metadata for the dataset.
        """
        return DatasetMeta(
            name="amass_dataset",
            source="amass",
            version="1.0.0",
            citation="TODO",
            num_trajectories=len(self._file_list),
            observation_keys=["qpos"],
            trajectory_lengths=self._trajectory_lengths,
        )

    def __len__(self):
        return len(self._file_list)

    def __getitem__(self, idx: int) -> Trajectory:
        file_path = self._file_list[idx]
        bdata = np.load(file_path)
        length = bdata["poses"].shape[0]
        if len(self._trajectory_lengths) < len(self._file_list):
            self._trajectory_lengths.append(length)
            self._metadata = self._load_metadata()
        body_pose = torch.from_numpy(bdata["poses"][:, 3:66]).float()
        betas = torch.from_numpy(bdata["betas"]).float()
        global_orient = torch.from_numpy(bdata["poses"][:, :3]).float()
        transl = torch.from_numpy(bdata["trans"]).float()

        smplx_output = self.smplx_model(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl,
        )

        return Trajectory(
            observations={"qpos": smplx_output.joints.detach().numpy()},
            infos={"smplx_output": smplx_output},
            dt=1.0 / bdata["mocap_framerate"].item(),
        )

    def as_dataset(self, **kwargs) -> "AmassTrajectoryDataset":
        return AmassTrajectoryDataset(self, **kwargs)

    @property
    def metadata(self) -> DatasetMeta:
        """
        Returns the metadata for the dataset.
        """
        return self._metadata


# PyTorch Dataset for AmassLoader
class AmassTrajectoryDataset(BaseDataset):
    def __init__(self, loader: AmassLoader, device="cpu"):
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
        return {
            "observations": obs,
            "dt": torch.tensor(traj.dt, dtype=torch.float32, device=self.device),
        }

    @property
    def metadata(self):
        return self.loader.metadata

    def as_loader(self, **kwargs) -> AmassLoader:
        return self.loader
