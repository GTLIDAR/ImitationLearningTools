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

    def save(
        self,
        out_dir: str,
        *,
        dataset_source: str = "trajopt",
        motion: str | None = None,
        zarr_name: str = "trajectories.zarr",
        max_trajs_per_motion: int | None = None,
    ) -> str:
        """Export trajopt npz directories to Zarr for vectorized access.

        Default behavior (motion=None):
          - Treat `data_path` as a root directory with subdirectories per motion
            (e.g., data_path/<motion>/traj_*.npz or *.npz), and export ALL motions
            into a single Zarr store.

        If `motion` is provided:
          - If `data_path/<motion>` exists, export that subdirectory.
          - Otherwise, treat `data_path` itself as the directory containing the
            motion's .npz files.

        Layout mirrors what `VectorizedTrajectoryDataset` expects:

            <out_dir>/<zarr_name>/<dataset_source>/<motion>/traj{i}/{qpos,qvel,action[,ee_pos]}

        Returns the Zarr path created.
        """
        import zarr
        from zarr.storage import LocalStore
        import os

        os.makedirs(out_dir, exist_ok=True)
        zarr_path = os.path.join(out_dir, zarr_name)
        store = LocalStore(zarr_path)
        # Ensure Zarr v3 format
        root = zarr.group(store=store, overwrite=True, zarr_format=3)

        ds_group = root.create_group(dataset_source)

        def _npz_files_in(directory: Path) -> list[Path]:
            files = sorted(list(directory.glob("traj_*.npz")))
            if not files:
                files = sorted(list(directory.glob("*.npz")))
            return files

        motions_to_export: list[tuple[str, list[Path]]] = []

        if motion is None:
            # enumerate subdirectories under data_path as motions
            for sub in sorted(p for p in self.data_path.iterdir() if p.is_dir()):
                files = _npz_files_in(sub)
                if files:
                    motions_to_export.append((sub.name, files))
        else:
            # specific motion: prefer data_path/<motion>, fallback to data_path
            motion_dir = self.data_path / motion
            if motion_dir.is_dir():
                files = _npz_files_in(motion_dir)
            else:
                files = _npz_files_in(self.data_path)
            motions_to_export.append(
                (motion if motion_dir.is_dir() else "default", files)
            )

        self._trajectory_lengths = []
        for motion_name, file_list in motions_to_export:
            if max_trajs_per_motion is not None:
                file_list = file_list[:max_trajs_per_motion]
            motion_group = ds_group.create_group(motion_name)
            for i, file_path in enumerate(file_list):
                with np.load(file_path) as data:
                    traj_group = motion_group.create_group(f"traj{i}")

                    qpos = np.asarray(data["qpos"], dtype=np.float32)
                    qvel = np.asarray(data["qvel"], dtype=np.float32)
                    action_key = (
                        "action"
                        if "action" in data
                        else ("actions" if "actions" in data else None)
                    )
                    act = (
                        np.asarray(data[action_key], dtype=np.float32)
                        if action_key is not None
                        else None
                    )
                    ee = (
                        np.asarray(data["ee_pos"], dtype=np.float32)
                        if "ee_pos" in data
                        else None
                    )

                    # Create arrays (Zarr v3 API)
                    traj_group.create_array("qpos", shape=qpos.shape, dtype=qpos.dtype)[
                        :
                    ] = qpos
                    traj_group.create_array("qvel", shape=qvel.shape, dtype=qvel.dtype)[
                        :
                    ] = qvel
                    if act is not None:
                        traj_group.create_array(
                            "action", shape=act.shape, dtype=act.dtype
                        )[:] = act
                    if ee is not None:
                        traj_group.create_array(
                            "ee_pos", shape=ee.shape, dtype=ee.dtype
                        )[:] = ee

                    # Track lengths (aggregate over motions)
                    self._trajectory_lengths.append(qpos.shape[0])

        return zarr_path

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
