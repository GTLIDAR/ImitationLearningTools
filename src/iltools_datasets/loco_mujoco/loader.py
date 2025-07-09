import os
from typing import Any

import numpy as np
import zarr
from loco_mujoco.environments import LocoEnv
from loco_mujoco.task_factories import (
    AMASSDatasetConf,
    DefaultDatasetConf,
    ImitationFactory,
    LAFAN1DatasetConf,
)
from loco_mujoco.trajectory.handler import TrajectoryHandler
from omegaconf import DictConfig
from zarr import storage

from iltools_core.metadata_schema import DatasetMeta
from iltools_datasets.base_loader import (
    BaseDataset,
    BaseLoader,
)
from iltools_datasets.storage import VectorizedTrajectoryDataset


class LocoMuJoCoLoader(BaseLoader):
    """
    Flexible loader for Loco-MuJoCo trajectories.
    Now supports multiple datasets/motions (default, lafan1, amass).

    The desired control dt is computed and set through num_substeps when initializing the environment.
    """

    def __init__(
        self,
        env_name: str,
        cfg: DictConfig,
        **kwargs,
    ):
        """cfg can from conf/ or from Isaaclab dataclass"""
        self.cfg = cfg
        self.env_name = env_name
        self.dataset_dict = getattr(
            cfg.dataset,
            "trajectories",
            {"default": ["walk"], "amass": [], "lafan1": []},
        )

        # self._setup_cache()

        self.env: LocoEnv = self._load_env()
        assert hasattr(self.env, "th") and isinstance(self.env.th, TrajectoryHandler), (
            "TrajectoryHandler not found in env"
        )

        # Store original frequency info
        self.original_freq = self.env.th.traj.info.frequency
        self.effective_freq = getattr(self.cfg, "control_freq", self.original_freq)

        self._metadata = self._discover_metadata()

    def _setup_cache(self):
        cache_path = os.path.expanduser("~/.loco-mujoco-caches")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        os.environ["LOCO_MUJOCO_CACHE"] = cache_path

    def _load_env(self):
        # Only create confs if the list is non-empty, otherwise set to None
        default_conf = (
            DefaultDatasetConf(self.dataset_dict["default"])
            if self.dataset_dict["default"]
            else None
        )
        lafan1_conf = (
            LAFAN1DatasetConf(self.dataset_dict["lafan1"])
            if self.dataset_dict["lafan1"]
            else None
        )
        amass_conf = (
            AMASSDatasetConf(self.dataset_dict["amass"])
            if self.dataset_dict["amass"]
            else None
        )
        env = ImitationFactory.make(
            self.env_name,
            default_dataset_conf=default_conf,  # type: ignore
            lafan1_dataset_conf=lafan1_conf,  # type: ignore
            amass_dataset_conf=amass_conf,  # type: ignore
            n_substeps=getattr(self.cfg, "n_substeps", 20),
        )

        self.num_traj = len(env.th.traj.data.split_points) - 1  # type: ignore

        self.split_points = env.th.traj.data.split_points  # type: ignore

        return env

    def _discover_metadata(self) -> DatasetMeta:
        """
        TODO: this is useless for now.
        """
        traj_info: dict[str, Any] = self.env.th.traj.info.to_dict()  # type: ignore

        traj_data_keys = [
            "qpos",
            "qvel",
            "xpos",
            "xquat",
            "cvel",
            "subtree_com",
            "site_xpos",
            "site_xmat",
        ]

        traj_data_lengths: list[int] = [
            self.env.th.traj.data.split_points[i + 1]  # type: ignore
            - self.env.th.traj.data.split_points[i]  # type: ignore
            for i in range(len(self.env.th.traj.data.split_points) - 1)  # type: ignore
        ]  # type: ignore

        return DatasetMeta(
            name="loco_mujoco",
            source="loco_mujoco",
            version="1.0.1",
            citation="TODO",
            num_trajectories=self.env.th.n_trajectories,  # type: ignore
            keys=traj_data_keys,
            trajectory_lengths=traj_data_lengths,
            dt=1.0 / traj_info["frequency"],
            joint_names=traj_info["joint_names"],
            body_names=traj_info["body_names"],
            site_names=traj_info["site_names"],
            metadata=traj_info["metadata"],
        )

    @property
    def metadata(self) -> DatasetMeta:
        return self._metadata

    def __len__(self) -> int:
        return self.num_traj

    def __getitem__(self, idx: int) -> None:
        return None

    def as_dataset(self, **kwargs) -> BaseDataset:
        return VectorizedTrajectoryDataset(self, **kwargs)  # type: ignore

    def save(self, path: str, **kwargs) -> None:
        """
        Saves the dataset to a directory. The dataset format is a Zarr store.
        The structure is as follows:
        Dataset/
        ├── motion1/  # e.g., default_walk
        │   ├── trajectory_0/
        │   │   ├── qpos/
        │   │   ├── qvel/
        │   │   ├── xpos/
        │   │   ├── xquat/
        │   │   ├── cvel/
        │   │   ├── subtree_com/
        │   │   ├── site_xpos/
        │   │   ├── site_xmat/
        │   ├── trajectory_1/
        │   └── ...
        └── ...

        However, the loco-mujoco dataset only assigns one trajectory to each motion.
        """

        chunk_size: int = kwargs.get("chunk_size", 10)
        shard_size: int = kwargs.get("shard_size", 100)
        if not os.path.exists(path):
            os.makedirs(path)
        store = storage.DirectoryStore(path)
        root = zarr.group(store=store, overwrite=False)
        locomujoco_group = root.create_group("loco_mujoco")

        # Need groupings because trajectories are in flat order in loco-mujoco, rather than grouped by dataset type
        groupings: list[str] = []
        for dataset_type, motions in self.dataset_dict.items():
            if not motions:
                continue
            for motion in motions:
                trajectory_name = f"{dataset_type}_{motion}"
                groupings.append(trajectory_name)

        # Save info, which is the same for all trajectories
        traj_info: dict[str, Any] = self.env.th.traj.info.to_dict()  # type: ignore
        for key, value in traj_info.items():
            if key in ["model", "metadata"]:
                continue
            locomujoco_group.attrs[key] = value

        # Save trajectories
        for idx, trajectory_name in enumerate(groupings):
            motion_group = locomujoco_group.create_group(trajectory_name)
            # only one trajectory per motion
            # TODO: consider splitting into multiple trajectories per motion
            trajectory_group = motion_group.create_group("trajectory_0")
            traj_start = self.env.th.traj.data.split_points[idx]  # type: ignore
            traj_end = self.env.th.traj.data.split_points[idx + 1]  # type: ignore

            for key, value in self.env.th.traj.to_dict().items():  # type: ignore
                if (
                    value is None
                    or isinstance(value, list)
                    or isinstance(value, int)
                    or isinstance(value, float)
                    or key
                    not in [
                        "qpos",
                        "qvel",
                        "xpos",
                        "xquat",
                        "cvel",
                        "subtree_com",
                        "site_xpos",
                        "site_xmat",
                    ]
                ):
                    continue
                # create chunked array
                sliced_value = np.array(value[traj_start:traj_end])
                chunks: list[int] = [chunk_size] + list(sliced_value.shape[1:])
                shards: list[int] = [shard_size] + list(sliced_value.shape[1:])
                trajectory_data = trajectory_group.create_dataset(
                    key,
                    shape=sliced_value.shape,
                    dtype=sliced_value.dtype,
                    chunks=chunks,
                    shards=shards,
                )
                trajectory_data[:] = sliced_value
