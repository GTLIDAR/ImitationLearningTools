from typing import Optional

import numpy as np
import zarr
from omegaconf import DictConfig
from iltools_datasets.base_loader import BaseDataset
from dask.distributed import Client
import dask.array as da
from dask.distributed import wait


class VectorizedTrajectoryDataset(BaseDataset):
    """Dataset that reads from per-trajectory Zarr format.

    Each trajectory is stored as a separate Zarr group:

    """

    def __init__(self, zarr_path: str, num_envs: int, cfg: DictConfig, **kwargs):
        self.dask_client = Client()
        self.zarr_dataset = zarr.open_group(zarr_path, mode="r")
        print(self.zarr_dataset.tree())
        print(sorted(self.zarr_dataset["loco_mujoco"].attrs))
        self.num_envs = num_envs
        self.cfg = cfg
        self.window_size = cfg.window_size
        self.all_keys = []

        # suppose we have num_envs trajectories which are specified by cfg
        self.shuttle = self.prepare_shuttle(num_envs)

        # TODO: add customization
        self.shuttle_pt = [0] * num_envs

        self.env_plan = self._build_env_plan()

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

    def prepare_shuttle(self, num_envs: int) -> dict[str, list[da.Array]]:
        """Prepare shuttle for fetching trajectories.

        Returns a dictionary of lists of trajectories, where the keys are the dataset sources
        and the values are the trajectories.
        """
        shuttle: dict[str, list[da.Array]] = {}

        typical_trajectory = self.zarr_dataset[self.available_trajectories[0]]
        assert isinstance(typical_trajectory, zarr.Group)
        self.all_keys = sorted(typical_trajectory.keys())
        for key in typical_trajectory.keys():
            shuttle[key] = []

            # initialize with the first trajectory
            for _ in range(num_envs):
                data_path = self.available_trajectories[0] + "/" + key
                shuttle[key].append(da.from_zarr(self.zarr_dataset[data_path]))

        return shuttle

    def prepare_shuttle_pt(self, num_envs: int) -> list[int]:
        """Prepare shuttle pointer for fetching trajectories."""
        return [0] * num_envs

    def update_shuttle_pt(self, env_to_traj: dict[int, int]):
        """Update shuttle pointer for fetching trajectories."""
        for env_id, idx in env_to_traj.items():
            self.shuttle_pt[env_id] = idx

    def update_shuttle(self, env_to_traj: dict[int, int]):
        """Update shuttle for fetching trajectories."""
        for key in self.shuttle.keys():
            for env_id, traj_id in env_to_traj.items():
                self.shuttle[key][env_id] = da.from_zarr(
                    self.zarr_dataset[self.available_trajectories[traj_id]]
                )

    def _build_env_plan(self):
        """Build slice for fetching trajectories. For example,
        env_plan = [
            (0, 10),  # env-0  ←  traj_0[10]
            (1, 10),  # env-1  ←  traj_1[35 000]
            (2, 10),  # env-2  ←  traj_2[1 234]
            (3, 10),
        ]  # env-3  ←  traj_3[42]
        """
        return [(env_id, self.shuttle_pt[env_id]) for env_id in range(self.num_envs)]

    def _update_env_plan(self, env_to_step: dict[int, int]):
        """Update env plan for fetching trajectories."""
        for env_id, step in env_to_step.items():
            self.env_plan[env_id] = (self.env_plan[env_id][0], step)

    def update_references(
        self,
        env_to_traj: Optional[dict[int, int]] = None,
        env_to_step: Optional[dict[int, int]] = None,
    ):
        if env_to_traj is not None:
            self.update_shuttle(env_to_traj)
        if env_to_step is not None:
            self.update_shuttle_pt(env_to_step)
            self._update_env_plan(env_to_step)

    def _maybe_prefetch(self, key: str):
        """Prefetch data if the number of environments is greater than the number of workers."""
        slices = [
            self.shuttle[key][env_id][step + 1][None, :]
            for env_id, step in self.env_plan
        ]
        batch_da = da.stack(slices, axis=0)
        batch_np = self.dask_client.persist(batch_da)
        wait(batch_np)

    def fetch(self, idx: list[int], key: Optional[str] = None) -> np.ndarray:
        # TODO: return all the data in the slice if key is None
        assert key is not None
        assert len(idx) == self.num_envs

        slices = [
            self.shuttle[key][env_id][step][None, :] for env_id, step in self.env_plan
        ]
        batch_da = da.stack(slices, axis=0)
        batch_np = batch_da.persist()

        # wait for the current batch to be computed
        wait(batch_np)

        # prefetch the next step
        # self._maybe_prefetch(key)

        return np.array(batch_np.compute())
