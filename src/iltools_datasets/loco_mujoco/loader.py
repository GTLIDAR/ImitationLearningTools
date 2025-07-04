import os
from functools import lru_cache
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch
import zarr
from loco_mujoco.task_factories import (
    AMASSDatasetConf,
    DefaultDatasetConf,
    ImitationFactory,
    LAFAN1DatasetConf,
)
from loco_mujoco.trajectory.dataclasses import interpolate_trajectories
from loco_mujoco.trajectory.handler import TrajectoryHandler

from iltools_core.metadata_schema import DatasetMeta
from iltools_core.trajectory import Trajectory as ILTTrajectory
from iltools_datasets.base_loader import (
    BaseDataset,
    BaseLoader,
)


class LocoMuJoCoLoader(BaseLoader):
    """
    Flexible loader for Loco-MuJoCo trajectories.
    Now supports multiple datasets/motions (default, lafan1, amass).
    """

    def __init__(
        self,
        env_name: str,
        dataset_motion_config: dict = None,  # e.g. {"default": ["walk", "squat"], "lafan1": ["dance2_subject4"], "amass": ["DanceDB/..."]}
        default_control_freq: Optional[float] = None,
    ):
        self.env_name = env_name
        self.dataset_motion_config = dataset_motion_config or {"default": ["walk"]}
        self.default_control_freq = default_control_freq
        self._setup_cache()
        self.env, self._traj_to_motion = self._load_env_and_mapping()
        assert hasattr(self.env, "th") and isinstance(self.env.th, TrajectoryHandler), (
            "TrajectoryHandler not found in env"
        )
        self.th: TrajectoryHandler = self.env.th

        # Store original frequency info
        self.original_freq = self.th.traj.info.frequency
        self.effective_freq = self.default_control_freq or self.original_freq

        self._metadata = self._discover_metadata()

    def _setup_cache(self):
        cache_path = os.path.expanduser("~/.loco-mujoco-caches")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        os.environ["LOCO_MUJOCO_CACHE"] = cache_path

    def _load_env_and_mapping(self):
        # Build config objects for each dataset type
        from loco_mujoco.task_factories import DefaultDatasetConf, LAFAN1DatasetConf, AMASSDatasetConf, ImitationFactory
        default_conf = DefaultDatasetConf(self.dataset_motion_config.get("default", [])) if "default" in self.dataset_motion_config else None
        lafan1_conf = LAFAN1DatasetConf(self.dataset_motion_config.get("lafan1", [])) if "lafan1" in self.dataset_motion_config else None
        amass_conf = AMASSDatasetConf(self.dataset_motion_config.get("amass", [])) if "amass" in self.dataset_motion_config else None
        # Call ImitationFactory.make with all configs
        env = ImitationFactory.make(
            self.env_name,
            default_dataset_conf=default_conf,
            lafan1_dataset_conf=lafan1_conf,
            amass_dataset_conf=amass_conf,
            n_substeps=20,
        )
        # Try to build a mapping from trajectory index to (dataset_type, motion)
        # This depends on env.th.traj.info having motion_names and dataset_types attributes
        info = getattr(env.th.traj, "info", None)
        if info is not None:
            motion_names = getattr(info, "motion_names", None)
            dataset_types = getattr(info, "dataset_types", None)
            if motion_names is not None and dataset_types is not None:
                mapping = [(dataset_types[i], motion_names[i]) for i in range(len(motion_names))]
            else:
                mapping = [("unknown", "unknown") for _ in range(env.th.n_trajectories)]
        else:
            mapping = [("unknown", "unknown") for _ in range(env.th.n_trajectories)]
        return env, mapping

    def _discover_metadata(self) -> DatasetMeta:
        # Discover available observation/action keys from the first trajectory
        # (Assume all trajectories have the same keys for now, but can be extended)
        assert isinstance(self.th, TrajectoryHandler), "TrajectoryHandler not found"
        # Get available keys from the TrajectoryData fields
        data_fields = list(vars(self.th.traj.data).keys())

        # Filter out structural/metadata fields that shouldn't be treated as observations
        excluded_fields = {
            "split_points",
            "episode_starts",
            "episode_ends",
            "time_stamps",
        }

        obs_keys = []
        for k in data_fields:
            if (
                not k.startswith("_")
                and k not in excluded_fields
                and hasattr(self.th.traj.data, k)
            ):
                field_value = getattr(self.th.traj.data, k)
                if (
                    field_value is not None
                    and hasattr(field_value, "size")
                    and field_value.size > 0
                ):
                    # Check if the field represents trajectory data by verifying
                    # its length matches total timesteps across all trajectories
                    total_timesteps = (
                        self.th.traj.data.split_points[-1]
                        if hasattr(self.th.traj.data, "split_points")
                        else 0
                    )
                    if hasattr(field_value, "shape") and len(field_value.shape) >= 1:
                        if field_value.shape[0] == total_timesteps:
                            obs_keys.append(k)
        # Try to detect action keys (if present)
        action_keys = None
        if (
            hasattr(self.th.traj, "transitions")
            and self.th.traj.transitions is not None
        ):
            if getattr(self.th.traj.transitions, "actions", None) is not None:
                action_keys = ["actions"]

        # Calculate trajectory lengths at effective frequency
        trajectory_lengths = []
        for traj_ind in range(self.th.n_trajectories):
            if (
                self.default_control_freq is not None
                and self.default_control_freq != self.original_freq
            ):
                # If using different frequency, need to calculate interpolated length
                original_length = int(self.th.len_trajectory(traj_ind))
                original_dt = 1.0 / self.original_freq
                total_time = original_length * original_dt
                new_length = int(total_time * self.effective_freq)
                trajectory_lengths.append(new_length)
            else:
                trajectory_lengths.append(int(self.th.len_trajectory(traj_ind)))

        # dt is 1.0 / effective_frequency for all trajectories
        dt = [1.0 / self.effective_freq] * self.th.n_trajectories

        return DatasetMeta(
            name=f"loco_mujoco_{self.env_name}_{self.dataset_motion_config['default'][0]}",
            source="loco_mujoco",
            version="1.0.1",
            citation="TODO",
            num_trajectories=self.th.n_trajectories,
            observation_keys=obs_keys,
            action_keys=action_keys,
            trajectory_lengths=trajectory_lengths,
            dt=dt,
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
        If control_freq is None, uses the loader's default_control_freq or original frequency.
        """
        raise NotImplementedError("Not implemented")
        # Determine effective control frequency
        effective_control_freq = control_freq or self.effective_freq

        # Get the slice for this trajectory
        start = self.th.traj.data.split_points[idx]
        end = self.th.traj.data.split_points[idx + 1]
        length = end - start

        # Optionally interpolate
        if effective_control_freq != self.th.traj.info.frequency:
            # Interpolate using loco-mujoco utility
            new_data, new_info = interpolate_trajectories(
                self.th.traj.data, self.th.traj.info, effective_control_freq, backend=np
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
        dt = 1.0 / effective_control_freq
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

    def as_dataset(self, **kwargs) -> BaseDataset:
        return LazyDataset(self, **kwargs)

    def save(self, path: str, **kwargs) -> None:
        """
        Saves the dataset to a directory. The dataset format is a Zarr store.
        The structure is as follows:
        Dataset/
        ├── motion1/  # e.g., default_walk
        │   ├── trajectory1/
        │   │   ├── observations/
        │   │   ├── actions/
        │   │   ├── rewards
        │   │   └── infos
        │   ├── trajectory2/
        │   └── ...
        └── ...
        """
        import zarr
        import numpy as np
        import os
        chunk_size: int = kwargs.get("chunk_size", 1000)
        if not os.path.exists(path):
            os.makedirs(path)
        store = zarr.DirectoryStore(os.path.join(path, "trajectories.zarr"))
        root = zarr.group(store=store, overwrite=True)
        # Group trajectories by motion name (with dataset type prefix)
        from collections import defaultdict
        groupings = defaultdict(list)
        for idx, (dataset_type, motion) in enumerate(self._traj_to_motion):
            motion_name = f"{dataset_type}_{motion}"
            groupings[motion_name].append(idx)
        for motion_name, indices in groupings.items():
            motion_group = root.require_group(motion_name)
            for i, idx in enumerate(indices):
                traj = self.__getitem__(idx)
                traj_group = motion_group.require_group(f"trajectory{i}")
                # Save observations
                obs_group = traj_group.require_group("observations")
                for k, v in traj.observations.items():
                    obs_group.create_dataset(k, data=v, chunks=(min(chunk_size, v.shape[0]),) + v.shape[1:], overwrite=True)
                # Save actions
                if traj.actions is not None:
                    act_group = traj_group.require_group("actions")
                    for k, v in traj.actions.items():
                        act_group.create_dataset(k, data=v, chunks=(min(chunk_size, v.shape[0]),) + v.shape[1:], overwrite=True)
                # Save rewards if present
                if hasattr(traj, "rewards") and traj.rewards is not None:
                    traj_group.create_dataset("rewards", data=traj.rewards, chunks=(min(chunk_size, traj.rewards.shape[0]),), overwrite=True)
                # Save infos if present
                if hasattr(traj, "infos") and traj.infos is not None:
                    import json
                    try:
                        traj_group.attrs["infos"] = json.dumps(traj.infos)
                    except Exception:
                        traj_group.attrs["infos"] = str(traj.infos)
