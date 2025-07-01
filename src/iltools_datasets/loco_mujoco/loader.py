import logging
import os
from functools import lru_cache
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch
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

    def __init__(
        self,
        env_name: str,
        task: str = "walk",
        default_control_freq: Optional[float] = None,
    ):
        self.env_name = env_name
        self.task = task
        self.default_control_freq = default_control_freq
        self._setup_cache()
        self.env = self._load_env()
        assert hasattr(self.env, "th") and isinstance(
            self.env.th, TrajectoryHandler
        ), "TrajectoryHandler not found in env"
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

        return DatasetMeta(
            name=f"loco_mujoco_{self.env_name}_{self.task}",
            source="loco_mujoco",
            version="1.0.1",
            citation="TODO",
            num_trajectories=self.th.n_trajectories,
            observation_keys=obs_keys,
            action_keys=action_keys,
            trajectory_lengths=trajectory_lengths,
            # Add frequency metadata
            original_frequency=self.original_freq,
            effective_frequency=self.effective_freq,
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
                self.th.traj.data, self.th.traj.info, effective_control_freq
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

    def as_dataset(self, **kwargs) -> BaseTrajectoryDataset:
        return LocoMuJoCoTrajectoryIndexDataset(self, **kwargs)


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
