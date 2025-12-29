from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
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
from omegaconf import DictConfig
from zarr.storage import LocalStore

from iltools_core.metadata_schema import DatasetMeta
from iltools_datasets.base_loader import BaseLoader

# Type aliases for improved readability
MotionEntry = dict[str, Any]
TrajectoryEntry = dict[str, Any]
MotionIndex = dict[str, dict[str, dict[str, Any]]]

# Supported trajectory data keys for export
TRAJECTORY_DATA_KEYS: frozenset[str] = frozenset(
    ["qpos", "qvel", "xpos", "xquat", "cvel", "subtree_com", "site_xpos", "site_xmat"]
)


@dataclass(frozen=True)
class TrajectoryInfo:
    """Immutable container for trajectory slice information."""

    dataset: str
    motion: str
    motion_name: str
    trajectory_index: int
    trajectory_in_motion: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start

    def to_dict(self) -> TrajectoryEntry:
        return {
            "dataset": self.dataset,
            "motion": self.motion,
            "motion_name": self.motion_name,
            "trajectory_index": self.trajectory_index,
            "trajectory_in_motion": self.trajectory_in_motion,
            "start": self.start,
            "end": self.end,
            "length": self.length,
        }


class LocoMuJoCoLoader(BaseLoader):
    """
    Flexible loader for Loco-MuJoCo trajectories.
    Supports multiple datasets/motions (default, lafan1, amass).

    The desired control dt is computed and set through num_substeps when
    initializing the environment.
    """

    def __init__(
        self,
        env_name: str,
        cfg: DictConfig,
        build_zarr_dataset: bool = True,
        zarr_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the LocoMuJoCoLoader.

        Args:
            env_name: Name of the Loco-MuJoCo environment (e.g., "UnitreeG1").
            cfg: Configuration containing dataset trajectories and control parameters.
            build_zarr_dataset: If True, save trajectories to Zarr store during initialization.
            zarr_path: Directory path for the Zarr store (required if build_zarr_dataset is True).
            **kwargs: Additional keyword arguments passed to _discover_and_save_trajectories.
        """
        super().__init__()
        self.logger.info("Initializing LocoMuJoCoLoader")
        self.cfg = cfg
        self.env_name = env_name
        self.dataset_dict: dict[str, list[str | dict[str, str]]] = cfg.dataset.get(
            "trajectories", {"default": ["walk"], "amass": [], "lafan1": []}
        )
        self.logger.info("Dataset dictionary: %s", self.dataset_dict)

        self._setup_cache()
        self.env_info = {}
        self._trajectory_info_list, self._motion_info_dict = self._get_trajectories(
            build_zarr_dataset=build_zarr_dataset,
            path=zarr_path,
            **kwargs,
        )
        self._metadata = self._discover_metadata()

    @property
    def num_traj(self) -> int:
        """Number of trajectories across all motions."""
        return len(self._trajectory_info_list)

    def _setup_cache(self) -> None:
        """Configure the Loco-MuJoCo cache directory."""
        cache_path = os.path.join(tempfile.gettempdir(), "loco-mujoco-caches")
        os.makedirs(cache_path, exist_ok=True)
        os.environ["LOCO_MUJOCO_CACHE"] = cache_path

    def _collect_motion_entries(self) -> list[MotionEntry]:
        """
        Collect all configured motions across dataset types.

        Returns:
            List of motion entries with dataset type, motion name, and combined name.

        Raises:
            ValueError: If a motion dict is missing the 'name' field.
        """
        motions: list[MotionEntry] = []
        for dataset_type, motion_list in self.dataset_dict.items():
            if not motion_list:
                continue
            for motion in motion_list:
                # Extract motion name from string or dict specification
                if isinstance(motion, dict):
                    name = motion.get("name")
                    if name is None:
                        raise ValueError(
                            f"Motion spec in {dataset_type} requires a 'name' field."
                        )
                    motion_name = name
                else:
                    motion_name = motion

                motions.append(
                    {
                        "dataset": dataset_type,
                        "motion": motion_name,
                        "motion_name": f"{dataset_type}_{motion_name}",
                    }
                )
        return motions

    def _load_env_for_motion(self, dataset_type: str, motion: str) -> LocoEnv:
        """
        Load a Loco-MuJoCo environment for a specific motion.

        Args:
            dataset_type: One of "default", "lafan1", or "amass".
            motion: The motion/task name within the dataset.

        Returns:
            Configured LocoEnv instance.

        Raises:
            ValueError: If dataset_type is not recognized.
        """
        factory_configs = {
            "default": lambda m: {"default_dataset_conf": DefaultDatasetConf(task=m)},
            "lafan1": lambda m: {
                "lafan1_dataset_conf": LAFAN1DatasetConf(dataset_name=m)
            },
            "amass": lambda m: {
                "amass_dataset_conf": AMASSDatasetConf(rel_dataset_path=m)
            },
        }
        if dataset_type not in factory_configs:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        return ImitationFactory.make(
            self.env_name, **factory_configs[dataset_type](motion)
        )

    def _get_trajectories(
        self,
        build_zarr_dataset: bool = False,
        path: str | None = None,
        **kwargs: Any,
    ) -> tuple[list[TrajectoryEntry], MotionIndex]:
        """
        Discover trajectory and motion metadata, and optionally save to Zarr.

        This function iterates through all configured motions, builds trajectory and motion
        manifests, and if build_zarr_dataset is True, saves the trajectory data to a Zarr store.

        Args:
            build_zarr_dataset: If True, save trajectories to Zarr store at path.
            path: Directory path for the Zarr store (required if build_zarr_dataset is True).
            **kwargs: Optional parameters for Zarr saving:
                - chunk_size (int): Chunk size for Zarr arrays (default: 10).
                - shard_size (int): Shard size for Zarr arrays (default: 100).
                - export_transitions (bool): Export obs/action transitions (default: False).
                - obs_key_name (str): Key name for observations (default: "obs").
                - next_obs_key_name (str): Key name for next observations (default: "next_obs").
                - action_key_name (str): Key name for actions (default: "action").

        Returns:
            Tuple of (trajectory_info_list, motion_info_dict).
            Trajectory info list is a list of dictionaries, each containing the information about a trajectory.
            Motion manifest is a dictionary of dictionaries, each containing the information about a motion.

        Raises:
            ValueError: If build_zarr_dataset is True but path is None.
        """
        if build_zarr_dataset and path is None:
            raise ValueError("path must be provided when build_zarr_dataset is True")

        trajectory_info_list: list[TrajectoryEntry] = []
        motion_info_dict: MotionIndex = {}
        global_idx = 0

        # Initialize Zarr store if needed
        locomujoco_group: zarr.Group | None = None
        if build_zarr_dataset:
            chunk_size: int = kwargs.get("chunk_size", 10)
            shard_size: int = kwargs.get("shard_size", 100)
            os.makedirs(path, exist_ok=True)
            store = LocalStore(path)
            root = zarr.group(store=store, overwrite=False)
            locomujoco_group = root.create_group("loco_mujoco")

        motion_metadata: dict[str, dict[str, Any]] = {}

        for entry in self._collect_motion_entries():
            # Note that each motion can have multiple trajectories, each with a different start and end index.
            env = self._load_env_for_motion(entry["dataset"], entry["motion"])
            if not self.env_info:
                # Store the environment information only once.
                self.env_info = {
                    "joint_names": env.th.traj.info.joint_names,
                    "body_names": env.th.traj.info.body_names,
                    "site_names": env.th.traj.info.site_names,
                }
                self.logger.info("Environment information: %s", self.env_info)
            split_points = env.th.traj.data.split_points
            num_traj = len(split_points) - 1
            motion_name = entry["motion_name"]

            motion_entry = motion_info_dict.setdefault(entry["dataset"], {}).setdefault(
                entry["motion"],
                {
                    "motion_name": motion_name,
                    "trajectory_indices": [],
                    "trajectory_lengths": [],
                    "trajectory_local_start_indices": [],
                    "trajectory_local_end_indices": [],
                },
            )

            # Create motion group in Zarr if saving
            motion_group: zarr.Group | None = None
            if build_zarr_dataset and locomujoco_group is not None:
                motion_group = locomujoco_group.create_group(motion_name)
                motion_group.attrs["num_trajectories"] = num_traj

            # Iterate over each trajectory in the motion.
            for local_idx in range(num_traj):
                traj_start = int(split_points[local_idx])
                traj_end = int(split_points[local_idx + 1])
                traj_len = traj_end - traj_start

                traj_info = TrajectoryInfo(
                    dataset=entry["dataset"],
                    motion=entry["motion"],
                    motion_name=motion_name,
                    trajectory_index=global_idx,
                    trajectory_in_motion=local_idx,
                    start=traj_start,
                    end=traj_end,
                )

                # Add to trajectory manifest, motion manifest, and global index
                # Also store the local start and end indices for the trajectory, i.e., the indices within the motion.
                trajectory_info_list.append(traj_info.to_dict())
                motion_entry["trajectory_indices"].append(global_idx)
                motion_entry["trajectory_lengths"].append(traj_info.length)
                motion_entry["trajectory_local_start_indices"].append(traj_start)
                motion_entry["trajectory_local_end_indices"].append(traj_end)

                # Save trajectory data to Zarr if requested
                if build_zarr_dataset and motion_group is not None:
                    traj_group = motion_group.create_group(f"trajectory_{local_idx}")
                    self._save_trajectory_data(
                        traj_group,
                        env,
                        traj_start,
                        traj_end,
                        chunk_size,
                        shard_size,
                    )

                    if kwargs.get("export_transitions", False):
                        self._save_transitions(
                            traj_group,
                            env,
                            split_points,
                            local_idx,
                            traj_len,
                            **kwargs,
                        )

                global_idx += 1

            # Update motion group attributes if saving
            if build_zarr_dataset and motion_group is not None:
                motion_group.attrs["trajectory_lengths"] = motion_entry[
                    "trajectory_lengths"
                ]
                motion_metadata[motion_name] = {
                    "num_trajectories": num_traj,
                    "trajectory_lengths": motion_entry["trajectory_lengths"],
                }

        # Finalize Zarr store attributes if saving
        if build_zarr_dataset and locomujoco_group is not None:
            dt = self.control_dt
            locomujoco_group.attrs["num_trajectories"] = len(trajectory_info_list)
            locomujoco_group.attrs["trajectory_lengths"] = [
                e["length"] for e in trajectory_info_list
            ]
            locomujoco_group.attrs["joint_names"] = self.env_info["joint_names"]
            locomujoco_group.attrs["body_names"] = self.env_info["body_names"]
            locomujoco_group.attrs["site_names"] = self.env_info["site_names"]
            locomujoco_group.attrs["keys"] = list(TRAJECTORY_DATA_KEYS)
            locomujoco_group.attrs["dt"] = dt
            locomujoco_group.attrs["motion_metadata"] = motion_metadata
            locomujoco_group.attrs["trajectory_info_list"] = trajectory_info_list
            locomujoco_group.attrs["motion_info_dict"] = motion_info_dict

        self.logger.info(
            "Built trajectory manifest with %d entries across %d motions",
            len(trajectory_info_list),
            sum(len(m) for m in motion_info_dict.values()),
        )
        if build_zarr_dataset:
            self.logger.info("Saved trajectories to Zarr store at %s", path)

        return trajectory_info_list, motion_info_dict

    def _discover_metadata(self) -> DatasetMeta:
        """Build dataset metadata from the trajectory manifest."""
        traj_lengths = [int(e["length"]) for e in self._trajectory_info_list]
        dt = self.control_dt

        return DatasetMeta(
            name="loco_mujoco",
            source="loco_mujoco",
            version="1.0.1",
            citation="LocoMuJoCo: A Comprehensive Imitation Learning Benchmark for Locomotion",
            num_trajectories=len(self._trajectory_info_list),
            keys=list(TRAJECTORY_DATA_KEYS),
            trajectory_lengths=traj_lengths,
            dt=dt,
            joint_names=self.env_info["joint_names"],
            body_names=self.env_info["body_names"],
            site_names=self.env_info["site_names"],
            metadata={
                "trajectory_info_list": self._trajectory_info_list,
                "motion_info_dict": self._motion_info_dict,
            },
        )

    @property
    def control_dt(self) -> float:
        """Compute control timestep from configuration."""
        control_freq = getattr(self.cfg, "control_freq", None)
        return 1.0 / control_freq if control_freq else 0.0

    @property
    def metadata(self) -> DatasetMeta:
        return self._metadata

    def __len__(self) -> int:
        return self.num_traj

    @property
    def trajectory_info_list(self) -> list[TrajectoryEntry]:
        """Return the trajectory manifest as a list."""
        return list(self._trajectory_info_list)

    @property
    def motion_info_dict(self) -> MotionIndex:
        """Return the motion manifest as a dictionary."""
        return dict(self._motion_info_dict)

    def _save_trajectory_data(
        self,
        traj_group: zarr.Group,
        env: LocoEnv,
        traj_start: int,
        traj_end: int,
        chunk_size: int,
        shard_size: int,
    ) -> None:
        """Save trajectory state data (qpos, qvel, etc.) to Zarr."""
        traj_dict = env.th.traj.to_dict()
        for key, value in traj_dict.items():
            # Check if trajectory data entry should be exported
            if value is None or isinstance(value, (list, int, float)):
                continue
            if key not in TRAJECTORY_DATA_KEYS:
                continue

            sliced = np.asarray(value[traj_start:traj_end])
            chunks = [chunk_size] + list(sliced.shape[1:])
            shards = [shard_size] + list(sliced.shape[1:])
            dataset = traj_group.create_dataset(
                key,
                shape=sliced.shape,
                dtype=sliced.dtype,
                chunks=chunks,
                shards=shards,
            )
            dataset[:] = sliced

    def _save_transitions(
        self,
        traj_group: zarr.Group,
        env: LocoEnv,
        split_points: np.ndarray,
        local_idx: int,
        traj_len: int,
        **kwargs: Any,
    ) -> None:
        """Save transition data (obs, action, next_obs) to Zarr."""
        obs_key = kwargs.get("obs_key_name", "obs")
        next_obs_key = kwargs.get("next_obs_key_name", "next_obs")
        action_key = kwargs.get("action_key_name", "action")

        transitions = env.th.traj.transitions or env.create_dataset()
        obs, next_obs, actions, absorbings, dones = self._extract_transitions(
            transitions
        )

        # Calculate offset into flattened transition array
        prev_transitions = sum(
            max(0, int(split_points[k + 1] - split_points[k]) - 1)
            for k in range(local_idx)
        )

        n_transitions = max(0, traj_len - 1)
        if n_transitions <= 0:
            return

        slice_range = slice(prev_transitions, prev_transitions + n_transitions)

        self._create_transition_dataset(traj_group, obs_key, obs[slice_range])
        self._create_transition_dataset(traj_group, next_obs_key, next_obs[slice_range])

        if actions is not None:
            self._create_transition_dataset(
                traj_group, action_key, actions[slice_range]
            )
        if absorbings is not None:
            self._create_transition_dataset(
                traj_group, "absorbing", absorbings[slice_range]
            )
        if dones is not None:
            self._create_transition_dataset(traj_group, "done", dones[slice_range])

    def _extract_transitions(
        self, transitions: Any
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        """Extract transition arrays from the transitions object."""
        try:
            obs = np.asarray(transitions.observations)
            next_obs = np.asarray(transitions.next_observations)
            actions = self._to_optional_array(transitions.actions)
            absorbings = self._to_optional_array(transitions.absorbings)
            dones = self._to_optional_array(transitions.dones)
        except Exception:
            try:
                np_trans = transitions.to_np()
                obs = np_trans.observations
                next_obs = np_trans.next_observations
                actions = self._to_optional_array(np_trans.actions)
                absorbings = self._to_optional_array(np_trans.absorbings)
                dones = self._to_optional_array(np_trans.dones)
            except Exception as e:
                raise RuntimeError(
                    "Failed to convert transitions to numpy arrays"
                ) from e

        return obs, next_obs, actions, absorbings, dones

    def _to_optional_array(self, arr: Any) -> np.ndarray | None:
        """Convert to numpy array if non-empty, otherwise return None."""
        if arr is None:
            return None
        arr = np.asarray(arr)
        return arr if arr.size > 0 else None

    def _create_transition_dataset(
        self, group: zarr.Group, name: str, data: np.ndarray
    ) -> None:
        """Create and populate a transition dataset in Zarr."""
        ds = group.create_dataset(name, shape=data.shape, dtype=data.dtype)
        ds[:] = data
