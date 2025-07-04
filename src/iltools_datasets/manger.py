from abc import ABC
import os
import json
from typing import Any, Optional, Union, List, Dict

import torch
from tensordict import TensorDict

from iltools_datasets.amass.loader import AmassLoader
from iltools_datasets.trajopt.loader import TrajoptLoader
from iltools_datasets.loco_mujoco.loader import LocoMuJoCoLoader
from iltools_datasets.windowed_dataset import WindowedTrajectoryDataset
from iltools_datasets.export_utils import export_trajectories_to_zarr

LoaderType = Union["LocoMuJoCoLoader", "AmassLoader", "TrajoptLoader"]


class TrajectoryDatasetManager:
    """
    Enhanced dataset manager that handles all trajectory operations internally.
    Provides a clean interface for the ImitationRLEnv with automatic COM and joint extraction.
    """

    def __init__(self, cfg: Any, num_envs: int, device: torch.device) -> None:
        """
        Initialize the trajectory dataset manager.

        Args:
            cfg: Configuration object with dataset_path, assignment_strategy, etc.
            num_envs: Number of environments
            device: Torch device
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        # Core configuration
        self.dataset_path = self._validate_dataset_path(cfg)
        self.assignment_strategy = getattr(cfg, "assignment_strategy", "random")
        self.assignment_sequence = getattr(cfg, "assignment_sequence", None)

        # Initialize dataset
        self.dataset = self._initialize_dataset(cfg)

        # Trajectory tracking
        self.env2traj = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.env2step = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Cache trajectory info
        self.traj_lengths = torch.tensor(
            self.dataset.lengths, device=device, dtype=torch.long
        )
        self.num_trajectories = len(self.traj_lengths)

        # Assignment tracking for round-robin
        self._round_robin_counter = 0

        print(
            f"[TrajectoryDatasetManager] Initialized with {self.num_trajectories} trajectories, {num_envs} envs"
        )

    def reset_trajectories(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """
        Reset trajectory tracking for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, resets all environments.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Assign trajectories based on strategy
        if self.assignment_strategy == "random":
            self.env2traj[env_ids] = torch.randint(
                0, self.num_trajectories, (len(env_ids),), device=self.device
            )
        elif self.assignment_strategy == "sequential":
            self.env2traj[env_ids] = env_ids % self.num_trajectories
        elif self.assignment_strategy == "round_robin":
            # Round-robin assignment across all trajectories
            for i, env_id in enumerate(env_ids):
                traj_id = (self._round_robin_counter + i) % self.num_trajectories
                self.env2traj[env_id] = traj_id
            self._round_robin_counter = (
                self._round_robin_counter + len(env_ids)
            ) % self.num_trajectories
        elif self.assignment_strategy == "sequence":
            # Use predefined sequence
            if self.assignment_sequence is None:
                raise ValueError(
                    "assignment_sequence must be provided when using 'sequence' strategy"
                )
            for i, env_id in enumerate(env_ids):
                seq_idx = env_id.item() % len(self.assignment_sequence)
                self.env2traj[env_id] = self.assignment_sequence[seq_idx]
        elif self.assignment_strategy == "curriculum":
            # Simple curriculum: start with shorter trajectories
            sorted_indices = torch.argsort(self.traj_lengths)
            self.env2traj[env_ids] = sorted_indices[env_ids % self.num_trajectories]
        else:
            raise ValueError(f"Unknown assignment strategy: {self.assignment_strategy}")

        # Reset step counters
        self.env2step[env_ids] = 0

    def get_reference_data(self) -> TensorDict:
        """
        Get the reference data for all environments with extracted COM and joint information.

        Returns:
            TensorDict containing:
                - com_pos: Center of mass position (num_envs, 3)
                - com_quat: Center of mass orientation as quaternion (num_envs, 4)
                - com_lin_vel: Center of mass linear velocity (num_envs, 3)
                - com_ang_vel: Center of mass angular velocity (num_envs, 3)
                - joint_pos: Joint positions (num_envs, num_joints)
                - joint_vel: Joint velocities (num_envs, num_joints)
                - raw_qpos: Raw qpos data (num_envs, qpos_dim)
                - raw_qvel: Raw qvel data (num_envs, qvel_dim) if available
        """
        # Get current trajectory and step for each env
        current_trajs = self.env2traj
        current_steps = self.env2step

        # Fetch raw data using the dataset's batch interface
        raw_data = self.dataset.batch_get(current_trajs, current_steps)

        # Extract data from the flat structure - use getattr to avoid typing issues
        qpos = getattr(raw_data, "get", lambda x: raw_data[x])(
            "observations/qpos"
        )  # Shape: (num_envs, window_size, qpos_dim)
        qvel = getattr(
            raw_data,
            "get",
            lambda x: raw_data.get(x, None) if hasattr(raw_data, "get") else None,
        )("observations/qvel")  # Shape: (num_envs, window_size, qvel_dim)

        # Take the first frame of each window for current reference
        current_qpos = qpos[:, 0]  # (num_envs, qpos_dim)
        current_qvel = qvel[:, 0] if qvel is not None else None  # (num_envs, qvel_dim)

        # Extract COM data (first 7 elements: x, y, z, qw, qx, qy, qz)
        com_pos = current_qpos[:, :3]  # (num_envs, 3)
        com_quat = current_qpos[:, 3:7]  # (num_envs, 4) - qw, qx, qy, qz

        # Extract COM velocities (first 6 elements: vx, vy, vz, wx, wy, wz)
        com_lin_vel = (
            current_qvel[:, :3]
            if current_qvel is not None
            else torch.zeros_like(com_pos)
        )
        com_ang_vel = (
            current_qvel[:, 3:6]
            if current_qvel is not None
            else torch.zeros_like(com_pos)
        )

        # Extract joint positions and velocities (skip first 7 elements for COM)
        joint_pos = current_qpos[:, 7:]  # (num_envs, num_joints)
        joint_vel = (
            current_qvel[:, 6:]
            if current_qvel is not None
            else torch.zeros_like(joint_pos)
        )

        # Create TensorDict with extracted data
        reference_data = TensorDict(
            {
                "com_pos": com_pos,
                "com_quat": com_quat,
                "com_lin_vel": com_lin_vel,
                "com_ang_vel": com_ang_vel,
                "joint_pos": joint_pos,
                "joint_vel": joint_vel,
                "raw_qpos": current_qpos,
                "raw_qvel": current_qvel,
            },
            batch_size=[self.num_envs],
            device=self.device,
        )

        # Increment step counters
        self.env2step += 1

        # Handle trajectory completion - reset environments that have reached the end
        max_steps = self.traj_lengths[current_trajs] - 1
        completed_envs = self.env2step > max_steps

        if torch.any(completed_envs):
            completed_env_ids = torch.where(completed_envs)[0]
            self.reset_trajectories(completed_env_ids)

        return reference_data

    def _validate_dataset_path(self, cfg: Any) -> str:
        """Validate and return the dataset path from config."""
        dataset_path = getattr(cfg, "dataset_path", None)
        if dataset_path is None:
            raise ValueError("dataset_path must be provided in the config.")
        return dataset_path

    def _initialize_dataset(self, cfg: Any) -> WindowedTrajectoryDataset:
        """Initialize or create the Zarr dataset."""
        if not self._check_zarr_exists():
            self._create_zarr_dataset(cfg)
            window_size = getattr(cfg, "window_size", 64)
        else:
            window_size = self._get_window_size_from_metadata()

        return WindowedTrajectoryDataset(
            self.dataset_path,
            window_size=window_size,
            device=self.device,
        )

    def _check_zarr_exists(self) -> bool:
        """Check if both the Zarr directory and metadata file exist."""
        zarr_dir = os.path.join(self.dataset_path, "trajectories.zarr")
        meta_file = os.path.join(self.dataset_path, "metadata.json")
        return os.path.exists(zarr_dir) and os.path.exists(meta_file)

    def _get_window_size_from_metadata(self) -> int:
        """Read window_size from metadata.json."""
        meta_file = os.path.join(self.dataset_path, "metadata.json")
        with open(meta_file, "r") as f:
            metadata = json.load(f)
        return int(metadata.get("window_size", 64))

    def _create_zarr_dataset(self, cfg: Any) -> None:
        """Create Zarr dataset from loader if it doesn't exist."""
        loader_type = getattr(cfg, "loader_type", None)
        loader_kwargs = getattr(cfg, "loader_kwargs", None)

        if loader_type is None or loader_kwargs is None:
            raise RuntimeError(
                "Zarr dataset not found and loader_type/loader_kwargs not provided in config."
            )

        loader = self._get_loader(loader_type, loader_kwargs)
        export_trajectories_to_zarr(
            loader,
            self.dataset_path,
            window_size=getattr(cfg, "window_size", 64),
            control_freq=self._get_control_freq(cfg),
            desired_horizon_steps=self._get_desired_horizon_steps(cfg),
            horizon_multiplier=2.0,
        )

    def _get_loader(
        self, loader_type: str, loader_kwargs: dict[str, Any]
    ) -> LoaderType:
        """Get the appropriate loader based on type."""
        if loader_type == "loco_mujoco":
            return LocoMuJoCoLoader(**loader_kwargs)
        elif loader_type == "amass":
            return AmassLoader(**loader_kwargs)
        elif loader_type == "trajopt":
            return TrajoptLoader(**loader_kwargs)
        else:
            raise ValueError(f"Unknown loader_type: {loader_type}")

    def _get_control_freq(self, cfg: Any) -> float:
        """Compute the control frequency from the config."""
        return 1.0 / (cfg.sim.dt * cfg.decimation)

    def _get_desired_horizon_steps(self, cfg: Any) -> int:
        """Compute the desired number of horizon steps from the config."""
        return int(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))
