from abc import ABC
import os
import json
from typing import Any, Optional, Union, List, Dict

import torch
from tensordict import TensorDict
from omegaconf import DictConfig

from iltools_datasets.amass.loader import AmassLoader
from iltools_datasets.trajopt.loader import TrajoptLoader
from iltools_datasets.loco_mujoco.loader import LocoMuJoCoLoader
from iltools_datasets.storage import VectorizedTrajectoryDataset

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
        self._device = device  # Use private attribute to avoid descriptor issue

        # Core configuration
        self.dataset_path = self._validate_dataset_path(cfg)
        self.assignment_strategy = getattr(cfg, "assignment_strategy", "random")
        self.assignment_sequence = getattr(cfg, "assignment_sequence", None)

        # Initialize dataset
        self.dataset = self._initialize_dataset(cfg)

        # Trajectory tracking
        self.env2traj = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.env2step = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Cache trajectory info from dataset
        self.num_trajectories = len(self.dataset.available_trajectories)

        # Assignment tracking for round-robin
        self._round_robin_counter = 0

        print(
            f"[TrajectoryDatasetManager] Initialized with {self.num_trajectories} trajectories, {num_envs} envs"
        )

    @property
    def device(self) -> torch.device:
        """Get the device."""
        return self._device

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
            # For now, just use sequential as we don't have length info readily available
            self.env2traj[env_ids] = env_ids % self.num_trajectories
        else:
            raise ValueError(f"Unknown assignment strategy: {self.assignment_strategy}")

        # Reset step counters
        self.env2step[env_ids] = 0

        # Update dataset references
        env_to_traj = {
            env_id.item(): self.env2traj[env_id].item() for env_id in env_ids
        }
        env_to_step = {
            env_id.item(): self.env2step[env_id].item() for env_id in env_ids
        }
        self.dataset.update_references(env_to_traj=env_to_traj, env_to_step=env_to_step)

    def get_reference_data(self, target_joints) -> TensorDict:
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
        # Get current step indices for all environments
        current_steps = list(range(self.num_envs))

        # Fetch raw data using the dataset's fetch interface
        qpos_data = self.dataset.fetch(
            current_steps, key="qpos"
        )  # Shape: (num_envs, qpos_dim)
        qvel_data = None
        try:
            qvel_data = self.dataset.fetch(
                current_steps, key="qvel"
            )  # Shape: (num_envs, qvel_dim)
        except (KeyError, ValueError):
            # qvel might not be available
            pass

        # Convert to torch tensors
        qpos = torch.from_numpy(qpos_data).to(self.device)
        qvel = (
            torch.from_numpy(qvel_data).to(self.device)
            if qvel_data is not None
            else None
        )

        # Extract COM data (first 7 elements: x, y, z, qw, qx, qy, qz)
        com_pos = qpos[:, :3]  # (num_envs, 3)
        com_quat = qpos[:, 3:7]  # (num_envs, 4) - qw, qx, qy, qz

        # Extract COM velocities (first 6 elements: vx, vy, vz, wx, wy, wz)
        com_lin_vel = qvel[:, :3] if qvel is not None else torch.zeros_like(com_pos)
        com_ang_vel = qvel[:, 3:6] if qvel is not None else torch.zeros_like(com_pos)

        # Extract joint positions and velocities (skip first 7 elements for COM)
        joint_pos = qpos[:, 7:]  # (num_envs, num_joints)
        joint_vel = qvel[:, 6:] if qvel is not None else torch.zeros_like(joint_pos)

        # Create TensorDict with extracted data
        reference_data = TensorDict(
            {
                "com_pos": com_pos,
                "com_quat": com_quat,
                "com_lin_vel": com_lin_vel,
                "com_ang_vel": com_ang_vel,
                "joint_pos": joint_pos,
                "joint_vel": joint_vel,
                "raw_qpos": qpos,
                "raw_qvel": qvel if qvel is not None else torch.zeros_like(qpos),
            },
            batch_size=[self.num_envs],
            device=self.device,
        )

        # Increment step counters
        self.env2step += 1

        # Update dataset with new step positions
        env_to_step = {
            env_id: self.env2step[env_id].item() for env_id in range(self.num_envs)
        }
        self.dataset.update_references(env_to_step=env_to_step)

        # Handle trajectory completion - for now, just reset to beginning
        # TODO: Implement proper trajectory length tracking
        max_steps = 1000  # Placeholder - should get from dataset metadata
        completed_envs = self.env2step >= max_steps

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

    def _initialize_dataset(self, cfg: Any) -> VectorizedTrajectoryDataset:
        """Initialize or create the Zarr dataset."""
        if not self._check_zarr_exists():
            self._create_zarr_dataset(cfg)

        # Convert cfg to DictConfig if needed
        if not isinstance(cfg, DictConfig):
            # Try to convert to DictConfig, fallback to direct usage
            try:
                from omegaconf import OmegaConf

                dict_cfg = OmegaConf.create(
                    cfg.__dict__ if hasattr(cfg, "__dict__") else {}
                )
            except:
                dict_cfg = cfg
        else:
            dict_cfg = cfg

        return VectorizedTrajectoryDataset(
            zarr_path=os.path.join(self.dataset_path, "trajectories.zarr"),
            num_envs=self.num_envs,
            cfg=dict_cfg,
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

        # Create zarr path
        zarr_path = os.path.join(self.dataset_path, "trajectories.zarr")

        # Save using the loader's save method
        loader.save(zarr_path)

        # Create metadata file
        self._create_metadata_file(loader, cfg)

    def _create_metadata_file(self, loader: LoaderType, cfg: Any) -> None:
        """Create metadata.json file."""
        metadata = {
            "window_size": getattr(cfg, "window_size", 64),
            "control_freq": self._get_control_freq(cfg),
            "desired_horizon_steps": self._get_desired_horizon_steps(cfg),
            "loader_type": type(loader).__name__,
            "num_trajectories": len(loader),
        }

        meta_file = os.path.join(self.dataset_path, "metadata.json")
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=2)

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
