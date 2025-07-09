import json
import os
from typing import Any, List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from iltools_datasets.amass.loader import AmassLoader
from iltools_datasets.loco_mujoco.loader import LocoMuJoCoLoader
from iltools_datasets.storage import VectorizedTrajectoryDataset
from iltools_datasets.trajopt.loader import TrajoptLoader

LoaderType = Union["LocoMuJoCoLoader", "AmassLoader", "TrajoptLoader"]


class TrajectoryDatasetManager:
    """
    Enhanced dataset manager that handles all trajectory operations internally.
    Provides a clean interface for the ImitationRLEnv with automatic COM and joint extraction.
    """

    def __init__(self, cfg: DictConfig, num_envs: int, device: torch.device) -> None:
        """
        Initialize the trajectory dataset manager.

        Args:
            cfg: Configuration object with dataset_path, assignment_strategy, etc.
            num_envs: Number of environments
            device: Torch device
        """
        self.cfg = cfg
        # store the joint sequence we need to export to, note that this must use the same joint names as the dataset
        assert cfg.target_joint_names is not None, (
            "target_joint_names must be provided in the config."
        )
        self.target_joint_names = cfg.target_joint_names

        # store the joint sequence we need to extract from the dataset.
        # This list of names might not be the same as provided in the dataset,
        # but should match the name string convention specified in target_joint_names
        assert cfg.reference_joint_names is not None, (
            "reference_joint_names must be provided in the config."
        )
        self.reference_joint_names = cfg.reference_joint_names

        self.num_envs = num_envs
        self._device = device  # type: ignore

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

        # Map reference to target joint names
        self.ref_to_target_map, self.target_to_ref_map = self._map_reference_to_target(
            self.reference_joint_names, self.target_joint_names
        )
        self.target_mask = torch.zeros(
            len(self.target_joint_names), dtype=torch.bool, device=device
        )
        self.target_mask[self.ref_to_target_map] = True

        # Memory allocate for important data
        self.joint_pos = torch.empty(
            num_envs, len(self.target_joint_names), device=device
        )
        self.joint_vel = torch.empty(
            num_envs, len(self.target_joint_names), device=device
        )
        self.root_pos = torch.empty(num_envs, 3, device=device, dtype=torch.float32)
        self.root_quat = torch.empty(num_envs, 4, device=device, dtype=torch.float32)
        self.root_lin_vel = torch.empty(num_envs, 3, device=device, dtype=torch.float32)
        self.root_ang_vel = torch.empty(num_envs, 3, device=device, dtype=torch.float32)

        # Pre-allocate reference data TensorDict for better performance
        self.reference_data = TensorDict(
            {
                "root_pos": self.root_pos,
                "root_quat": self.root_quat,
                "root_lin_vel": self.root_lin_vel,
                "root_ang_vel": self.root_ang_vel,
                "joint_pos": self.joint_pos,
                "joint_vel": self.joint_vel,
                "raw_qpos": torch.empty(
                    num_envs,
                    len(self.reference_joint_names),
                    device=device,
                    dtype=torch.float32,
                ),
                "raw_qvel": torch.empty(
                    num_envs,
                    len(self.reference_joint_names),
                    device=device,
                    dtype=torch.float32,
                ),
            },
            batch_size=[num_envs],
            device=device,
        )

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
            assert self.assignment_sequence is not None, (
                "assignment_sequence must be provided when using 'sequence' strategy"
            )
            assert isinstance(self.assignment_sequence, list)
            for env_id in env_ids:
                seq_idx = int(env_id)
                self.env2traj[env_id] = self.assignment_sequence[seq_idx]  # type: ignore
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
            int(env_id): int(self.env2traj[env_id]) for env_id in range(self.num_envs)
        }
        env_to_step = {
            int(env_id): int(self.env2step[env_id]) for env_id in range(self.num_envs)
        }
        self.dataset.update_references(env_to_traj=env_to_traj, env_to_step=env_to_step)

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
        self.reference_data["raw_qpos"] = qpos
        self.reference_data["raw_qvel"] = qvel

        # Extract root data (first 7 elements: x, y, z, qw, qx, qy, qz)
        root_pos = qpos[:, :3]  # (num_envs, 3)
        root_quat = qpos[:, 3:7]  # (num_envs, 4) - qw, qx, qy, qz
        self.reference_data["root_pos"] = root_pos
        self.reference_data["root_quat"] = root_quat

        # Extract COM velocities (first 6 elements: vx, vy, vz, wx, wy, wz)
        root_lin_vel = qvel[:, :3] if qvel is not None else torch.zeros_like(root_pos)
        root_ang_vel = qvel[:, 3:6] if qvel is not None else torch.zeros_like(root_pos)
        self.reference_data["root_lin_vel"] = root_lin_vel
        self.reference_data["root_ang_vel"] = root_ang_vel

        # Extract joint positions and velocities (skip first 7 elements for COM)
        joint_pos = qpos[:, 7:]  # (num_envs, num_joints)
        joint_vel = qvel[:, 6:] if qvel is not None else torch.zeros_like(joint_pos)

        self.joint_pos[..., self.ref_to_target_map] = joint_pos[
            ..., self.target_to_ref_map
        ]
        self.joint_vel[..., self.ref_to_target_map] = joint_vel[
            ..., self.target_to_ref_map
        ]
        self.joint_pos[..., ~self.target_mask] = torch.nan
        self.joint_vel[..., ~self.target_mask] = torch.nan
        self.reference_data["joint_pos"] = self.joint_pos.clone()
        self.reference_data["joint_vel"] = self.joint_vel.clone()

        # Increment step counters
        self.env2step += 1

        # Update dataset with new step positions
        env_to_step = {
            int(env_id): int(self.env2step[env_id].item())
            for env_id in range(self.num_envs)
        }
        self.dataset.update_references(env_to_step=env_to_step)

        return self.reference_data

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
            except:  # noqa: E722
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
        return os.path.exists(zarr_dir)

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

    def _get_loader(
        self, loader_type: str, loader_kwargs: dict[str, Any]
    ) -> LoaderType:
        """Get the appropriate loader based on type."""
        loader_kwargs.update({"cfg": self.cfg})
        if loader_type == "loco_mujoco":
            return LocoMuJoCoLoader(**loader_kwargs)
        elif loader_type == "amass":
            # return AmassLoader(**loader_kwargs)
            raise NotImplementedError("AmassLoader is not implemented")
        elif loader_type == "trajopt":
            # return TrajoptLoader(**loader_kwargs)
            raise NotImplementedError("TrajoptLoader is not implemented")
        else:
            raise ValueError(f"Unknown loader_type: {loader_type}")

    def _get_control_freq(self, cfg: Any) -> float:
        """Compute the control frequency from the config."""
        return 1.0 / (cfg.sim.dt * cfg.decimation)

    def _get_desired_horizon_steps(self, cfg: Any) -> int:
        """Compute the desired number of horizon steps from the config."""
        return int(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))

    def _map_reference_to_target(
        self, reference_joint_names: List[str], target_joint_names: List[str]
    ) -> Tuple[List[int], List[int]]:
        """
        Map the reference joint names to the target joint names; and return the target tensor and the mapping as a list of indices, so that the target tensor can be indexed by the indices, e.g., tensor[inv_map] = reference_tensor[map] produces a tensor with the same shape as the target tensor but the values are the same as the reference tensor re-ordered. Also, tensor[~inv_map] = NaN.

        Args:
            reference_joint_names: List of reference joint names
            target_joint_names: List of target joint names

        Returns:
            Tuple containing:
                - mapping: List of indices for mapping
                - inv_map: List of indices for inverse mapping
        """
        # Create mapping from reference to target joint positions
        mapping = []
        inv_map = []
        all_joint_names = list(set(target_joint_names + reference_joint_names))
        for joint_name in all_joint_names:
            if (
                joint_name not in target_joint_names
                or joint_name not in reference_joint_names
            ):
                continue
            map_idx = target_joint_names.index(joint_name)
            mapping.append(map_idx)
            inv_map_idx = reference_joint_names.index(joint_name)
            inv_map.append(inv_map_idx)

        return mapping, inv_map
