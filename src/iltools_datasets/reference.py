from __future__ import annotations

"""Lightweight reference trajectory utilities built on top of the replay stack.

This module replaces the older Zarr-based TrajectoryDatasetManager with a much
smaller, replay-native helper that:

- Assumes trajectories have been exported to a memmap-backed replay buffer
  via `build_replay_from_zarr(...)` and are served through `ExpertReplayManager`.
- Uses `sample_per_env_window(...)` to fetch per-environment windows of `qpos`
  and `qvel` (or user-specified keys).
- Extracts root and joint features in a shape convenient for locomotion agents.

It deliberately avoids any dependency on loco-mujoco or JAX; it only needs
PyTorch and the replay manager.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from tensordict import TensorDict

from .replay_manager import ExpertReplayManager


@dataclass(frozen=True)
class JointMapping:
    """Mapping from reference joint names to target joint names.

    This lets you describe the naming in your offline dataset ('reference') and
    how joints should be ordered in the consumer ('target'). Only joints that
    appear in both lists are exported; missing joints are filled with NaNs on
    the target side.
    """

    reference_joint_names: List[str]
    target_joint_names: List[str]

    def build_index_maps(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (ref_to_target, target_to_ref, target_mask) index tensors.

        Shapes:
          - ref_to_target: [num_common] indices into `target_joint_names`
          - target_to_ref: [num_common] indices into `reference_joint_names`
          - target_mask:  [len(target_joint_names)] bool mask; True where joint
            name exists in both reference and target.
        """
        mapping: list[int] = []
        inv_map: list[int] = []
        all_joint_names = list(
            set(self.target_joint_names + self.reference_joint_names)
        )
        for joint_name in all_joint_names:
            if (
                joint_name not in self.target_joint_names
                or joint_name not in self.reference_joint_names
            ):
                continue
            target_idx = self.target_joint_names.index(joint_name)
            ref_idx = self.reference_joint_names.index(joint_name)
            mapping.append(target_idx)
            inv_map.append(ref_idx)

        ref_to_target = torch.tensor(mapping, dtype=torch.long)
        target_to_ref = torch.tensor(inv_map, dtype=torch.long)
        target_mask = torch.zeros(
            len(self.target_joint_names), dtype=torch.bool
        )
        if ref_to_target.numel() > 0:
            target_mask[ref_to_target] = True
        return ref_to_target, target_to_ref, target_mask


class ReferenceManager:
    """Small helper around `ExpertReplayManager` for locomotion-style features.

    It expects the replay trajectories to contain:
      - a `qpos`-like vector with layout:
          [root_pos(3), root_quat(4), joint_pos(...)]
      - optionally a `qvel`-like vector with layout:
          [root_lin_vel(3), root_ang_vel(3), joint_vel(...)]

    Given a `JointMapping`, it produces per-env (and optionally per-time-step)
    tensors for:
      - root_pos, root_quat, root_lin_vel, root_ang_vel
      - joint_pos, joint_vel
      - raw_qpos, raw_qvel

    All indexing over (env, traj, step) is delegated to `ExpertReplayManager`
    via `sample_per_env_window(...)`; this class is purely about feature
    extraction and joint name mapping.
    """

    def __init__(
        self,
        replay_manager: ExpertReplayManager,
        *,
        joint_mapping: JointMapping,
        qpos_key: str = "qpos",
        qvel_key: str = "qvel",
        device: torch.device | None = None,
    ) -> None:
        self.replay = replay_manager
        self.qpos_key = qpos_key
        self.qvel_key = qvel_key

        self.ref_to_target, self.target_to_ref, self.target_mask = (
            joint_mapping.build_index_maps()
        )
        self.device = device

    def _to_device(self, t: torch.Tensor) -> torch.Tensor:
        if self.device is None:
            return t
        return t.to(self.device)

    def get_reference_window(
        self,
        window_size: int = 1,
        *,
        wrap: bool = True,
    ) -> TensorDict:
        """Return a window of reference data per env as a TensorDict.

        Shape conventions:
          - If `window_size == 1`: batch_size = [num_envs]
          - Else:                  batch_size = [num_envs, window_size]

        Keys:
          - root_pos:  [..., 3]
          - root_quat: [..., 4]
          - root_lin_vel: [..., 3]
          - root_ang_vel: [..., 3]
          - joint_pos:  [..., num_target_joints]
          - joint_vel:  [..., num_target_joints]
          - raw_qpos:   [..., qpos_dim]
          - raw_qvel:   [..., qvel_dim] (zeros if missing)
        """
        td_window = self.replay.sample_per_env_window(window_size=window_size, wrap=wrap)

        qpos = self._to_device(td_window[self.qpos_key])
        qvel = (
            self._to_device(td_window[self.qvel_key])
            if self.qvel_key in td_window.keys(True)
            else None
        )

        # Ensure we always treat the last dimension as feature dim
        *batch_shape, qpos_dim = qpos.shape

        # Root pose from qpos: [x,y,z, qw,qx,qy,qz]
        root_pos = qpos[..., 0:3]
        root_quat = qpos[..., 3:7]

        # Velocities from qvel if present; otherwise zeros
        if qvel is not None:
            root_lin_vel = qvel[..., 0:3]
            root_ang_vel = qvel[..., 3:6]
        else:
            root_lin_vel = torch.zeros_like(root_pos)
            root_ang_vel = torch.zeros_like(root_pos)

        # Joint positions / velocities from the remainder of qpos/qvel
        joint_pos_ref = qpos[..., 7:]
        if qvel is not None and qvel.shape[-1] > 6:
            joint_vel_ref = qvel[..., 6:]
        else:
            joint_vel_ref = torch.zeros_like(joint_pos_ref)

        num_target = len(self.target_mask)
        target_shape = (*batch_shape, num_target)
        joint_pos = torch.empty(target_shape, dtype=joint_pos_ref.dtype, device=joint_pos_ref.device)
        joint_vel = torch.empty_like(joint_pos)

        if self.ref_to_target.numel() > 0:
            joint_pos[..., self.ref_to_target] = joint_pos_ref[..., self.target_to_ref]
            joint_vel[..., self.ref_to_target] = joint_vel_ref[..., self.target_to_ref]

        # Fill joints not present in reference with NaNs
        missing_mask = ~self.target_mask
        if torch.any(missing_mask):
            joint_pos[..., missing_mask] = torch.nan
            joint_vel[..., missing_mask] = torch.nan

        out = TensorDict(
            {
                "root_pos": root_pos,
                "root_quat": root_quat,
                "root_lin_vel": root_lin_vel,
                "root_ang_vel": root_ang_vel,
                "joint_pos": joint_pos,
                "joint_vel": joint_vel,
                "raw_qpos": qpos,
                "raw_qvel": qvel
                if qvel is not None
                else torch.zeros_like(qpos),
            },
            batch_size=batch_shape,
            device=root_pos.device,
        )

        # For window_size==1, it is often convenient to squeeze the time axis.
        # Callers that need the explicit [E,1,...] shape can pass window_size>1
        # or handle the broadcasting themselves.
        return out


