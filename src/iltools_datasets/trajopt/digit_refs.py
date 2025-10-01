from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class DigitRefSpec:
    a_pos_index: np.ndarray
    a_vel_index: np.ndarray
    subsample: int = 1


class DigitTrajoptRefBuilder:
    """Build Digit reference arrays from trajopt-produced .npz files.

    - No full-directory preload; index files and lazily load one trajectory.
    - Computes derived references and pads to a fixed max length for JIT/vmap safety.
    - Returns numpy arrays; callers convert to framework tensors as needed.
    """

    def __init__(self, data_dir: str, ref_spec: DigitRefSpec):
        self.data_dir = data_dir
        self.spec = ref_spec
        self.files: List[str] = self._list_npz_files(data_dir)
        if not self.files:
            raise FileNotFoundError(f"No .npz files found under: {data_dir}")
        self.lengths: List[int] = [self._probe_length(p) for p in self.files]
        self.max_len: int = int(max(self.lengths))

    @property
    def num_trajectories(self) -> int:
        return len(self.files)

    def _list_npz_files(self, root: str) -> List[str]:
        if os.path.isdir(root):
            return sorted(
                [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".npz")]
            )
        if os.path.isfile(root) and root.endswith(".npz"):
            return [root]
        return []

    def _probe_length(self, path: str) -> int:
        with np.load(path) as data:
            if "true_length" in data:
                n = int(np.asarray(data["true_length"]))
            else:
                n = int(data["qpos"].shape[0])
        return max(1, n // max(1, int(self.spec.subsample)))

    def build_by_index(self, traj_index: int) -> Dict[str, np.ndarray]:
        assert 0 <= traj_index < len(self.files)
        return self._build_from_file(self.files[traj_index])

    def _pad_time(self, arr: np.ndarray, T: int) -> np.ndarray:
        if arr.shape[0] >= T:
            return arr[:T]
        pad_len = T - arr.shape[0]
        pad_slice = arr[-1:] if arr.ndim == 1 else arr[-1:, ...]
        return np.concatenate([arr, np.repeat(pad_slice, pad_len, axis=0)], axis=0)

    def _quat2euler_batch(self, quat_wxyz: np.ndarray) -> np.ndarray:
        w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.stack([roll, pitch, yaw], axis=-1)

    def _euler2mat_batch(self, euler_rpy: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = euler_rpy[:, 0], euler_rpy[:, 1], euler_rpy[:, 2]
        cx, sx = np.cos(roll), np.sin(roll)
        cy, sy = np.cos(pitch), np.sin(pitch)
        cz, sz = np.cos(yaw), np.sin(yaw)
        rot_x = np.stack(
            [
                np.ones_like(cx),
                np.zeros_like(cx),
                np.zeros_like(cx),
                np.zeros_like(cx),
                cx,
                -sx,
                np.zeros_like(cx),
                sx,
                cx,
            ],
            axis=-1,
        ).reshape(-1, 3, 3)
        rot_y = np.stack(
            [
                cy,
                np.zeros_like(cy),
                sy,
                np.zeros_like(cy),
                np.ones_like(cy),
                np.zeros_like(cy),
                -sy,
                np.zeros_like(cy),
                cy,
            ],
            axis=-1,
        ).reshape(-1, 3, 3)
        rot_z = np.stack(
            [
                cz,
                -sz,
                np.zeros_like(cz),
                sz,
                cz,
                np.zeros_like(cz),
                np.zeros_like(cz),
                np.zeros_like(cz),
                np.ones_like(cz),
            ],
            axis=-1,
        ).reshape(-1, 3, 3)
        return rot_z @ rot_y @ rot_x

    def _build_from_file(self, path: str) -> Dict[str, np.ndarray]:
        step = max(1, int(self.spec.subsample))
        with np.load(path) as data:
            qpos = np.asarray(data["qpos"])[::step]
            qvel = np.asarray(data["qvel"])[::step]
            if "ee_pos" in data:
                ee_pos = np.asarray(data["ee_pos"]).reshape(-1, 5, 3)[::step]
            else:
                ee_pos = np.zeros((qpos.shape[0], 5, 3), dtype=qpos.dtype)

        base_quat_wxyz = qpos[:, [6, 3, 4, 5]]
        base_euler_local = self._quat2euler_batch(base_quat_wxyz)
        base_pos = qpos[:, :3]
        base_trans = base_pos - base_pos[0]
        base_rot_local = self._euler2mat_batch(
            np.stack(
                [
                    np.zeros_like(base_euler_local[:, 0]),
                    np.zeros_like(base_euler_local[:, 1]),
                    -base_euler_local[:, 2],
                ],
                axis=-1,
            )
        )
        base_rot_robot = self._euler2mat_batch(base_euler_local)

        lin_vel_robot = np.einsum(
            "bij,bj->bi", base_rot_robot.transpose(0, 2, 1), qvel[:, :3]
        )
        ang_vel_robot = np.einsum(
            "bij,bj->bi", base_rot_robot.transpose(0, 2, 1), qvel[:, 3:6]
        )

        base_local_ee_pos = np.einsum(
            "bij,bkj->bki", base_rot_local, ee_pos - base_pos[:, None, :]
        )

        motor_joint_pos = qpos[:, self.spec.a_pos_index]
        motor_joint_vel = qvel[:, self.spec.a_vel_index]

        T = qpos.shape[0]
        maxT = self.max_len
        return {
            "ref_len": np.array([T], dtype=np.int32),
            "ref_motor_joint_pos": self._pad_time(motor_joint_pos, maxT),
            "ref_motor_joint_vel": self._pad_time(motor_joint_vel, maxT),
            "ref_base_local_pos": self._pad_time(base_pos, maxT),
            "ref_base_local_trans": self._pad_time(base_trans, maxT),
            "ref_base_local_ori": self._pad_time(base_euler_local, maxT),
            "ref_base_robot_lin_vel": self._pad_time(lin_vel_robot, maxT),
            "ref_base_robot_ang_vel": self._pad_time(ang_vel_robot, maxT),
            "ref_local_ee_pos": self._pad_time(ee_pos, maxT),
            "ref_base_local_ee_pos": self._pad_time(base_local_ee_pos, maxT),
        }


def build_refs_from_window(
    qpos: np.ndarray,
    qvel: np.ndarray,
    spec: DigitRefSpec,
    ee_pos: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build Digit reference arrays from a fixed-size window (no padding).

    Args:
        qpos: [T, nq]
        qvel: [T, nv]
        spec: DigitRefSpec with actuator indices
        ee_pos: optional [T, 5, 3]

    Returns:
        dict with keys like 'ref_motor_joint_pos', 'ref_base_local_ori', ... of length T.
    """
    if ee_pos is None:
        ee_pos = np.zeros((qpos.shape[0], 5, 3), dtype=qpos.dtype)

    base_quat_wxyz = qpos[:, [6, 3, 4, 5]]
    w, x, y, z = (
        base_quat_wxyz[:, 0],
        base_quat_wxyz[:, 1],
        base_quat_wxyz[:, 2],
        base_quat_wxyz[:, 3],
    )
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    base_euler_local = np.stack([roll, pitch, yaw], axis=-1)

    base_pos = qpos[:, :3]
    base_trans = base_pos - base_pos[0]

    # Rotations
    def euler2mat_batch(euler_rpy: np.ndarray) -> np.ndarray:
        r, p, yw = euler_rpy[:, 0], euler_rpy[:, 1], euler_rpy[:, 2]
        cx, sx = np.cos(r), np.sin(r)
        cy, sy = np.cos(p), np.sin(p)
        cz, sz = np.cos(yw), np.sin(yw)
        rot_x = np.stack(
            [
                np.ones_like(cx),
                np.zeros_like(cx),
                np.zeros_like(cx),
                np.zeros_like(cx),
                cx,
                -sx,
                np.zeros_like(cx),
                sx,
                cx,
            ],
            axis=-1,
        ).reshape(-1, 3, 3)
        rot_y = np.stack(
            [
                cy,
                np.zeros_like(cy),
                sy,
                np.zeros_like(cy),
                np.ones_like(cy),
                np.zeros_like(cy),
                -sy,
                np.zeros_like(cy),
                cy,
            ],
            axis=-1,
        ).reshape(-1, 3, 3)
        rot_z = np.stack(
            [
                cz,
                -sz,
                np.zeros_like(cz),
                sz,
                cz,
                np.zeros_like(cz),
                np.zeros_like(cz),
                np.zeros_like(cz),
                np.ones_like(cz),
            ],
            axis=-1,
        ).reshape(-1, 3, 3)
        return rot_z @ rot_y @ rot_x

    base_rot_local = euler2mat_batch(
        np.stack(
            [
                np.zeros_like(base_euler_local[:, 0]),
                np.zeros_like(base_euler_local[:, 1]),
                -base_euler_local[:, 2],
            ],
            axis=-1,
        )
    )
    base_rot_robot = euler2mat_batch(base_euler_local)

    lin_vel_robot = np.einsum(
        "bij,bj->bi", base_rot_robot.transpose(0, 2, 1), qvel[:, :3]
    )
    ang_vel_robot = np.einsum(
        "bij,bj->bi", base_rot_robot.transpose(0, 2, 1), qvel[:, 3:6]
    )
    base_local_ee_pos = np.einsum(
        "bij,bkj->bki", base_rot_local, ee_pos - base_pos[:, None, :]
    )

    motor_joint_pos = qpos[:, spec.a_pos_index]
    motor_joint_vel = qvel[:, spec.a_vel_index]

    return {
        "ref_len": np.array([qpos.shape[0]], dtype=np.int32),
        "ref_motor_joint_pos": motor_joint_pos,
        "ref_motor_joint_vel": motor_joint_vel,
        "ref_base_local_pos": base_pos,
        "ref_base_local_trans": base_trans,
        "ref_base_local_ori": base_euler_local,
        "ref_base_robot_lin_vel": lin_vel_robot,
        "ref_base_robot_ang_vel": ang_vel_robot,
        "ref_local_ee_pos": ee_pos,
        "ref_base_local_ee_pos": base_local_ee_pos,
    }
