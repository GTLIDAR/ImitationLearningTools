"""Fixed-base dual-wrist pose and finger-target retargeting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import mujoco as _mujoco
import numpy as np

from iltools.core.dexterous_reference import (
    CollisionAssetDependency,
    ContactSequence,
    DexterousReference,
    ScenePhysics,
    TrainingQualification,
)
from iltools.core.trajectory import Trajectory

from .base_retarget import BaseRetarget
from .keypoint_retarget import _retargeted_trajectory

mujoco: Any = _mujoco


def _unit_wxyz(quaternion: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(norm <= 1.0e-8) or not np.isfinite(value).all():
        raise ValueError(f"{name} contains an invalid quaternion.")
    if not np.allclose(norm, 1.0, rtol=0.0, atol=1.0e-4):
        raise ValueError(f"{name} must contain unit WXYZ quaternions.")
    return value / norm


def _quaternion_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _orientation_error_w(
    target_wxyz: np.ndarray,
    current_wxyz: np.ndarray,
) -> np.ndarray:
    """Return the shortest world-frame rotation vector from current to target."""

    conjugate = np.asarray(
        [current_wxyz[0], -current_wxyz[1], -current_wxyz[2], -current_wxyz[3]]
    )
    error = _quaternion_multiply_wxyz(target_wxyz, conjugate)
    if error[0] < 0.0:
        error = -error
    vector_norm = float(np.linalg.norm(error[1:]))
    if vector_norm <= 1.0e-10:
        return 2.0 * error[1:]
    angle = 2.0 * np.arctan2(vector_norm, np.clip(error[0], -1.0, 1.0))
    return error[1:] * (angle / vector_norm)


def _poses_from_robot_to_world(
    poses: np.ndarray, root_pose_w: np.ndarray
) -> np.ndarray:
    root = np.asarray(root_pose_w, dtype=np.float64)
    if root.shape != (7,):
        raise ValueError("fixed_root_pose_w must have shape [7].")
    root_quaternion = _unit_wxyz(root[3:7], name="fixed_root_pose_w")
    rotation_flat = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation_flat, root_quaternion)
    rotation = rotation_flat.reshape(3, 3)
    result = np.asarray(poses, dtype=np.float64).copy()
    if result.shape[-1] != 7:
        raise ValueError("Pose arrays must end with seven values.")
    result[..., :3] = result[..., :3] @ rotation.T + root[:3]
    flat_quaternions = result[..., 3:7].reshape(-1, 4)
    for index in range(len(flat_quaternions)):
        flat_quaternions[index] = _quaternion_multiply_wxyz(
            root_quaternion, flat_quaternions[index]
        )
    result[..., 3:7] = _unit_wxyz(result[..., 3:7], name="world poses")
    return result


def _twists_from_robot_to_world(
    twists: np.ndarray, root_pose_w: np.ndarray
) -> np.ndarray:
    """Rotate linear-then-angular twists from robot axes to world axes."""

    root = np.asarray(root_pose_w, dtype=np.float64)
    if root.shape != (7,):
        raise ValueError("fixed_root_pose_w must have shape [7].")
    root_quaternion = _unit_wxyz(root[3:7], name="fixed_root_pose_w")
    rotation_flat = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation_flat, root_quaternion)
    rotation = rotation_flat.reshape(3, 3)
    result = np.asarray(twists, dtype=np.float64).copy()
    if result.ndim != 3 or result.shape[-1] != 6:
        raise ValueError("Object twists must have shape [T, O, 6].")
    result[..., :3] = result[..., :3] @ rotation.T
    result[..., 3:6] = result[..., 3:6] @ rotation.T
    return result


def _contacts_from_robot_to_world(
    contacts: ContactSequence, root_pose_w: np.ndarray
) -> ContactSequence:
    root = np.asarray(root_pose_w, dtype=np.float64)
    root_quaternion = _unit_wxyz(root[3:7], name="fixed_root_pose_w")
    rotation_flat = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation_flat, root_quaternion)
    rotation = rotation_flat.reshape(3, 3)

    def transform_positions(values: np.ndarray) -> np.ndarray:
        return np.asarray(values) @ rotation.T + root[:3]

    def transform_normals(values: np.ndarray) -> np.ndarray:
        return np.asarray(values) @ rotation.T

    return ContactSequence(
        hand_sides=contacts.hand_sides,
        link_names=contacts.link_names,
        link_positions_w=transform_positions(contacts.link_positions_w),
        link_normals_w=transform_normals(contacts.link_normals_w),
        object_positions_w=transform_positions(contacts.object_positions_w),
        object_normals_w=transform_normals(contacts.object_normals_w),
        object_indices=contacts.object_indices,
        active=contacts.active,
    )


class MujocoDualHandRetargeter(BaseRetarget):
    """Solve two wrist SE(3) targets and copy bounded robot finger targets.

    The wrist targets are poses of the named MuJoCo sites. They must already be
    in the MuJoCo robot frame and use pose order
    ``[x, y, z, qw, qx, qy, qz]``. The solver changes only the named arm
    joints. Finger arrays are robot targets, not raw human joint angles; use
    ``KeypointRetargeter`` or ``JointMapRetargeter`` before this stage.
    Previous-frame arm positions provide the seed for the next frame.
    """

    def __init__(
        self,
        model: mujoco.MjModel | str | Path,
        *,
        target_joint_names: Sequence[str],
        arm_joint_names: Sequence[str],
        left_finger_joint_names: Sequence[str],
        right_finger_joint_names: Sequence[str],
        left_wrist_site: str,
        right_wrist_site: str,
        left_wrist_key: str = "left_wrist_pose_w",
        right_wrist_key: str = "right_wrist_pose_w",
        left_finger_key: str = "left_finger_qpos",
        right_finger_key: str = "right_finger_qpos",
        iterations: int = 100,
        damping: float = 0.02,
        max_step: float = 0.12,
        tolerance: float = 1.0e-4,
        position_weight: float = 1.0,
        orientation_weight: float = 0.25,
        previous_posture_weight: float = 0.0,
        neutral_posture_weight: float = 0.0,
        position_priority: bool = False,
        position_priority_slack: float = 0.0,
        initial_qpos: Sequence[float] | None = None,
        restart_position_error_m: float | None = None,
    ) -> None:
        if isinstance(model, (str, Path)):
            model_path = Path(model).expanduser().resolve()
            if not model_path.is_file():
                raise FileNotFoundError(f"MuJoCo model not found: {model_path}")
            self.model = mujoco.MjModel.from_xml_path(str(model_path))
        else:
            self.model = model

        self.target_joint_names = tuple(target_joint_names)
        self.arm_joint_names = tuple(arm_joint_names)
        self.left_finger_joint_names = tuple(left_finger_joint_names)
        self.right_finger_joint_names = tuple(right_finger_joint_names)
        self.left_wrist_key = left_wrist_key
        self.right_wrist_key = right_wrist_key
        self.left_wrist_site = str(left_wrist_site)
        self.right_wrist_site = str(right_wrist_site)
        self.left_finger_key = left_finger_key
        self.right_finger_key = right_finger_key
        self._validate_joint_groups()
        if iterations < 1:
            raise ValueError("iterations must be positive.")
        if (
            not np.isfinite(
                (
                    damping,
                    max_step,
                    tolerance,
                    position_weight,
                    orientation_weight,
                    previous_posture_weight,
                    neutral_posture_weight,
                    position_priority_slack,
                )
            ).all()
            or min(
                damping,
                max_step,
                tolerance,
                position_weight,
                orientation_weight,
            )
            <= 0.0
            or previous_posture_weight < 0.0
            or neutral_posture_weight < 0.0
            or position_priority_slack < 0.0
        ):
            raise ValueError(
                "Solver parameters and task weights must be positive; posture "
                "weights and position-priority slack must be non-negative."
            )
        self.iterations = int(iterations)
        self.damping = float(damping)
        self.max_step = float(max_step)
        self.tolerance = float(tolerance)
        self.position_weight = float(position_weight)
        self.orientation_weight = float(orientation_weight)
        self.previous_posture_weight = float(previous_posture_weight)
        self.neutral_posture_weight = float(neutral_posture_weight)
        self.position_priority = bool(position_priority)
        self.position_priority_slack = float(position_priority_slack)
        if restart_position_error_m is not None and (
            not np.isfinite(restart_position_error_m) or restart_position_error_m <= 0
        ):
            raise ValueError("The restart position error must be positive.")
        self.restart_position_error_m = restart_position_error_m
        self._task_weights = np.tile(
            np.sqrt(
                np.asarray(
                    [position_weight] * 3 + [orientation_weight] * 3,
                    dtype=np.float64,
                )
            ),
            2,
        )

        self._target_joint_ids = np.asarray(
            [self.model.joint(name).id for name in self.target_joint_names],
            dtype=np.int32,
        )
        unsupported = [
            name
            for name, joint_id in zip(
                self.target_joint_names, self._target_joint_ids, strict=True
            )
            if int(self.model.jnt_type[joint_id])
            not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            )
        ]
        if unsupported:
            raise ValueError(
                "Dual-hand retargeting supports hinge and slide joints only: "
                f"{unsupported}."
            )
        self._target_qpos_indices = np.asarray(
            self.model.jnt_qposadr[self._target_joint_ids], dtype=np.int32
        )
        self._target_dof_indices = np.asarray(
            self.model.jnt_dofadr[self._target_joint_ids], dtype=np.int32
        )
        target_index = {
            name: index for index, name in enumerate(self.target_joint_names)
        }
        self._arm_target_indices = np.asarray(
            [target_index[name] for name in self.arm_joint_names], dtype=np.int32
        )
        self._arm_dof_indices = self._target_dof_indices[self._arm_target_indices]
        self._left_finger_target_indices = np.asarray(
            [target_index[name] for name in self.left_finger_joint_names],
            dtype=np.int32,
        )
        self._right_finger_target_indices = np.asarray(
            [target_index[name] for name in self.right_finger_joint_names],
            dtype=np.int32,
        )
        self._site_ids = (
            self.model.site(self.left_wrist_site).id,
            self.model.site(self.right_wrist_site).id,
        )

        data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, data)
        self._base_qpos = data.qpos.copy()
        if initial_qpos is None:
            self._initial_qpos = self._base_qpos[self._target_qpos_indices].copy()
        else:
            self._initial_qpos = np.asarray(initial_qpos, dtype=np.float64)
            if self._initial_qpos.shape != (len(self.target_joint_names),):
                raise ValueError(
                    "initial_qpos must align with target_joint_names; got "
                    f"{self._initial_qpos.shape}."
                )
            if not np.isfinite(self._initial_qpos).all():
                raise ValueError("initial_qpos contains non-finite values.")
        self._lower, self._upper = self._joint_limits()
        self._initial_qpos = np.clip(self._initial_qpos, self._lower, self._upper)

    @staticmethod
    def _null_space(jacobian: np.ndarray) -> np.ndarray:
        """Return an orthonormal basis for the exact numerical null space."""

        matrix = np.asarray(jacobian, dtype=np.float64)
        _, singular_values, right_vectors = np.linalg.svd(matrix, full_matrices=True)
        largest = float(singular_values[0]) if len(singular_values) else 0.0
        tolerance = (
            max(matrix.shape) * np.finfo(np.float64).eps * largest
            if largest > 0.0
            else 0.0
        )
        rank = int(np.count_nonzero(singular_values > tolerance))
        return right_vectors[rank:].T.copy()

    def _weighted_delta(
        self,
        *,
        position_error: np.ndarray,
        orientation_error: np.ndarray,
        position_jacobian: np.ndarray,
        orientation_jacobian: np.ndarray,
        current_arm: np.ndarray,
        previous: np.ndarray,
        neutral: np.ndarray,
    ) -> np.ndarray:
        position_errors = position_error.reshape(2, 3)
        orientation_errors = orientation_error.reshape(2, 3)
        position_jacobians = position_jacobian.reshape(
            2, 3, len(self._arm_target_indices)
        )
        orientation_jacobians = orientation_jacobian.reshape(
            2, 3, len(self._arm_target_indices)
        )
        error = np.concatenate(
            tuple(
                np.concatenate((position_errors[index], orientation_errors[index]))
                for index in range(2)
            )
        )
        jacobian = np.vstack(
            tuple(
                np.vstack((position_jacobians[index], orientation_jacobians[index]))
                for index in range(2)
            )
        )
        error *= self._task_weights
        jacobian *= self._task_weights[:, None]
        normal = jacobian.T @ jacobian
        normal += self.damping**2 * np.eye(len(self._arm_target_indices))
        rhs = jacobian.T @ error
        if self.previous_posture_weight > 0.0:
            normal += self.previous_posture_weight * np.eye(
                len(self._arm_target_indices)
            )
            rhs += self.previous_posture_weight * (previous - current_arm)
        if self.neutral_posture_weight > 0.0:
            normal += self.neutral_posture_weight * np.eye(
                len(self._arm_target_indices)
            )
            rhs += self.neutral_posture_weight * (neutral - current_arm)
        return np.linalg.solve(normal, rhs)

    def _position_priority_delta(
        self,
        *,
        position_error: np.ndarray,
        orientation_error: np.ndarray,
        position_jacobian: np.ndarray,
        orientation_jacobian: np.ndarray,
        current_arm: np.ndarray,
        previous: np.ndarray,
        neutral: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve position, orientation, then posture in strict null spaces."""

        joint_count = len(self._arm_target_indices)
        identity = np.eye(joint_count)
        primary_normal = (
            self.position_weight * position_jacobian.T @ position_jacobian
            + self.damping**2 * identity
        )
        primary_rhs = self.position_weight * position_jacobian.T @ position_error
        primary_delta = np.linalg.solve(primary_normal, primary_rhs)

        position_null = self._null_space(position_jacobian)
        if position_null.shape[1] == 0:
            return primary_delta, np.zeros_like(primary_delta)

        secondary_jacobian = orientation_jacobian @ position_null
        secondary_residual = orientation_error - orientation_jacobian @ primary_delta
        secondary_identity = np.eye(position_null.shape[1])
        secondary_normal = (
            self.orientation_weight * secondary_jacobian.T @ secondary_jacobian
            + self.damping**2 * secondary_identity
        )
        secondary_rhs = (
            self.orientation_weight * secondary_jacobian.T @ secondary_residual
        )
        secondary_coordinates = np.linalg.solve(secondary_normal, secondary_rhs)
        secondary_delta = position_null @ secondary_coordinates
        delta = primary_delta + secondary_delta

        posture_weight = self.previous_posture_weight + self.neutral_posture_weight
        if posture_weight <= 0.0:
            return primary_delta, secondary_delta
        orientation_null = self._null_space(secondary_jacobian)
        if orientation_null.shape[1] == 0:
            return primary_delta, secondary_delta
        task_null = position_null @ orientation_null
        posture_rhs = np.zeros(task_null.shape[1], dtype=np.float64)
        if self.previous_posture_weight > 0.0:
            posture_rhs += (
                self.previous_posture_weight
                * task_null.T
                @ (previous - current_arm - delta)
            )
        if self.neutral_posture_weight > 0.0:
            posture_rhs += (
                self.neutral_posture_weight
                * task_null.T
                @ (neutral - current_arm - delta)
            )
        posture_normal = posture_weight * np.eye(
            task_null.shape[1]
        ) + self.damping**2 * np.eye(task_null.shape[1])
        posture_delta = task_null @ np.linalg.solve(posture_normal, posture_rhs)
        return primary_delta, secondary_delta + posture_delta

    def _position_objective(
        self,
        data: mujoco.MjData,
        q: np.ndarray,
        target_poses: tuple[np.ndarray, np.ndarray],
    ) -> float:
        data.qpos[:] = self._base_qpos
        data.qpos[self._target_qpos_indices] = q
        mujoco.mj_forward(self.model, data)
        return float(
            sum(
                np.dot(
                    target_pose[:3] - data.site_xpos[site_id],
                    target_pose[:3] - data.site_xpos[site_id],
                )
                for site_id, target_pose in zip(
                    self._site_ids, target_poses, strict=True
                )
            )
        )

    def _validate_joint_groups(self) -> None:
        groups = (
            self.target_joint_names,
            self.arm_joint_names,
            self.left_finger_joint_names,
            self.right_finger_joint_names,
        )
        if any(not group or len(set(group)) != len(group) for group in groups):
            raise ValueError("All dual-hand joint groups must be non-empty and unique.")
        task_names = (
            set(self.arm_joint_names)
            | set(self.left_finger_joint_names)
            | set(self.right_finger_joint_names)
        )
        if not task_names.issubset(self.target_joint_names):
            missing = task_names - set(self.target_joint_names)
            raise ValueError(
                f"Task joints are missing from target_joint_names: {missing}."
            )
        if (
            set(self.arm_joint_names) & set(self.left_finger_joint_names)
            or set(self.arm_joint_names) & set(self.right_finger_joint_names)
            or set(self.left_finger_joint_names) & set(self.right_finger_joint_names)
        ):
            raise ValueError(
                "Arm, left-finger, and right-finger groups must be disjoint."
            )

    def _joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(len(self._target_joint_ids), -np.inf, dtype=np.float64)
        upper = np.full(len(self._target_joint_ids), np.inf, dtype=np.float64)
        for index, joint_id in enumerate(self._target_joint_ids):
            if self.model.jnt_limited[joint_id]:
                lower[index], upper[index] = self.model.jnt_range[joint_id]
        return lower, upper

    def _solve_frame(
        self,
        data: mujoco.MjData,
        target_poses: tuple[np.ndarray, np.ndarray],
        seed: np.ndarray,
        previous_arm_qpos: np.ndarray,
    ) -> np.ndarray:
        q = np.array(seed, dtype=np.float64, copy=True)
        previous = np.asarray(previous_arm_qpos, dtype=np.float64)
        if previous.shape != (len(self._arm_target_indices),):
            raise ValueError("previous_arm_qpos must align with arm_joint_names.")
        neutral = self._initial_qpos[self._arm_target_indices]
        for _ in range(self.iterations):
            data.qpos[:] = self._base_qpos
            data.qpos[self._target_qpos_indices] = q
            mujoco.mj_forward(self.model, data)
            position_errors: list[np.ndarray] = []
            orientation_errors: list[np.ndarray] = []
            position_jacobians: list[np.ndarray] = []
            orientation_jacobians: list[np.ndarray] = []
            for site_id, target_pose in zip(self._site_ids, target_poses, strict=True):
                current_quaternion = np.empty(4, dtype=np.float64)
                mujoco.mju_mat2Quat(current_quaternion, data.site_xmat[site_id])
                position_error = target_pose[:3] - data.site_xpos[site_id]
                orientation_error = _orientation_error_w(
                    target_pose[3:7], current_quaternion
                )
                position_errors.append(position_error)
                orientation_errors.append(orientation_error)
                position_jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
                rotation_jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
                mujoco.mj_jacSite(
                    self.model,
                    data,
                    position_jacobian,
                    rotation_jacobian,
                    site_id,
                )
                position_jacobians.append(position_jacobian[:, self._arm_dof_indices])
                orientation_jacobians.append(
                    rotation_jacobian[:, self._arm_dof_indices]
                )
            position_error_array = np.asarray(position_errors)
            orientation_error_array = np.asarray(orientation_errors)
            weighted_error = (
                np.concatenate(
                    tuple(
                        np.concatenate(
                            (
                                position_error_array[index],
                                orientation_error_array[index],
                            )
                        )
                        for index in range(2)
                    )
                )
                * self._task_weights
            )
            if np.linalg.norm(weighted_error) <= self.tolerance:
                break
            current_arm = q[self._arm_target_indices]
            solve_arguments = {
                "position_error": position_error_array.reshape(-1),
                "orientation_error": orientation_error_array.reshape(-1),
                "position_jacobian": np.vstack(position_jacobians),
                "orientation_jacobian": np.vstack(orientation_jacobians),
                "current_arm": current_arm,
                "previous": previous,
                "neutral": neutral,
            }
            if not self.position_priority:
                delta = self._weighted_delta(**solve_arguments)
                norm = float(np.linalg.norm(delta))
                if norm > self.max_step:
                    delta *= self.max_step / norm
                q[self._arm_target_indices] = np.clip(
                    current_arm + delta,
                    self._lower[self._arm_target_indices],
                    self._upper[self._arm_target_indices],
                )
                continue

            primary_delta, lower_priority_delta = self._position_priority_delta(
                **solve_arguments
            )
            primary_norm = float(np.linalg.norm(primary_delta))
            if primary_norm > self.max_step:
                primary_delta *= self.max_step / primary_norm
                primary_norm = self.max_step
            remaining_step = max(0.0, self.max_step - primary_norm)
            lower_priority_norm = float(np.linalg.norm(lower_priority_delta))
            if lower_priority_norm > remaining_step:
                lower_priority_delta *= remaining_step / lower_priority_norm

            primary_arm = np.clip(
                current_arm + primary_delta,
                self._lower[self._arm_target_indices],
                self._upper[self._arm_target_indices],
            )
            primary_q = q.copy()
            primary_q[self._arm_target_indices] = primary_arm
            primary_objective = self._position_objective(data, primary_q, target_poses)
            accepted_arm = primary_arm
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
                candidate_arm = np.clip(
                    primary_arm + scale * lower_priority_delta,
                    self._lower[self._arm_target_indices],
                    self._upper[self._arm_target_indices],
                )
                candidate_q = q.copy()
                candidate_q[self._arm_target_indices] = candidate_arm
                if self._position_objective(data, candidate_q, target_poses) <= (
                    primary_objective + 2.0 * self.position_priority_slack**2 + 1.0e-12
                ):
                    accepted_arm = candidate_arm
                    break
            q[self._arm_target_indices] = accepted_arm
        return q

    def retarget(self, trajectory: Trajectory) -> Trajectory:
        coordinate_frame = (trajectory.infos or {}).get("coordinate_frame")
        if coordinate_frame != "robot":
            raise ValueError(
                "Dual-hand wrist poses must be in the MuJoCo robot frame; "
                f"got coordinate_frame={coordinate_frame!r}."
            )
        observations = trajectory.observations
        left_pose = self._pose_array(observations, self.left_wrist_key)
        right_pose = self._pose_array(observations, self.right_wrist_key)
        if len(left_pose) != len(right_pose):
            raise ValueError("Left and right wrist targets must have the same frames.")
        left_fingers = self._finger_array(
            observations,
            self.left_finger_key,
            frame_count=len(left_pose),
            width=len(self.left_finger_joint_names),
        )
        right_fingers = self._finger_array(
            observations,
            self.right_finger_key,
            frame_count=len(left_pose),
            width=len(self.right_finger_joint_names),
        )

        data = mujoco.MjData(self.model)
        qpos = np.empty(
            (len(left_pose), len(self.target_joint_names)), dtype=np.float64
        )
        seed = self._initial_qpos.copy()
        for frame_index in range(len(left_pose)):
            seed[self._left_finger_target_indices] = np.clip(
                left_fingers[frame_index],
                self._lower[self._left_finger_target_indices],
                self._upper[self._left_finger_target_indices],
            )
            seed[self._right_finger_target_indices] = np.clip(
                right_fingers[frame_index],
                self._lower[self._right_finger_target_indices],
                self._upper[self._right_finger_target_indices],
            )
            previous_arm_qpos = seed[self._arm_target_indices].copy()
            seed = self._solve_frame(
                data,
                (left_pose[frame_index], right_pose[frame_index]),
                seed,
                previous_arm_qpos,
            )
            if self.restart_position_error_m is not None:
                poses = (left_pose[frame_index], right_pose[frame_index])
                objective = self._position_objective(data, seed, poses)
                if objective > self.restart_position_error_m**2:
                    restart = seed.copy()
                    restart[self._arm_target_indices] = self._initial_qpos[self._arm_target_indices]
                    candidate = self._solve_frame(data, poses, restart, previous_arm_qpos)
                    if self._position_objective(data, candidate, poses) < objective:
                        seed = candidate
            qpos[frame_index] = seed
        result = _retargeted_trajectory(
            trajectory,
            qpos,
            target_key="qpos",
            target_joint_names=self.target_joint_names,
            method="mujoco_dual_wrist_se3",
        )
        result_infos = result.infos
        if result_infos is None:
            raise RuntimeError("Retargeted trajectory metadata is missing.")
        result_infos["retarget"].update(
            {
                "left_wrist_frame_name": self.left_wrist_site,
                "right_wrist_frame_name": self.right_wrist_site,
                "previous_posture_weight": self.previous_posture_weight,
                "neutral_posture_weight": self.neutral_posture_weight,
                "position_priority": self.position_priority,
                "position_priority_slack": self.position_priority_slack,
                "restart_position_error_m": self.restart_position_error_m,
            }
        )
        return result

    @staticmethod
    def _pose_array(observations: Mapping[str, np.ndarray], key: str) -> np.ndarray:
        if key not in observations:
            raise KeyError(f"trajectory observations must contain {key!r}.")
        poses = np.asarray(observations[key], dtype=np.float64)
        if poses.ndim != 2 or poses.shape[-1] != 7 or len(poses) < 2:
            raise ValueError(f"{key!r} must have shape [frames, 7] with two frames.")
        if not np.isfinite(poses).all():
            raise ValueError(f"{key!r} contains non-finite values.")
        result = poses.copy()
        result[:, 3:7] = _unit_wxyz(result[:, 3:7], name=key)
        return result

    @staticmethod
    def _finger_array(
        observations: Mapping[str, np.ndarray],
        key: str,
        *,
        frame_count: int,
        width: int,
    ) -> np.ndarray:
        if key not in observations:
            raise KeyError(f"trajectory observations must contain {key!r}.")
        values = np.asarray(observations[key], dtype=np.float64)
        if values.shape != (frame_count, width):
            raise ValueError(
                f"{key!r} must have shape {(frame_count, width)}, got {values.shape}."
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{key!r} contains non-finite values.")
        return values


def dexterous_reference_from_trajectory(
    trajectory: Trajectory,
    *,
    sequence_id: str,
    robot_name: str,
    fps: float,
    fixed_root_pose_w: np.ndarray,
    object_names: Sequence[str],
    object_poses_w: np.ndarray | None = None,
    object_pose_key: str = "object_poses_w",
    object_twists_w: np.ndarray | None = None,
    object_twist_key: str = "object_twists_w",
    object_asset_paths: Sequence[str] = (),
    object_asset_sha256: Sequence[str] = (),
    object_scales: np.ndarray | None = None,
    object_radii: np.ndarray | None = None,
    left_hand_frame_names: Sequence[str] = (),
    left_hand_frame_poses_w: np.ndarray | None = None,
    right_hand_frame_names: Sequence[str] = (),
    right_hand_frame_poses_w: np.ndarray | None = None,
    support_surface_names: Sequence[str] = (),
    support_surface_asset_paths: Sequence[str] = (),
    support_surface_asset_sha256: Sequence[str] = (),
    collision_asset_dependencies: Sequence[CollisionAssetDependency] = (),
    support_surface_scales: np.ndarray | None = None,
    support_surface_poses_w: np.ndarray | None = None,
    scene_physics: ScenePhysics | None = None,
    contacts: ContactSequence | None = None,
    training_qualification: TrainingQualification | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DexterousReference:
    """Convert a retarget result to the validated Isaac runtime contract.

    Object twists are world-frame linear XYZ followed by angular XYZ.  They
    may be supplied directly, read from ``object_twist_key``, or omitted; the
    resulting :class:`DexterousReference` deterministically derives omitted
    twists from its world-frame pose samples at ``fps``.
    """

    retarget_info = (trajectory.infos or {}).get("retarget", {})
    joint_names = tuple(
        str(name) for name in retarget_info.get("target_joint_names", ())
    )
    if not joint_names:
        raise ValueError(
            "Trajectory retarget metadata must contain target_joint_names."
        )
    left_wrist_frame_name = str(retarget_info.get("left_wrist_frame_name", ""))
    right_wrist_frame_name = str(retarget_info.get("right_wrist_frame_name", ""))
    if not left_wrist_frame_name or not right_wrist_frame_name:
        raise ValueError(
            "Trajectory retarget metadata must contain left and right wrist "
            "frame names."
        )
    if "qpos" not in trajectory.observations:
        raise KeyError("Retargeted trajectory must contain qpos.")
    if "left_wrist_pose_w" not in trajectory.observations:
        raise KeyError("Retargeted trajectory must contain left_wrist_pose_w.")
    if "right_wrist_pose_w" not in trajectory.observations:
        raise KeyError("Retargeted trajectory must contain right_wrist_pose_w.")
    if trajectory.dt is not None and not np.isclose(
        trajectory.dt, 1.0 / float(fps), rtol=0.0, atol=1.0e-8
    ):
        raise ValueError("Trajectory dt does not match the requested fps.")
    if object_poses_w is None:
        if object_pose_key not in trajectory.observations:
            raise KeyError(
                f"Trajectory must contain {object_pose_key!r} or receive object_poses_w."
            )
        object_poses_w = trajectory.observations[object_pose_key]
    object_twist_source = "argument"
    if object_twists_w is None:
        if object_twist_key in trajectory.observations:
            object_twists_w = trajectory.observations[object_twist_key]
            object_twist_source = f"trajectory:{object_twist_key}"
        else:
            object_twist_source = "derived_from_world_poses"
    coordinate_frame = (trajectory.infos or {}).get("coordinate_frame")
    if coordinate_frame not in {"robot", "world"}:
        raise ValueError("Trajectory coordinate_frame must be robot or world.")
    left_wrist_poses = np.asarray(trajectory.observations["left_wrist_pose_w"])
    right_wrist_poses = np.asarray(trajectory.observations["right_wrist_pose_w"])
    object_poses = np.asarray(object_poses_w)
    object_twists = None if object_twists_w is None else np.asarray(object_twists_w)
    left_hand_poses = left_hand_frame_poses_w
    right_hand_poses = right_hand_frame_poses_w
    surface_poses = support_surface_poses_w
    output_contacts = contacts
    if coordinate_frame == "robot":
        left_wrist_poses = _poses_from_robot_to_world(
            left_wrist_poses, fixed_root_pose_w
        )
        right_wrist_poses = _poses_from_robot_to_world(
            right_wrist_poses, fixed_root_pose_w
        )
        object_poses = _poses_from_robot_to_world(object_poses, fixed_root_pose_w)
        if object_twists is not None:
            object_twists = _twists_from_robot_to_world(
                object_twists, fixed_root_pose_w
            )
        if left_hand_poses is not None:
            left_hand_poses = _poses_from_robot_to_world(
                left_hand_poses, fixed_root_pose_w
            )
        if right_hand_poses is not None:
            right_hand_poses = _poses_from_robot_to_world(
                right_hand_poses, fixed_root_pose_w
            )
        if surface_poses is not None:
            surface_poses = _poses_from_robot_to_world(surface_poses, fixed_root_pose_w)
        if output_contacts is not None:
            output_contacts = _contacts_from_robot_to_world(
                output_contacts, fixed_root_pose_w
            )
    reference_metadata = dict(metadata or {})
    reference_metadata.setdefault("source_coordinate_frame", coordinate_frame)
    reference_metadata["object_twist_frame"] = "world"
    reference_metadata["object_twist_order"] = [
        "linear_x",
        "linear_y",
        "linear_z",
        "angular_x",
        "angular_y",
        "angular_z",
    ]
    reference_metadata["object_twist_source"] = object_twist_source
    if isinstance(retarget_info, dict):
        reference_metadata.setdefault("retarget", dict(retarget_info))
    return DexterousReference(
        sequence_id=sequence_id,
        robot_name=robot_name,
        fps=fps,
        joint_names=joint_names,
        qpos=np.asarray(trajectory.observations["qpos"]),
        qvel=(
            None
            if "qvel" not in trajectory.observations
            else np.asarray(trajectory.observations["qvel"])
        ),
        fixed_root_pose_w=fixed_root_pose_w,
        left_wrist_pose_w=left_wrist_poses,
        right_wrist_pose_w=right_wrist_poses,
        left_wrist_frame_name=left_wrist_frame_name,
        right_wrist_frame_name=right_wrist_frame_name,
        object_names=tuple(object_names),
        object_poses_w=object_poses,
        object_twists_w=object_twists,
        object_asset_paths=tuple(object_asset_paths),
        object_asset_sha256=tuple(object_asset_sha256),
        object_scales=object_scales,
        object_radii=object_radii,
        left_hand_frame_names=tuple(left_hand_frame_names),
        left_hand_frame_poses_w=left_hand_poses,
        right_hand_frame_names=tuple(right_hand_frame_names),
        right_hand_frame_poses_w=right_hand_poses,
        support_surface_names=tuple(support_surface_names),
        support_surface_asset_paths=tuple(support_surface_asset_paths),
        support_surface_asset_sha256=tuple(support_surface_asset_sha256),
        collision_asset_dependencies=tuple(collision_asset_dependencies),
        support_surface_scales=support_surface_scales,
        support_surface_poses_w=surface_poses,
        scene_physics=scene_physics,
        contacts=output_contacts,
        training_qualification=training_qualification,
        metadata=reference_metadata,
    )


__all__ = ["MujocoDualHandRetargeter", "dexterous_reference_from_trajectory"]
