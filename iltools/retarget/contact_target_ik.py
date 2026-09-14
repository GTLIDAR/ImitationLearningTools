"""Contact-target inverse kinematics for dexterous retargeting.

CHORD-style source contacts identify an object-surface point for each hand
link.  This module turns those semantic targets into a feasible robot pose by
optimizing the corresponding robot collision points with MuJoCo Jacobians.
It adjusts only explicitly authorized arm/finger joints and keeps a quadratic
pull toward the original retarget, so contact constraints augment rather than
replace wrist/body tracking.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np


@dataclass(frozen=True)
class ContactTargetIKConfig:
    """Numerical settings for contact-target IK."""

    iterations: int = 60
    damping: float = 0.03
    regularization: float = 0.02
    max_step_rad: float = 0.08
    tolerance_m: float = 0.001
    query_cutoff_m: float = 0.30
    line_search_steps: int = 8

    def __post_init__(self) -> None:
        if self.iterations < 1 or self.line_search_steps < 1:
            raise ValueError("iterations and line_search_steps must be positive.")
        for name in (
            "damping",
            "regularization",
            "max_step_rad",
            "tolerance_m",
            "query_cutoff_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")


@dataclass
class ContactTargetIKReport:
    """Residual and motion evidence from contact-target IK."""

    frame_count: int = 0
    active_side_frames: int = 0
    active_targets: int = 0
    solved_targets: int = 0
    skipped_targets: int = 0
    converged_frames: int = 0
    mean_initial_error_m: float = 0.0
    mean_final_error_m: float = 0.0
    p95_final_error_m: float = 0.0
    max_final_error_m: float = 0.0
    mean_joint_shift_rad: float = 0.0
    max_joint_shift_rad: float = 0.0
    per_side_final_errors_m: dict[str, list[float]] = field(default_factory=dict)
    worst_final_target: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "active_side_frames": self.active_side_frames,
            "active_targets": self.active_targets,
            "solved_targets": self.solved_targets,
            "skipped_targets": self.skipped_targets,
            "converged_frames": self.converged_frames,
            "mean_initial_error_m": self.mean_initial_error_m,
            "mean_final_error_m": self.mean_final_error_m,
            "p95_final_error_m": self.p95_final_error_m,
            "max_final_error_m": self.max_final_error_m,
            "mean_joint_shift_rad": self.mean_joint_shift_rad,
            "max_joint_shift_rad": self.max_joint_shift_rad,
            "per_side_mean_final_error_m": {
                side: (float(np.mean(values)) if values else None)
                for side, values in self.per_side_final_errors_m.items()
            },
            "worst_final_target": self.worst_final_target,
        }


def _id(model: mujoco.MjModel, object_type: Any, name: str) -> int:
    result = int(mujoco.mj_name2id(model, object_type, str(name)))
    if result < 0:
        raise ValueError(f"MuJoCo model has no {object_type!s} named {name!r}.")
    return result


def _body_contact_geom(model: mujoco.MjModel, body_name: str) -> int:
    body_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    candidates = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == body_id
        and (
            int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
        )
    ]
    if not candidates:
        candidates = [
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) == body_id
        ]
    if not candidates:
        raise ValueError(f"Robot contact body {body_name!r} has no geom.")
    candidates.sort(
        key=lambda geom_id: (
            "tip_pad_proxy" not in (model.geom(geom_id).name or ""),
            -float(np.prod(model.geom_size[geom_id] + 1.0e-9)),
            geom_id,
        )
    )
    return int(candidates[0])


class MujocoContactTargetRetargeter:
    """Pull robot link collision witnesses onto CHORD object targets."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        trajectory_joint_names: Sequence[str],
        variable_joint_names: Mapping[str, Sequence[str]],
        contact_body_names: Mapping[str, Sequence[str]],
        object_geom_name: str,
        object_mocap_body_name: str,
        shared_joint_names: Sequence[str] = (),
        contact_geom_names: Mapping[str, Sequence[str | int]] | None = None,
        config: ContactTargetIKConfig | None = None,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.config = config or ContactTargetIKConfig()
        self.sides = tuple(str(side) for side in contact_body_names)
        if not self.sides or tuple(variable_joint_names) != self.sides:
            raise ValueError(
                "variable_joint_names and contact_body_names must share a non-empty side order."
            )
        self.trajectory_joint_names = tuple(
            str(name) for name in trajectory_joint_names
        )
        if len(set(self.trajectory_joint_names)) != len(self.trajectory_joint_names):
            raise ValueError("trajectory_joint_names must be unique.")
        self.column_by_joint = {
            name: index for index, name in enumerate(self.trajectory_joint_names)
        }
        self.qpos_addresses = np.asarray(
            [
                int(model.jnt_qposadr[_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
                for name in self.trajectory_joint_names
            ],
            dtype=np.int32,
        )
        self.dof_by_joint = {
            name: int(model.jnt_dofadr[_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
            for name in self.trajectory_joint_names
        }
        self.lower = np.asarray(
            [
                model.jnt_range[_id(model, mujoco.mjtObj.mjOBJ_JOINT, name), 0]
                for name in self.trajectory_joint_names
            ],
            dtype=np.float64,
        )
        self.upper = np.asarray(
            [
                model.jnt_range[_id(model, mujoco.mjtObj.mjOBJ_JOINT, name), 1]
                for name in self.trajectory_joint_names
            ],
            dtype=np.float64,
        )
        self.shared_joint_names = tuple(str(name) for name in shared_joint_names)
        self.variable_joint_names = {
            side: tuple(str(name) for name in variable_joint_names[side])
            for side in self.sides
        }
        for name in self.shared_joint_names + tuple(
            name for side in self.sides for name in self.variable_joint_names[side]
        ):
            if name not in self.column_by_joint:
                raise ValueError(
                    f"Variable joint {name!r} is absent from the trajectory."
                )

        self.contact_body_names = {
            side: tuple(str(name) for name in contact_body_names[side])
            for side in self.sides
        }
        self.contact_body_ids = {
            side: tuple(
                _id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in self.contact_body_names[side]
            )
            for side in self.sides
        }
        if contact_geom_names is None:
            self.contact_geom_ids = {
                side: tuple(
                    _body_contact_geom(model, name)
                    for name in self.contact_body_names[side]
                )
                for side in self.sides
            }
        else:
            if tuple(contact_geom_names) != self.sides:
                raise ValueError("contact_geom_names must use the same side order.")
            self.contact_geom_ids = {}
            for side in self.sides:
                identifiers = tuple(contact_geom_names[side])
                if len(identifiers) != len(self.contact_body_names[side]):
                    raise ValueError(f"{side} contact geom/body counts differ.")
                geom_ids: list[int] = []
                for identifier in identifiers:
                    if isinstance(identifier, (int, np.integer)):
                        geom_id = int(identifier)
                        if not 0 <= geom_id < model.ngeom:
                            raise ValueError(f"MuJoCo model has no geom id {geom_id}.")
                        geom_ids.append(geom_id)
                    else:
                        geom_ids.append(
                            _id(model, mujoco.mjtObj.mjOBJ_GEOM, str(identifier))
                        )
                self.contact_geom_ids[side] = tuple(geom_ids)
        self.object_geom_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, object_geom_name)
        object_body_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, object_mocap_body_name)
        self.object_mocap_id = int(model.body_mocapid[object_body_id])
        if self.object_mocap_id < 0:
            raise ValueError(
                f"Object body {object_mocap_body_name!r} is not mocap-driven."
            )

    def _variable_contract(
        self, active_sides: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        names: list[str] = list(self.shared_joint_names)
        for side in active_sides:
            names.extend(self.variable_joint_names[side])
        names = list(dict.fromkeys(names))
        columns = np.asarray(
            [self.column_by_joint[name] for name in names], dtype=np.int32
        )
        dofs = np.asarray([self.dof_by_joint[name] for name in names], dtype=np.int32)
        return columns, dofs

    def retarget(
        self,
        qpos: np.ndarray,
        *,
        object_poses_wxyz: np.ndarray,
        target_positions: np.ndarray,
        active: np.ndarray,
    ) -> tuple[np.ndarray, ContactTargetIKReport]:
        """Optimize a trajectory against dense per-link object targets.

        Args:
            qpos: Initial trajectory ``[T, J]`` aligned to
                ``trajectory_joint_names``.
            object_poses_wxyz: Mocap object poses ``[T, 7]`` as XYZ+WXYZ.
            target_positions: Object-surface targets ``[T, S, K, 3]``.
            active: Valid target mask ``[T, S, K]``.
        """

        values = np.asarray(qpos, dtype=np.float64)
        poses = np.asarray(object_poses_wxyz, dtype=np.float64)
        targets = np.asarray(target_positions, dtype=np.float64)
        valid = np.asarray(active, dtype=bool)
        if values.ndim != 2 or values.shape[1] != len(self.trajectory_joint_names):
            raise ValueError("qpos must have shape [T, trajectory joints].")
        frame_count = len(values)
        max_slots = max(len(self.contact_body_names[side]) for side in self.sides)
        if poses.shape != (frame_count, 7):
            raise ValueError("object_poses_wxyz must have shape [T, 7].")
        if targets.shape != (frame_count, len(self.sides), max_slots, 3):
            raise ValueError("target_positions has an invalid shape.")
        if valid.shape != targets.shape[:-1]:
            raise ValueError("active must align with target_positions.")
        if not np.isfinite(values).all() or not np.isfinite(poses).all():
            raise ValueError("qpos/object poses contain non-finite values.")
        if not np.isfinite(targets[valid]).all():
            raise ValueError("Active contact targets contain non-finite values.")

        output = values.copy()
        report = ContactTargetIKReport(frame_count=frame_count)
        report.per_side_final_errors_m = {side: [] for side in self.sides}
        initial_errors: list[float] = []
        final_errors: list[float] = []
        joint_shifts: list[float] = []
        fromto = np.zeros(6, dtype=np.float64)
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)

        for frame in range(frame_count):
            active_sides = [
                side
                for side_index, side in enumerate(self.sides)
                if np.any(valid[frame, side_index])
            ]
            if not active_sides:
                continue
            report.active_side_frames += len(active_sides)
            variable_columns, variable_dofs = self._variable_contract(active_sides)
            if len(variable_columns) == 0:
                continue
            seed = output[frame].copy()
            self.data.qpos[self.qpos_addresses] = output[frame]
            self.data.mocap_pos[self.object_mocap_id] = poses[frame, :3]
            self.data.mocap_quat[self.object_mocap_id] = poses[frame, 3:7]
            mujoco.mj_forward(self.model, self.data)

            frame_targets: list[tuple[str, int, int, int, np.ndarray, np.ndarray]] = []
            for side_index, side in enumerate(self.sides):
                for slot in np.flatnonzero(valid[frame, side_index]):
                    report.active_targets += 1
                    if slot >= len(self.contact_body_ids[side]):
                        report.skipped_targets += 1
                        continue
                    body_id = self.contact_body_ids[side][slot]
                    geom_id = self.contact_geom_ids[side][slot]
                    distance = float(
                        mujoco.mj_geomDistance(
                            self.model,
                            self.data,
                            geom_id,
                            self.object_geom_id,
                            self.config.query_cutoff_m,
                            fromto,
                        )
                    )
                    if not np.isfinite(distance):
                        report.skipped_targets += 1
                        continue
                    witness = fromto[:3].copy()
                    geom_center = self.data.geom_xpos[geom_id].copy()
                    body_position = self.data.xpos[body_id]
                    # Penetration witnesses are not stable semantic points:
                    # for non-convex mesh pairs MuJoCo may return an internal
                    # feature far outside the link.  Use the authored link
                    # geom centre in that case.  For separated pairs retain
                    # the surface witness, but reject impossible body-local
                    # offsets fail-closed.
                    maximum_local_radius = (
                        float(np.linalg.norm(geom_center - body_position))
                        + 2.0 * float(np.linalg.norm(self.model.geom_size[geom_id]))
                        + 0.02
                    )
                    if (
                        distance <= 0.0
                        or distance >= self.config.query_cutoff_m - 1.0e-9
                        or not np.isfinite(witness).all()
                        or float(np.linalg.norm(witness - body_position))
                        > maximum_local_radius
                    ):
                        witness = geom_center
                    body_rotation = self.data.xmat[body_id].reshape(3, 3)
                    local_point = body_rotation.T @ (witness - self.data.xpos[body_id])
                    frame_targets.append(
                        (
                            side,
                            side_index,
                            int(slot),
                            body_id,
                            local_point,
                            targets[frame, side_index, slot],
                        )
                    )
            if not frame_targets:
                continue
            report.solved_targets += len(frame_targets)

            def residuals_and_jacobian() -> tuple[np.ndarray, np.ndarray]:
                errors: list[np.ndarray] = []
                jacobians: list[np.ndarray] = []
                for _, _, _, body_id, local_point, target in frame_targets:
                    body_rotation = self.data.xmat[body_id].reshape(3, 3)
                    point = self.data.xpos[body_id] + body_rotation @ local_point
                    errors.append(target - point)
                    jacp.fill(0.0)
                    jacr.fill(0.0)
                    mujoco.mj_jac(
                        self.model,
                        self.data,
                        jacp,
                        jacr,
                        point,
                        body_id,
                    )
                    jacobians.append(jacp[:, variable_dofs].copy())
                return np.concatenate(errors), np.concatenate(jacobians, axis=0)

            initial, _ = residuals_and_jacobian()
            initial_errors.extend(np.linalg.norm(initial.reshape(-1, 3), axis=1))
            for _ in range(self.config.iterations):
                error, jacobian = residuals_and_jacobian()
                if (
                    float(np.max(np.linalg.norm(error.reshape(-1, 3), axis=1)))
                    <= self.config.tolerance_m
                ):
                    break
                normal = jacobian.T @ jacobian
                normal += (
                    self.config.damping**2 + self.config.regularization
                ) * np.eye(len(variable_columns))
                current = output[frame, variable_columns]
                right = jacobian.T @ error + self.config.regularization * (
                    seed[variable_columns] - current
                )
                delta = np.linalg.solve(normal, right)
                norm = float(np.linalg.norm(delta))
                if norm > self.config.max_step_rad:
                    delta *= self.config.max_step_rad / norm
                objective = float(error @ error) + self.config.regularization * float(
                    np.sum((current - seed[variable_columns]) ** 2)
                )
                accepted = False
                for search_step in range(self.config.line_search_steps):
                    scale = 0.5**search_step
                    candidate = np.clip(
                        current + scale * delta,
                        self.lower[variable_columns],
                        self.upper[variable_columns],
                    )
                    output[frame, variable_columns] = candidate
                    self.data.qpos[self.qpos_addresses] = output[frame]
                    mujoco.mj_forward(self.model, self.data)
                    candidate_error, _ = residuals_and_jacobian()
                    candidate_objective = float(candidate_error @ candidate_error)
                    candidate_objective += self.config.regularization * float(
                        np.sum((candidate - seed[variable_columns]) ** 2)
                    )
                    if candidate_objective < objective - 1.0e-12:
                        accepted = True
                        break
                if not accepted:
                    output[frame, variable_columns] = current
                    self.data.qpos[self.qpos_addresses] = output[frame]
                    mujoco.mj_forward(self.model, self.data)
                    break

            final, _ = residuals_and_jacobian()
            per_target = np.linalg.norm(final.reshape(-1, 3), axis=1)
            final_errors.extend(per_target)
            if float(np.max(per_target)) <= self.config.tolerance_m:
                report.converged_frames += 1
            for target_index, target_spec in enumerate(frame_targets):
                report.per_side_final_errors_m[target_spec[0]].append(
                    float(per_target[target_index])
                )
                error_value = float(per_target[target_index])
                if (
                    report.worst_final_target is None
                    or error_value > report.worst_final_target["error_m"]
                ):
                    report.worst_final_target = {
                        "frame": frame,
                        "side": target_spec[0],
                        "slot": int(target_spec[2]),
                        "error_m": error_value,
                    }
            joint_shifts.extend(np.abs(output[frame] - seed))

        if initial_errors:
            report.mean_initial_error_m = float(np.mean(initial_errors))
        if final_errors:
            report.mean_final_error_m = float(np.mean(final_errors))
            report.p95_final_error_m = float(np.percentile(final_errors, 95.0))
            report.max_final_error_m = float(np.max(final_errors))
        if joint_shifts:
            report.mean_joint_shift_rad = float(np.mean(joint_shifts))
            report.max_joint_shift_rad = float(np.max(joint_shifts))
        return output, report


__all__ = [
    "ContactTargetIKConfig",
    "ContactTargetIKReport",
    "MujocoContactTargetRetargeter",
]
