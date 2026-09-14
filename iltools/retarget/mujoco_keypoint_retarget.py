"""Model-based keypoint retargeting with damped-least-squares MuJoCo IK."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import mujoco as _mujoco
import numpy as np

from iltools.core.trajectory import Trajectory

from .keypoint_retarget import _retargeted_trajectory

mujoco: Any = _mujoco


@dataclass(frozen=True, slots=True)
class MujocoPositionTaskSpec:
    """Map one source keypoint to one MuJoCo site position."""

    source_keypoint: str
    target_site: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.source_keypoint or not self.target_site:
            raise ValueError("MuJoCo position task names must be non-empty.")
        if not np.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("MuJoCo position task weight must be positive.")


@dataclass(frozen=True)
class MujocoKeypointProjectionConfig:
    """Numerical settings for projecting sites onto trajectory keypoints."""

    iterations: int = 100
    damping: float = 0.03
    regularization: float = 0.001
    max_step: float = 0.08
    tolerance_m: float = 0.001
    line_search_steps: int = 8

    def __post_init__(self) -> None:
        if int(self.iterations) < 1 or int(self.line_search_steps) < 1:
            raise ValueError("iterations and line_search_steps must be positive.")
        for name in (
            "damping",
            "regularization",
            "max_step",
            "tolerance_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")


@dataclass
class MujocoKeypointProjectionReport:
    """Residual and correction evidence for trajectory keypoint projection."""

    frame_count: int = 0
    active_targets: int = 0
    converged_frames: int = 0
    initial_errors_m: list[float] = field(default_factory=list)
    final_errors_m: list[float] = field(default_factory=list)
    joint_shifts: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "active_targets": self.active_targets,
            "converged_frames": self.converged_frames,
            "mean_initial_error_m": (
                float(np.mean(self.initial_errors_m)) if self.initial_errors_m else 0.0
            ),
            "mean_final_error_m": (
                float(np.mean(self.final_errors_m)) if self.final_errors_m else 0.0
            ),
            "p95_final_error_m": (
                float(np.percentile(self.final_errors_m, 95.0))
                if self.final_errors_m
                else 0.0
            ),
            "max_final_error_m": (
                float(np.max(self.final_errors_m)) if self.final_errors_m else 0.0
            ),
            "mean_joint_shift": (
                float(np.mean(self.joint_shifts)) if self.joint_shifts else 0.0
            ),
            "max_joint_shift": (
                float(np.max(self.joint_shifts)) if self.joint_shifts else 0.0
            ),
        }


class MujocoKeypointRetargeter:
    """Retarget source keypoints to bounded robot joints with MuJoCo IK.

    The source keypoints and target MuJoCo sites must use the same coordinate
    frame. The solver updates only single-DOF hinge or slide joints listed in
    ``target_joint_names``. It uses the previous frame as the next seed, which
    keeps the result temporally continuous for egocentric motion clips.
    """

    def __init__(
        self,
        model: mujoco.MjModel | str | Path,
        target_joint_names: Sequence[str],
        tasks: Sequence[MujocoPositionTaskSpec],
        *,
        source_key: str = "keypoints",
        target_key: str = "qpos",
        iterations: int = 100,
        damping: float = 0.02,
        max_step: float = 0.12,
        tolerance: float = 1.0e-4,
        initial_qpos: Sequence[float] | None = None,
    ) -> None:
        if isinstance(model, (str, Path)):
            model_path = Path(model).expanduser().resolve()
            if not model_path.is_file():
                raise FileNotFoundError(f"MuJoCo model not found: {model_path}")
            self.model = mujoco.MjModel.from_xml_path(str(model_path))
        else:
            self.model = model

        self.target_joint_names = tuple(target_joint_names)
        self.tasks = tuple(tasks)
        self.source_key = source_key
        self.target_key = target_key
        if not self.target_joint_names or len(set(self.target_joint_names)) != len(
            self.target_joint_names
        ):
            raise ValueError("target_joint_names must be non-empty and unique.")
        if not self.tasks:
            raise ValueError("At least one MuJoCo position task is required.")
        if iterations < 1:
            raise ValueError("iterations must be positive.")
        if not np.isfinite((damping, max_step, tolerance)).all() or damping <= 0.0:
            raise ValueError(
                "damping, max_step, and tolerance must be finite and positive."
            )
        self.iterations = int(iterations)
        self.damping = float(damping)
        self.max_step = float(max_step)
        self.tolerance = float(tolerance)

        self._joint_ids = np.asarray(
            [self.model.joint(name).id for name in self.target_joint_names],
            dtype=np.int32,
        )
        unsupported = [
            name
            for name, joint_id in zip(
                self.target_joint_names, self._joint_ids, strict=True
            )
            if int(self.model.jnt_type[joint_id])
            not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            )
        ]
        if unsupported:
            raise ValueError(
                "MuJoCo keypoint retargeting supports hinge/slide joints only: "
                f"{unsupported}"
            )
        self._qpos_indices = np.asarray(
            self.model.jnt_qposadr[self._joint_ids], dtype=np.int32
        )
        self._dof_indices = np.asarray(
            self.model.jnt_dofadr[self._joint_ids], dtype=np.int32
        )
        self._site_ids = tuple(
            self.model.site(task.target_site).id for task in self.tasks
        )
        self._task_weights = np.repeat(
            np.sqrt(np.asarray([task.weight for task in self.tasks])), 3
        )

        data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, data)
        self._base_qpos = data.qpos.copy()
        if initial_qpos is None:
            self._initial_qpos = self._base_qpos[self._qpos_indices].copy()
        else:
            self._initial_qpos = np.asarray(initial_qpos, dtype=np.float64)
            if self._initial_qpos.shape != (len(self.target_joint_names),):
                raise ValueError(
                    "initial_qpos must match target_joint_names: "
                    f"expected {(len(self.target_joint_names),)}, "
                    f"got {self._initial_qpos.shape}."
                )
            if not np.isfinite(self._initial_qpos).all():
                raise ValueError("initial_qpos contains non-finite values.")
        self._lower, self._upper = self._joint_limits()

    def _joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(len(self._joint_ids), -np.inf, dtype=np.float64)
        upper = np.full(len(self._joint_ids), np.inf, dtype=np.float64)
        for index, joint_id in enumerate(self._joint_ids):
            if self.model.jnt_limited[joint_id]:
                lower[index], upper[index] = self.model.jnt_range[joint_id]
        return lower, upper

    def _solve_frame(
        self,
        data: mujoco.MjData,
        source_points: dict[str, np.ndarray],
        seed: np.ndarray,
    ) -> np.ndarray:
        q = np.array(seed, dtype=np.float64, copy=True)
        for _ in range(self.iterations):
            data.qpos[:] = self._base_qpos
            data.qpos[self._qpos_indices] = q
            mujoco.mj_forward(self.model, data)
            errors: list[np.ndarray] = []
            jacobians: list[np.ndarray] = []
            for task, site_id in zip(self.tasks, self._site_ids, strict=True):
                target = source_points[task.source_keypoint]
                errors.append(target - data.site_xpos[site_id])
                jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
                jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
                mujoco.mj_jacSite(
                    self.model,
                    data,
                    jacobian_position,
                    jacobian_rotation,
                    site_id,
                )
                jacobians.append(jacobian_position[:, self._dof_indices])
            error = np.concatenate(errors) * self._task_weights
            if np.linalg.norm(error) <= self.tolerance:
                break
            jacobian = np.vstack(jacobians) * self._task_weights[:, None]
            normal = jacobian.T @ jacobian
            normal += self.damping**2 * np.eye(len(self._dof_indices))
            delta = np.linalg.solve(normal, jacobian.T @ error)
            norm = float(np.linalg.norm(delta))
            if norm > self.max_step:
                delta *= self.max_step / norm
            q = np.clip(q + delta, self._lower, self._upper)
        return q

    def retarget(self, trajectory: Trajectory) -> Trajectory:
        coordinate_frame = (trajectory.infos or {}).get("coordinate_frame")
        if coordinate_frame != "robot":
            raise ValueError(
                "MuJoCo keypoint retargeting requires keypoints in the robot "
                "model frame. Apply transform_keypoint_trajectory first; "
                f"got coordinate_frame={coordinate_frame!r}."
            )
        if self.source_key not in trajectory.observations:
            raise KeyError(f"trajectory observations must contain {self.source_key!r}.")
        points = np.asarray(trajectory.observations[self.source_key], dtype=np.float64)
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(
                f"{self.source_key!r} must have shape [frames, keypoints, 3], "
                f"got {points.shape}."
            )
        keypoint_names = tuple(
            (trajectory.infos or {}).get(
                "keypoint_names", [f"keypoint_{i}" for i in range(points.shape[1])]
            )
        )
        if len(keypoint_names) != points.shape[1] or len(set(keypoint_names)) != len(
            keypoint_names
        ):
            raise ValueError("trajectory keypoint_names must match the keypoint axis.")
        point_index = {name: index for index, name in enumerate(keypoint_names)}
        for task in self.tasks:
            if task.source_keypoint not in point_index:
                raise KeyError(f"Unknown source keypoint: {task.source_keypoint!r}.")
        if not np.isfinite(points).all():
            raise ValueError(f"{self.source_key!r} contains non-finite values.")

        data = mujoco.MjData(self.model)
        qpos = np.empty((len(points), len(self.target_joint_names)), dtype=np.float64)
        seed = self._initial_qpos
        for frame_index, frame in enumerate(points):
            named_points = {name: frame[index] for name, index in point_index.items()}
            seed = self._solve_frame(data, named_points, seed)
            qpos[frame_index] = seed
        return _retargeted_trajectory(
            trajectory,
            qpos,
            target_key=self.target_key,
            target_joint_names=self.target_joint_names,
            method="mujoco_dls_keypoint",
        )


class MujocoKeypointProjector:
    """Fit named robot sites while preserving a full trajectory seed."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        trajectory_joint_names: Sequence[str],
        variable_joint_names: Sequence[str],
        target_site_names: Sequence[str],
        task_weights: Sequence[float] | None = None,
        config: MujocoKeypointProjectionConfig | None = None,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.config = config or MujocoKeypointProjectionConfig()
        self.trajectory_joint_names = tuple(
            str(name) for name in trajectory_joint_names
        )
        self.variable_joint_names = tuple(str(name) for name in variable_joint_names)
        self.target_site_names = tuple(str(name) for name in target_site_names)
        if not self.trajectory_joint_names or len(
            set(self.trajectory_joint_names)
        ) != len(self.trajectory_joint_names):
            raise ValueError("trajectory_joint_names must be non-empty and unique.")
        if not self.variable_joint_names or len(set(self.variable_joint_names)) != len(
            self.variable_joint_names
        ):
            raise ValueError("variable_joint_names must be non-empty and unique.")
        if not self.target_site_names or len(set(self.target_site_names)) != len(
            self.target_site_names
        ):
            raise ValueError("target_site_names must be non-empty and unique.")
        columns = {
            name: index for index, name in enumerate(self.trajectory_joint_names)
        }
        missing = set(self.variable_joint_names) - set(columns)
        if missing:
            raise ValueError(f"Variable joints are absent from trajectory: {missing}.")
        self.joint_ids = np.asarray(
            [self.model.joint(name).id for name in self.trajectory_joint_names],
            dtype=np.int32,
        )
        self.qpos_addresses = np.asarray(
            self.model.jnt_qposadr[self.joint_ids], dtype=np.int32
        )
        self.variable_columns = np.asarray(
            [columns[name] for name in self.variable_joint_names], dtype=np.int32
        )
        variable_joint_ids = self.joint_ids[self.variable_columns]
        unsupported = [
            name
            for name, joint_id in zip(
                self.variable_joint_names, variable_joint_ids, strict=True
            )
            if int(self.model.jnt_type[joint_id])
            not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            )
        ]
        if unsupported:
            raise ValueError(
                "MuJoCo keypoint projection supports hinge/slide joints only: "
                f"{unsupported}"
            )
        self.variable_dofs = np.asarray(
            self.model.jnt_dofadr[variable_joint_ids], dtype=np.int32
        )
        limited = np.asarray(self.model.jnt_limited[variable_joint_ids], dtype=bool)
        self.lower = np.where(
            limited,
            self.model.jnt_range[variable_joint_ids, 0],
            -np.inf,
        ).astype(np.float64)
        self.upper = np.where(
            limited,
            self.model.jnt_range[variable_joint_ids, 1],
            np.inf,
        ).astype(np.float64)
        self.site_ids = tuple(
            self.model.site(name).id for name in self.target_site_names
        )
        if task_weights is None:
            self.task_weights = np.ones(len(self.site_ids), dtype=np.float64)
        else:
            self.task_weights = np.asarray(task_weights, dtype=np.float64)
            if (
                self.task_weights.shape != (len(self.site_ids),)
                or np.any(~np.isfinite(self.task_weights))
                or np.any(self.task_weights <= 0.0)
            ):
                raise ValueError(
                    "task_weights must be finite, positive, and align with sites."
                )

    def _set_q(self, row: np.ndarray) -> None:
        self.data.qpos[self.qpos_addresses] = row
        mujoco.mj_forward(self.model, self.data)

    def _errors(self, targets: np.ndarray, active: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                targets[index] - self.data.site_xpos[site_id]
                for index, site_id in enumerate(self.site_ids)
                if active[index]
            ],
            dtype=np.float64,
        )

    def project(
        self,
        qpos: np.ndarray,
        *,
        target_positions: np.ndarray,
        active: np.ndarray | None = None,
    ) -> tuple[np.ndarray, MujocoKeypointProjectionReport]:
        """Project active sites and return a temporally seeded report."""

        values = np.asarray(qpos, dtype=np.float64)
        targets = np.asarray(target_positions, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.trajectory_joint_names):
            raise ValueError("qpos must have shape [T, trajectory joints].")
        if targets.shape != (len(values), len(self.site_ids), 3):
            raise ValueError("target_positions must have shape [T, sites, 3].")
        valid = (
            np.ones((len(values), len(self.site_ids)), dtype=bool)
            if active is None
            else np.asarray(active, dtype=bool)
        )
        if valid.shape != targets.shape[:2]:
            raise ValueError("active must have shape [T, sites].")
        if not np.isfinite(values).all() or not np.isfinite(targets).all():
            raise ValueError("qpos/target_positions contain non-finite values.")

        output = values.copy()
        report = MujocoKeypointProjectionReport(frame_count=len(output))
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        for frame, row in enumerate(output):
            if not np.any(valid[frame]):
                continue
            seed = row.copy()
            self._set_q(row)
            initial = self._errors(targets[frame], valid[frame])
            report.initial_errors_m.extend(np.linalg.norm(initial, axis=1))
            report.active_targets += len(initial)
            for _ in range(self.config.iterations):
                self._set_q(row)
                raw_errors = self._errors(targets[frame], valid[frame])
                if (
                    float(np.max(np.linalg.norm(raw_errors, axis=1)))
                    <= self.config.tolerance_m
                ):
                    break
                errors: list[np.ndarray] = []
                jacobians: list[np.ndarray] = []
                for index, site_id in enumerate(self.site_ids):
                    if not valid[frame, index]:
                        continue
                    weight = float(np.sqrt(self.task_weights[index]))
                    errors.append(
                        weight * (targets[frame, index] - self.data.site_xpos[site_id])
                    )
                    jacp.fill(0.0)
                    jacr.fill(0.0)
                    mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
                    jacobians.append(weight * jacp[:, self.variable_dofs])
                error = np.concatenate(errors)
                jacobian = np.vstack(jacobians)
                normal = jacobian.T @ jacobian
                normal += (
                    self.config.damping**2 + self.config.regularization
                ) * np.eye(len(self.variable_columns))
                current = row[self.variable_columns].copy()
                right = jacobian.T @ error + self.config.regularization * (
                    seed[self.variable_columns] - current
                )
                delta = np.linalg.solve(normal, right)
                norm = float(np.linalg.norm(delta))
                if norm > self.config.max_step:
                    delta *= self.config.max_step / norm
                current_objective = float(
                    error @ error
                ) + self.config.regularization * float(
                    np.sum((current - seed[self.variable_columns]) ** 2)
                )
                accepted = False
                for search_step in range(self.config.line_search_steps):
                    row[self.variable_columns] = np.clip(
                        current + 0.5**search_step * delta,
                        self.lower,
                        self.upper,
                    )
                    self._set_q(row)
                    candidate = self._errors(targets[frame], valid[frame])
                    candidate_weights = np.sqrt(self.task_weights[valid[frame]])[
                        :, None
                    ]
                    candidate_objective = float(
                        np.sum((candidate_weights * candidate) ** 2)
                    ) + self.config.regularization * float(
                        np.sum(
                            (row[self.variable_columns] - seed[self.variable_columns])
                            ** 2
                        )
                    )
                    if candidate_objective < current_objective - 1.0e-16:
                        accepted = True
                        break
                if not accepted:
                    row[self.variable_columns] = current
                    break
            self._set_q(row)
            final = self._errors(targets[frame], valid[frame])
            final_norms = np.linalg.norm(final, axis=1)
            report.final_errors_m.extend(final_norms)
            if float(np.max(final_norms)) <= self.config.tolerance_m:
                report.converged_frames += 1
            report.joint_shifts.extend(
                np.abs(row[self.variable_columns] - seed[self.variable_columns])
            )
        return output, report


__all__ = [
    "MujocoKeypointProjectionConfig",
    "MujocoKeypointProjectionReport",
    "MujocoKeypointProjector",
    "MujocoKeypointRetargeter",
    "MujocoPositionTaskSpec",
]
