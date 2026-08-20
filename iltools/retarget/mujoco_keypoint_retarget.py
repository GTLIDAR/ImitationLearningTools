"""Model-based keypoint retargeting with damped-least-squares MuJoCo IK."""

from __future__ import annotations

from dataclasses import dataclass
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


__all__ = ["MujocoKeypointRetargeter", "MujocoPositionTaskSpec"]
