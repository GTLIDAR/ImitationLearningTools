"""Fail-closed self-collision projection for dexterous finger trajectories."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np


@dataclass(frozen=True)
class SelfCollisionProjectionConfig:
    """Settings for the maximal-safe-closure search."""

    penetration_tolerance_m: float = 0.001
    search_iterations: int = 20

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.penetration_tolerance_m)
            or self.penetration_tolerance_m < 0.0
        ):
            raise ValueError("penetration_tolerance_m must be finite and non-negative.")
        if int(self.search_iterations) < 1:
            raise ValueError("search_iterations must be positive.")


@dataclass
class SelfCollisionProjectionReport:
    """Self-collision counts and retained closure evidence."""

    frame_count: int = 0
    violating_frames_before: int = 0
    violating_frames_after: int = 0
    open_baseline_violating_frames: list[int] = field(default_factory=list)
    residual_frames: list[int] = field(default_factory=list)
    mean_closure_scale: float = 1.0
    minimum_closure_scale: float = 1.0
    mean_joint_shift_rad: float = 0.0
    max_joint_shift_rad: float = 0.0

    @property
    def qualified(self) -> bool:
        return (
            self.violating_frames_after == 0 and not self.open_baseline_violating_frames
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "qualified": self.qualified,
            "frame_count": self.frame_count,
            "violating_frames_before": self.violating_frames_before,
            "violating_frames_after": self.violating_frames_after,
            "open_baseline_violating_frames": list(self.open_baseline_violating_frames),
            "residual_frames": list(self.residual_frames),
            "mean_closure_scale": self.mean_closure_scale,
            "minimum_closure_scale": self.minimum_closure_scale,
            "mean_joint_shift_rad": self.mean_joint_shift_rad,
            "max_joint_shift_rad": self.max_joint_shift_rad,
        }


def _named_id(model: mujoco.MjModel, kind: Any, name: str) -> int:
    result = int(mujoco.mj_name2id(model, kind, str(name)))
    if result < 0:
        raise ValueError(f"MuJoCo object {name!r} does not exist.")
    return result


class MujocoSelfCollisionClosureProjector:
    """Keep the largest collision-safe interpolation from open to target."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        trajectory_joint_names: Sequence[str],
        closure_joint_names: Sequence[str],
        robot_geom_names: Sequence[str | int],
        open_joint_positions: Sequence[float] | None = None,
        config: SelfCollisionProjectionConfig | None = None,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.config = config or SelfCollisionProjectionConfig()
        self.trajectory_joint_names = tuple(
            str(name) for name in trajectory_joint_names
        )
        self.closure_joint_names = tuple(str(name) for name in closure_joint_names)
        if not self.trajectory_joint_names or len(
            set(self.trajectory_joint_names)
        ) != len(self.trajectory_joint_names):
            raise ValueError("trajectory_joint_names must be non-empty and unique.")
        if not self.closure_joint_names or len(set(self.closure_joint_names)) != len(
            self.closure_joint_names
        ):
            raise ValueError("closure_joint_names must be non-empty and unique.")
        columns = {
            name: index for index, name in enumerate(self.trajectory_joint_names)
        }
        missing = set(self.closure_joint_names) - set(columns)
        if missing:
            raise ValueError(f"Closure joints are absent from trajectory: {missing}.")
        self.closure_columns = np.asarray(
            [columns[name] for name in self.closure_joint_names], dtype=np.int32
        )
        joint_ids = np.asarray(
            [
                _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.trajectory_joint_names
            ],
            dtype=np.int32,
        )
        self.qpos_addresses = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int32)
        closure_joint_ids = joint_ids[self.closure_columns]
        lower = np.asarray(
            [model.jnt_range[joint_id, 0] for joint_id in closure_joint_ids],
            dtype=np.float64,
        )
        upper = np.asarray(
            [model.jnt_range[joint_id, 1] for joint_id in closure_joint_ids],
            dtype=np.float64,
        )
        if open_joint_positions is None:
            self.open_positions = np.clip(np.zeros(len(lower)), lower, upper)
        else:
            open_values = np.asarray(open_joint_positions, dtype=np.float64)
            if open_values.shape != lower.shape or not np.isfinite(open_values).all():
                raise ValueError("open_joint_positions must align with closure joints.")
            self.open_positions = np.clip(open_values, lower, upper)
        geom_ids: list[int] = []
        for identifier in robot_geom_names:
            if isinstance(identifier, (int, np.integer)):
                geom_id = int(identifier)
                if not 0 <= geom_id < model.ngeom:
                    raise ValueError(f"MuJoCo geom id {geom_id} does not exist.")
            else:
                geom_id = _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, str(identifier))
            geom_ids.append(geom_id)
        if not geom_ids:
            raise ValueError("robot_geom_names must be non-empty.")
        self.robot_geom_ids = frozenset(geom_ids)

    def _violates(self, row: np.ndarray) -> bool:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_addresses] = row
        mujoco.mj_forward(self.model, self.data)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if (
                int(contact.geom1) in self.robot_geom_ids
                and int(contact.geom2) in self.robot_geom_ids
                and float(contact.dist) < -self.config.penetration_tolerance_m - 1.0e-9
            ):
                return True
        return False

    def project(
        self, qpos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, SelfCollisionProjectionReport]:
        """Project each frame and return qpos, closure scales, and report."""

        values = np.asarray(qpos, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.trajectory_joint_names):
            raise ValueError("qpos must have shape [T, trajectory joints].")
        if not np.isfinite(values).all():
            raise ValueError("qpos contains non-finite values.")
        output = values.copy()
        scales = np.ones(len(output), dtype=np.float64)
        report = SelfCollisionProjectionReport(frame_count=len(output))
        shifts: list[float] = []
        for frame, row in enumerate(output):
            target = row.copy()
            if not self._violates(target):
                continue
            report.violating_frames_before += 1
            open_row = target.copy()
            open_row[self.closure_columns] = self.open_positions
            if self._violates(open_row):
                report.open_baseline_violating_frames.append(frame)
                report.residual_frames.append(frame)
                report.violating_frames_after += 1
                scales[frame] = 0.0
                row[:] = open_row
                continue
            low = 0.0
            high = 1.0
            for _ in range(self.config.search_iterations):
                middle = 0.5 * (low + high)
                candidate = open_row.copy()
                candidate[self.closure_columns] = self.open_positions + middle * (
                    target[self.closure_columns] - self.open_positions
                )
                if self._violates(candidate):
                    high = middle
                else:
                    low = middle
            row[:] = open_row
            row[self.closure_columns] = self.open_positions + low * (
                target[self.closure_columns] - self.open_positions
            )
            scales[frame] = low
            if self._violates(row):
                report.violating_frames_after += 1
                report.residual_frames.append(frame)
            shifts.extend(np.abs(row - target))
        report.mean_closure_scale = float(np.mean(scales)) if len(scales) else 1.0
        report.minimum_closure_scale = float(np.min(scales)) if len(scales) else 1.0
        if shifts:
            report.mean_joint_shift_rad = float(np.mean(shifts))
            report.max_joint_shift_rad = float(np.max(shifts))
        return output, scales, report


__all__ = [
    "MujocoSelfCollisionClosureProjector",
    "SelfCollisionProjectionConfig",
    "SelfCollisionProjectionReport",
]
