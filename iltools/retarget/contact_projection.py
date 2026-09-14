"""Project source-active semantic hand links into a measured contact band.

Source contact targets are often over-constrained after an embodiment change:
every human link cannot generally occupy its original object-surface point on
a differently proportioned robot hand.  This projector preserves the semantic
lower bound instead.  For each source-active hand frame it selects the closest
active mapped robot link and moves that side's authorized joints until the
link is in a shallow, non-penetrating contact band.  Forbidden-contact and
self-collision projectors remain separate constraints and can be alternated
with this task in a deterministic projection loop.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np


def _object_id(model: mujoco.MjModel, kind: Any, identifier: str | int) -> int:
    if isinstance(identifier, (int, np.integer)):
        result = int(identifier)
    else:
        result = int(mujoco.mj_name2id(model, kind, str(identifier)))
    if result < 0:
        raise ValueError(f"MuJoCo object {identifier!r} does not exist.")
    return result


@dataclass(frozen=True)
class ContactConstraintProjectionConfig:
    """Numerical settings for semantic contact-band projection."""

    target_distance_m: float = 0.0005
    contact_tolerance_m: float = 0.002
    max_penetration_m: float = 0.001
    iterations: int = 160
    damping: float = 0.01
    finite_difference_rad: float = 1.0e-4
    max_step: float = 0.03
    distance_tolerance_m: float = 0.00015
    query_cutoff_m: float = 0.15
    line_search_steps: int = 8

    def __post_init__(self) -> None:
        if int(self.iterations) < 1 or int(self.line_search_steps) < 1:
            raise ValueError("iterations and line_search_steps must be positive.")
        for name in (
            "target_distance_m",
            "contact_tolerance_m",
            "max_penetration_m",
            "damping",
            "finite_difference_rad",
            "max_step",
            "distance_tolerance_m",
            "query_cutoff_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.target_distance_m > self.contact_tolerance_m:
            raise ValueError("target_distance_m must lie inside the contact band.")
        if self.query_cutoff_m <= self.contact_tolerance_m:
            raise ValueError("query_cutoff_m must exceed contact_tolerance_m.")


@dataclass
class ContactConstraintProjectionReport:
    """Contact-band evidence after projection."""

    frame_count: int = 0
    source_active_side_frames: int = 0
    contact_side_frames_before: int = 0
    contact_side_frames_after: int = 0
    corrected_side_frames: int = 0
    mean_joint_shift: float = 0.0
    max_joint_shift: float = 0.0
    minimum_distance_m: float = float("inf")
    maximum_selected_distance_m: float = 0.0
    per_side_contact_frames: dict[str, int] = field(default_factory=dict)
    unreached_frames: dict[str, list[int]] = field(default_factory=dict)
    selected_slot_frames: dict[str, dict[int, int]] = field(default_factory=dict)

    @property
    def recovery_rate(self) -> float:
        if self.source_active_side_frames == 0:
            return 0.0
        return self.contact_side_frames_after / self.source_active_side_frames

    @property
    def qualified(self) -> bool:
        return (
            self.source_active_side_frames > 0
            and self.contact_side_frames_after == self.source_active_side_frames
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "qualified": self.qualified,
            "frame_count": self.frame_count,
            "source_active_side_frames": self.source_active_side_frames,
            "contact_side_frames_before": self.contact_side_frames_before,
            "contact_side_frames_after": self.contact_side_frames_after,
            "corrected_side_frames": self.corrected_side_frames,
            "recovery_rate": self.recovery_rate,
            "mean_joint_shift": self.mean_joint_shift,
            "max_joint_shift": self.max_joint_shift,
            "minimum_distance_m": (
                None
                if not np.isfinite(self.minimum_distance_m)
                else float(self.minimum_distance_m)
            ),
            "maximum_selected_distance_m": self.maximum_selected_distance_m,
            "per_side_contact_frames": dict(self.per_side_contact_frames),
            "unreached_frames": {
                side: list(frames) for side, frames in self.unreached_frames.items()
            },
            "selected_slot_frames": {
                side: {str(slot): count for slot, count in sorted(counts.items())}
                for side, counts in self.selected_slot_frames.items()
            },
        }


class MujocoContactConstraintProjector:
    """Attract one active semantic link per hand to an object surface."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        trajectory_joint_names: Sequence[str],
        variable_joint_names: Mapping[str, Sequence[str]],
        contact_geom_names: Mapping[str, Sequence[str | int]],
        object_geom_name: str | int,
        object_mocap_body_name: str,
        config: ContactConstraintProjectionConfig | None = None,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.config = config or ContactConstraintProjectionConfig()
        self.sides = tuple(str(side) for side in contact_geom_names)
        if not self.sides or tuple(variable_joint_names) != self.sides:
            raise ValueError(
                "variable_joint_names and contact_geom_names must share side order."
            )
        self.trajectory_joint_names = tuple(
            str(name) for name in trajectory_joint_names
        )
        if not self.trajectory_joint_names or len(
            set(self.trajectory_joint_names)
        ) != len(self.trajectory_joint_names):
            raise ValueError("trajectory_joint_names must be non-empty and unique.")
        columns = {
            name: index for index, name in enumerate(self.trajectory_joint_names)
        }
        self.joint_ids = np.asarray(
            [
                _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.trajectory_joint_names
            ],
            dtype=np.int32,
        )
        self.qpos_addresses = np.asarray(
            model.jnt_qposadr[self.joint_ids], dtype=np.int32
        )
        self.variable_columns: dict[str, np.ndarray] = {}
        self.variable_dofs: dict[str, np.ndarray] = {}
        self.lower: dict[str, np.ndarray] = {}
        self.upper: dict[str, np.ndarray] = {}
        for side in self.sides:
            names = tuple(str(name) for name in variable_joint_names[side])
            if not names or len(set(names)) != len(names):
                raise ValueError(
                    f"{side} variable joints must be non-empty and unique."
                )
            missing = set(names) - set(columns)
            if missing:
                raise ValueError(f"{side} variable joints are absent: {missing}.")
            side_columns = np.asarray([columns[name] for name in names], dtype=np.int32)
            side_joint_ids = self.joint_ids[side_columns]
            unsupported = [
                names[index]
                for index, joint_id in enumerate(side_joint_ids)
                if int(model.jnt_type[joint_id])
                not in (
                    int(mujoco.mjtJoint.mjJNT_HINGE),
                    int(mujoco.mjtJoint.mjJNT_SLIDE),
                )
            ]
            if unsupported:
                raise ValueError(
                    f"Only scalar variable joints are supported: {unsupported}."
                )
            self.variable_columns[side] = side_columns
            self.variable_dofs[side] = np.asarray(
                model.jnt_dofadr[side_joint_ids], dtype=np.int32
            )
            self.lower[side] = np.asarray(
                model.jnt_range[side_joint_ids, 0], dtype=np.float64
            )
            self.upper[side] = np.asarray(
                model.jnt_range[side_joint_ids, 1], dtype=np.float64
            )
        self.contact_geom_ids = {
            side: tuple(
                _object_id(model, mujoco.mjtObj.mjOBJ_GEOM, value)
                for value in contact_geom_names[side]
            )
            for side in self.sides
        }
        slot_counts = {len(values) for values in self.contact_geom_ids.values()}
        if len(slot_counts) != 1 or not slot_counts or next(iter(slot_counts)) == 0:
            raise ValueError("Every side must provide the same non-zero contact slots.")
        self.slot_count = next(iter(slot_counts))
        self.object_geom_id = _object_id(
            model, mujoco.mjtObj.mjOBJ_GEOM, object_geom_name
        )
        body_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, object_mocap_body_name)
        self.object_mocap_id = int(model.body_mocapid[body_id])
        if self.object_mocap_id < 0:
            raise ValueError(f"Body {object_mocap_body_name!r} is not a mocap body.")

    def _set_frame(self, row: np.ndarray, pose: np.ndarray) -> None:
        self.data.qpos[self.qpos_addresses] = row
        self.data.mocap_pos[self.object_mocap_id] = pose[:3]
        self.data.mocap_quat[self.object_mocap_id] = pose[3:7]
        mujoco.mj_forward(self.model, self.data)

    def _distance_with_segment(self, geom_id: int) -> tuple[float, np.ndarray]:
        segment = np.zeros(6, dtype=np.float64)
        distance = float(
            mujoco.mj_geomDistance(
                self.model,
                self.data,
                geom_id,
                self.object_geom_id,
                self.config.query_cutoff_m,
                segment,
            )
        )
        return distance, segment

    def _distance(self, geom_id: int) -> float:
        return self._distance_with_segment(geom_id)[0]

    def _finite_difference_gradient(
        self, row: np.ndarray, pose: np.ndarray, side: str, geom_id: int
    ) -> np.ndarray:
        columns = self.variable_columns[side]
        lower = self.lower[side]
        upper = self.upper[side]
        gradient = np.zeros(len(columns), dtype=np.float64)
        epsilon = self.config.finite_difference_rad
        for local_index, column in enumerate(columns):
            original = float(row[column])
            low = max(float(lower[local_index]), original - epsilon)
            high = min(float(upper[local_index]), original + epsilon)
            if high <= low + 1.0e-12:
                continue
            row[column] = high
            self._set_frame(row, pose)
            high_distance = self._distance(geom_id)
            row[column] = low
            self._set_frame(row, pose)
            low_distance = self._distance(geom_id)
            row[column] = original
            gradient[local_index] = (high_distance - low_distance) / (high - low)
        self._set_frame(row, pose)
        return gradient

    def _distance_gradient(
        self, row: np.ndarray, pose: np.ndarray, side: str, geom_id: int
    ) -> np.ndarray:
        self._set_frame(row, pose)
        distance, segment = self._distance_with_segment(geom_id)
        direction = segment[:3] - segment[3:]
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-10:
            direction = (
                self.data.geom_xpos[geom_id] - self.data.geom_xpos[self.object_geom_id]
            )
            norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-10:
            return self._finite_difference_gradient(row, pose, side, geom_id)
        direction /= norm
        jacobian_first = np.zeros((3, self.model.nv), dtype=np.float64)
        jacobian_second = np.zeros((3, self.model.nv), dtype=np.float64)
        rotation_jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jac(
            self.model,
            self.data,
            jacobian_first,
            rotation_jacobian,
            segment[:3],
            int(self.model.geom_bodyid[geom_id]),
        )
        rotation_jacobian.fill(0.0)
        mujoco.mj_jac(
            self.model,
            self.data,
            jacobian_second,
            rotation_jacobian,
            segment[3:],
            int(self.model.geom_bodyid[self.object_geom_id]),
        )
        gradient = direction @ (
            jacobian_first[:, self.variable_dofs[side]]
            - jacobian_second[:, self.variable_dofs[side]]
        )
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm <= 1.0e-10 or not np.isfinite(gradient).all():
            return self._finite_difference_gradient(row, pose, side, geom_id)
        current = row[self.variable_columns[side]].copy()
        probe = np.clip(
            current + self.config.finite_difference_rad * gradient / gradient_norm,
            self.lower[side],
            self.upper[side],
        )
        row[self.variable_columns[side]] = probe
        self._set_frame(row, pose)
        probe_distance = self._distance(geom_id)
        row[self.variable_columns[side]] = current
        self._set_frame(row, pose)
        if probe_distance < distance:
            gradient = -gradient
        return gradient

    def _side_in_contact(
        self, row: np.ndarray, pose: np.ndarray, side: str, slots: np.ndarray
    ) -> tuple[bool, float, int]:
        self._set_frame(row, pose)
        distances = [
            (self._distance(self.contact_geom_ids[side][int(slot)]), int(slot))
            for slot in slots
        ]
        distance, slot = min(
            distances,
            key=lambda value: abs(value[0] - self.config.target_distance_m),
        )
        in_band = (
            distance <= self.config.contact_tolerance_m
            and distance >= -self.config.max_penetration_m
        )
        return in_band, float(distance), slot

    def project(
        self,
        qpos: np.ndarray,
        *,
        object_poses_wxyz: np.ndarray,
        active: np.ndarray,
    ) -> tuple[np.ndarray, ContactConstraintProjectionReport]:
        """Return a side-local semantic contact projection."""

        values = np.asarray(qpos, dtype=np.float64)
        poses = np.asarray(object_poses_wxyz, dtype=np.float64)
        valid = np.asarray(active, dtype=bool)
        if values.ndim != 2 or values.shape[1] != len(self.trajectory_joint_names):
            raise ValueError("qpos must have shape [T, trajectory joints].")
        if poses.shape != (len(values), 7):
            raise ValueError("object_poses_wxyz must have shape [T, 7].")
        if valid.shape != (len(values), len(self.sides), self.slot_count):
            raise ValueError("active must have shape [T, sides, contact slots].")
        if not np.isfinite(values).all() or not np.isfinite(poses).all():
            raise ValueError("qpos/object poses contain non-finite values.")
        if not np.allclose(
            np.linalg.norm(poses[:, 3:7], axis=1), 1.0, rtol=0.0, atol=1.0e-5
        ):
            raise ValueError("object_poses_wxyz contains non-unit quaternions.")

        output = values.copy()
        report = ContactConstraintProjectionReport(frame_count=len(output))
        report.per_side_contact_frames = {side: 0 for side in self.sides}
        report.unreached_frames = {side: [] for side in self.sides}
        report.selected_slot_frames = {side: {} for side in self.sides}
        shifts: list[float] = []
        selected_distances: list[float] = []
        for frame, row in enumerate(output):
            for side_index, side in enumerate(self.sides):
                slots = np.flatnonzero(valid[frame, side_index])
                if len(slots) == 0:
                    continue
                # Select the closest source-active semantic link once per
                # frame and retain it through the local solve.  Re-selecting
                # during an iteration can jump IK branches, while forcing one
                # globally frequent link can move the wrist when that link is
                # locally unreachable.
                before_contact, _, slot = self._side_in_contact(
                    row, poses[frame], side, slots
                )
                selected_slots = np.asarray((slot,), dtype=np.int64)
                side_counts = report.selected_slot_frames[side]
                side_counts[slot] = side_counts.get(slot, 0) + 1
                report.source_active_side_frames += 1
                seed = row.copy()
                if before_contact:
                    report.contact_side_frames_before += 1
                for _ in range(self.config.iterations):
                    in_contact, distance, slot = self._side_in_contact(
                        row, poses[frame], side, selected_slots
                    )
                    if (
                        in_contact
                        and abs(distance - self.config.target_distance_m)
                        <= self.config.distance_tolerance_m
                    ):
                        break
                    geom_id = self.contact_geom_ids[side][slot]
                    gradient = self._distance_gradient(row, poses[frame], side, geom_id)
                    denominator = float(gradient @ gradient) + self.config.damping**2
                    if denominator <= 1.0e-12:
                        break
                    delta = gradient * (
                        (self.config.target_distance_m - distance) / denominator
                    )
                    delta_norm = float(np.linalg.norm(delta))
                    if delta_norm > self.config.max_step:
                        delta *= self.config.max_step / delta_norm
                    columns = self.variable_columns[side]
                    current = row[columns].copy()
                    objective = abs(distance - self.config.target_distance_m)
                    accepted = False
                    for search_step in range(self.config.line_search_steps):
                        row[columns] = np.clip(
                            current + 0.5**search_step * delta,
                            self.lower[side],
                            self.upper[side],
                        )
                        self._set_frame(row, poses[frame])
                        candidate_distance = self._distance(geom_id)
                        if (
                            abs(candidate_distance - self.config.target_distance_m)
                            < objective
                        ):
                            accepted = True
                            break
                    if not accepted:
                        row[columns] = current
                        break
                after_contact, distance, _ = self._side_in_contact(
                    row, poses[frame], side, selected_slots
                )
                selected_distances.append(distance)
                report.minimum_distance_m = min(report.minimum_distance_m, distance)
                if after_contact:
                    report.contact_side_frames_after += 1
                    report.per_side_contact_frames[side] += 1
                    if not before_contact:
                        report.corrected_side_frames += 1
                else:
                    report.unreached_frames[side].append(frame)
                shifts.extend(np.abs(row - seed))
        if shifts:
            report.mean_joint_shift = float(np.mean(shifts))
            report.max_joint_shift = float(np.max(shifts))
        if selected_distances:
            report.maximum_selected_distance_m = float(np.max(selected_distances))
        return output, report


__all__ = [
    "ContactConstraintProjectionConfig",
    "ContactConstraintProjectionReport",
    "MujocoContactConstraintProjector",
]
