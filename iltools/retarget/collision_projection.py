"""Kinematic projection of retargeted poses out of forbidden scene geometry.

Contact retargeting must permit the hand to touch the manipulated object, but
that does not make every collision intentional.  This module resolves a named
set of forbidden robot/scene pairs (for example hand/arm against a tabletop)
while preserving selected task sites and staying close to the input pose.

The signed-distance gradient is computed by finite differences.  That is more
expensive than assuming a closest-point normal remains valid through
penetration, but it is deterministic and works for MuJoCo mesh/cylinder pairs
whose penetration witnesses can change discontinuously.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np


@dataclass(frozen=True)
class CollisionProjectionConfig:
    """Numerical and geometric settings for forbidden-contact projection."""

    clearance_m: float = 0.005
    iterations: int = 30
    damping: float = 0.02
    regularization: float = 0.01
    collision_weight: float = 8.0
    preserve_site_weight: float = 1.0
    finite_difference_rad: float = 1.0e-4
    max_step: float = 0.08
    tolerance_m: float = 2.0e-4
    query_margin_m: float = 0.02
    line_search_steps: int = 8

    def __post_init__(self) -> None:
        if int(self.iterations) < 1 or int(self.line_search_steps) < 1:
            raise ValueError("iterations and line_search_steps must be positive.")
        if not np.isfinite(self.clearance_m):
            raise ValueError("clearance_m must be finite.")
        for name in (
            "damping",
            "regularization",
            "collision_weight",
            "preserve_site_weight",
            "finite_difference_rad",
            "max_step",
            "tolerance_m",
            "query_margin_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.clearance_m + self.query_margin_m <= 0.0:
            raise ValueError(
                "query_margin_m must extend beyond a negative clearance_m."
            )


@dataclass
class CollisionProjectionReport:
    """Evidence that forbidden penetrations were removed or remained."""

    frame_count: int = 0
    violating_frames_before: int = 0
    violating_frames_after: int = 0
    corrected_frames: int = 0
    worst_clearance_before_m: float = 0.0
    worst_clearance_after_m: float = 0.0
    mean_joint_shift: float = 0.0
    max_joint_shift: float = 0.0
    mean_preserved_site_drift_m: float = 0.0
    max_preserved_site_drift_m: float = 0.0
    residual_frames: list[int] = field(default_factory=list)

    @property
    def qualified(self) -> bool:
        return self.violating_frames_after == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "qualified": self.qualified,
            "frame_count": self.frame_count,
            "violating_frames_before": self.violating_frames_before,
            "violating_frames_after": self.violating_frames_after,
            "corrected_frames": self.corrected_frames,
            "worst_clearance_before_m": self.worst_clearance_before_m,
            "worst_clearance_after_m": self.worst_clearance_after_m,
            "mean_joint_shift": self.mean_joint_shift,
            "max_joint_shift": self.max_joint_shift,
            "mean_preserved_site_drift_m": self.mean_preserved_site_drift_m,
            "max_preserved_site_drift_m": self.max_preserved_site_drift_m,
            "residual_frames": list(self.residual_frames),
        }


@dataclass(frozen=True)
class RadialEscapeProjectionConfig:
    """Settings for escaping a multi-geometry penetration basin."""

    clearance_m: float = -0.0005
    tolerance_m: float = 1.0e-5
    retreat_step_m: float = 0.005
    maximum_retreat_m: float = 0.10
    ik_iterations: int = 100
    damping: float = 0.01
    max_step: float = 0.05
    position_tolerance_m: float = 1.0e-5
    query_margin_m: float = 0.02

    def __post_init__(self) -> None:
        if int(self.ik_iterations) < 1:
            raise ValueError("ik_iterations must be positive.")
        if not np.isfinite(self.clearance_m):
            raise ValueError("clearance_m must be finite.")
        for name in (
            "tolerance_m",
            "retreat_step_m",
            "maximum_retreat_m",
            "damping",
            "max_step",
            "position_tolerance_m",
            "query_margin_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.maximum_retreat_m < self.retreat_step_m:
            raise ValueError("maximum_retreat_m must cover at least one retreat step.")
        if self.clearance_m + self.query_margin_m <= 0.0:
            raise ValueError(
                "query_margin_m must extend beyond a negative clearance_m."
            )


@dataclass(frozen=True)
class SceneCollisionClosureProjectionConfig:
    """Settings for maximal safe finger closure against a moving scene."""

    clearance_m: float = -0.0005
    tolerance_m: float = 1.0e-5
    search_iterations: int = 20
    query_margin_m: float = 0.02

    def __post_init__(self) -> None:
        if int(self.search_iterations) < 1:
            raise ValueError("search_iterations must be positive.")
        if not np.isfinite(self.clearance_m):
            raise ValueError("clearance_m must be finite.")
        for name in ("tolerance_m", "query_margin_m"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.clearance_m + self.query_margin_m <= 0.0:
            raise ValueError(
                "query_margin_m must extend beyond a negative clearance_m."
            )


def _object_id(model: mujoco.MjModel, kind: Any, identifier: str | int) -> int:
    if isinstance(identifier, (int, np.integer)):
        result = int(identifier)
    else:
        result = int(mujoco.mj_name2id(model, kind, str(identifier)))
    if result < 0:
        raise ValueError(f"MuJoCo object {identifier!r} does not exist.")
    return result


class MujocoCollisionProjector:
    """Project trajectory samples out of explicitly forbidden scene geoms."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        trajectory_joint_names: Sequence[str],
        variable_joint_names: Sequence[str],
        robot_geom_names: Sequence[str | int] = (),
        scene_geom_names: Sequence[str | int] = (),
        forbidden_geom_pairs: Sequence[tuple[str | int, str | int]] = (),
        preserve_site_names: Sequence[str] = (),
        scene_mocap_body_names: Sequence[str] = (),
        config: CollisionProjectionConfig | None = None,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.config = config or CollisionProjectionConfig()
        self.trajectory_joint_names = tuple(
            str(name) for name in trajectory_joint_names
        )
        self.variable_joint_names = tuple(str(name) for name in variable_joint_names)
        if not self.trajectory_joint_names or len(
            set(self.trajectory_joint_names)
        ) != len(self.trajectory_joint_names):
            raise ValueError("trajectory_joint_names must be non-empty and unique.")
        if not self.variable_joint_names or len(set(self.variable_joint_names)) != len(
            self.variable_joint_names
        ):
            raise ValueError("variable_joint_names must be non-empty and unique.")
        columns = {
            name: index for index, name in enumerate(self.trajectory_joint_names)
        }
        missing = set(self.variable_joint_names) - set(columns)
        if missing:
            raise ValueError(f"Variable joints are absent from trajectory: {missing}.")

        self.joint_ids = np.asarray(
            [
                _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.trajectory_joint_names
            ],
            dtype=np.int32,
        )
        unsupported = [
            self.trajectory_joint_names[index]
            for index, joint_id in enumerate(self.joint_ids)
            if int(model.jnt_type[joint_id])
            not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            )
        ]
        if unsupported:
            raise ValueError(
                f"Only scalar trajectory joints are supported: {unsupported}."
            )
        self.qpos_addresses = np.asarray(
            model.jnt_qposadr[self.joint_ids], dtype=np.int32
        )
        self.variable_columns = np.asarray(
            [columns[name] for name in self.variable_joint_names], dtype=np.int32
        )
        self.variable_joint_ids = self.joint_ids[self.variable_columns]
        self.variable_dofs = np.asarray(
            model.jnt_dofadr[self.variable_joint_ids], dtype=np.int32
        )
        self.lower = np.asarray(
            [model.jnt_range[joint_id, 0] for joint_id in self.variable_joint_ids],
            dtype=np.float64,
        )
        self.upper = np.asarray(
            [model.jnt_range[joint_id, 1] for joint_id in self.variable_joint_ids],
            dtype=np.float64,
        )
        self.robot_geom_ids = tuple(
            _object_id(model, mujoco.mjtObj.mjOBJ_GEOM, value)
            for value in robot_geom_names
        )
        self.scene_geom_ids = tuple(
            _object_id(model, mujoco.mjtObj.mjOBJ_GEOM, value)
            for value in scene_geom_names
        )
        explicit_pairs = tuple(
            (
                _object_id(model, mujoco.mjtObj.mjOBJ_GEOM, first),
                _object_id(model, mujoco.mjtObj.mjOBJ_GEOM, second),
            )
            for first, second in forbidden_geom_pairs
        )
        if explicit_pairs:
            self.geom_pairs = tuple(dict.fromkeys(explicit_pairs))
        else:
            if not self.robot_geom_ids or not self.scene_geom_ids:
                raise ValueError(
                    "Provide forbidden_geom_pairs or non-empty robot/scene groups."
                )
            overlap = set(self.robot_geom_ids) & set(self.scene_geom_ids)
            if overlap:
                raise ValueError(f"Robot and scene geoms overlap: {sorted(overlap)}.")
            self.geom_pairs = tuple(
                (robot_geom, scene_geom)
                for robot_geom in self.robot_geom_ids
                for scene_geom in self.scene_geom_ids
            )
        self.site_ids = tuple(
            _object_id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in preserve_site_names
        )
        self.scene_mocap_ids = tuple(
            self._mocap_id(name) for name in scene_mocap_body_names
        )
        self._query_cutoff = self.config.clearance_m + self.config.query_margin_m

    def _mocap_id(self, name: str) -> int:
        body_id = _object_id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        mocap_id = int(self.model.body_mocapid[body_id])
        if mocap_id < 0:
            raise ValueError(f"Body {name!r} is not a mocap body.")
        return mocap_id

    def _set_scene_mocap(self, poses: np.ndarray) -> None:
        for body_index, mocap_id in enumerate(self.scene_mocap_ids):
            self.data.mocap_pos[mocap_id] = poses[body_index, :3]
            self.data.mocap_quat[mocap_id] = poses[body_index, 3:7]

    def _set_q(self, row: np.ndarray) -> None:
        self.data.qpos[self.qpos_addresses] = row
        mujoco.mj_forward(self.model, self.data)

    def _pair_distances(
        self,
        pairs: Sequence[tuple[int, int]] | None = None,
    ) -> list[tuple[int, int, float]]:
        segment = np.zeros(6, dtype=np.float64)
        result: list[tuple[int, int, float]] = []
        for robot_geom, scene_geom in self.geom_pairs if pairs is None else pairs:
            distance = float(
                mujoco.mj_geomDistance(
                    self.model,
                    self.data,
                    robot_geom,
                    scene_geom,
                    self._query_cutoff,
                    segment,
                )
            )
            result.append((robot_geom, scene_geom, distance))
        return result

    def _distance(self, robot_geom: int, scene_geom: int) -> float:
        distance, _ = self._distance_with_segment(robot_geom, scene_geom)
        return distance

    def _distance_with_segment(
        self, robot_geom: int, scene_geom: int
    ) -> tuple[float, np.ndarray]:
        segment = np.zeros(6, dtype=np.float64)
        distance = float(
            mujoco.mj_geomDistance(
                self.model,
                self.data,
                robot_geom,
                scene_geom,
                self._query_cutoff,
                segment,
            )
        )
        return distance, segment

    def _finite_difference_gradient(
        self,
        row: np.ndarray,
        robot_geom: int,
        scene_geom: int,
    ) -> np.ndarray:
        gradient = np.zeros(len(self.variable_columns), dtype=np.float64)
        epsilon = self.config.finite_difference_rad
        for local_index, column in enumerate(self.variable_columns):
            original = float(row[column])
            low = max(float(self.lower[local_index]), original - epsilon)
            high = min(float(self.upper[local_index]), original + epsilon)
            if high <= low + 1.0e-12:
                continue
            row[column] = high
            self._set_q(row)
            distance_high = self._distance(robot_geom, scene_geom)
            row[column] = low
            self._set_q(row)
            distance_low = self._distance(robot_geom, scene_geom)
            row[column] = original
            gradient[local_index] = (distance_high - distance_low) / (high - low)
        self._set_q(row)
        return gradient

    def _distance_gradient(
        self,
        row: np.ndarray,
        robot_geom: int,
        scene_geom: int,
    ) -> np.ndarray:
        """Differentiate a pair distance, validating the witness orientation."""

        self._set_q(row)
        distance, segment = self._distance_with_segment(robot_geom, scene_geom)
        # Penetrating mesh witnesses can jump between faces and their segment
        # orientation is not a reliable signed-distance derivative.  Central
        # differences are slower but measure the actual local escape direction
        # and are only needed for already-invalid configurations.
        if distance < 0.0:
            return self._finite_difference_gradient(row, robot_geom, scene_geom)
        direction = segment[:3] - segment[3:]
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1.0e-10:
            direction = (
                self.data.geom_xpos[robot_geom] - self.data.geom_xpos[scene_geom]
            )
            direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1.0e-10:
            return self._finite_difference_gradient(row, robot_geom, scene_geom)
        direction /= direction_norm

        jacobian_first = np.zeros((3, self.model.nv), dtype=np.float64)
        jacobian_second = np.zeros((3, self.model.nv), dtype=np.float64)
        rotation_jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
        body_first = int(self.model.geom_bodyid[robot_geom])
        body_second = int(self.model.geom_bodyid[scene_geom])
        mujoco.mj_jac(
            self.model,
            self.data,
            jacobian_first,
            rotation_jacobian,
            segment[:3],
            body_first,
        )
        rotation_jacobian.fill(0.0)
        mujoco.mj_jac(
            self.model,
            self.data,
            jacobian_second,
            rotation_jacobian,
            segment[3:],
            body_second,
        )
        gradient = direction @ (
            jacobian_first[:, self.variable_dofs]
            - jacobian_second[:, self.variable_dofs]
        )
        norm = float(np.linalg.norm(gradient))
        if norm <= 1.0e-10 or not np.isfinite(gradient).all():
            return self._finite_difference_gradient(row, robot_geom, scene_geom)

        # Penetrating mesh witnesses do not guarantee a consistent point
        # ordering.  One directional query establishes the sign without
        # finite-differencing every joint.
        epsilon = self.config.finite_difference_rad
        current = row[self.variable_columns].copy()
        probe = np.clip(current + epsilon * gradient / norm, self.lower, self.upper)
        if np.allclose(probe, current, rtol=0.0, atol=1.0e-14):
            return self._finite_difference_gradient(row, robot_geom, scene_geom)
        row[self.variable_columns] = probe
        self._set_q(row)
        probe_distance = self._distance(robot_geom, scene_geom)
        row[self.variable_columns] = current
        self._set_q(row)
        if probe_distance < distance:
            gradient = -gradient
        return gradient

    def project(
        self,
        qpos: np.ndarray,
        *,
        scene_mocap_poses: np.ndarray | None = None,
        active_geom_pairs_by_frame: Sequence[Sequence[tuple[str | int, str | int]]]
        | None = None,
    ) -> tuple[np.ndarray, CollisionProjectionReport]:
        """Return a projected copy of ``qpos`` and a fail-closed report."""

        values = np.asarray(qpos, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.trajectory_joint_names):
            raise ValueError("qpos must have shape [T, trajectory joints].")
        if not np.isfinite(values).all():
            raise ValueError("qpos contains non-finite values.")
        poses: np.ndarray | None = None
        if scene_mocap_poses is not None:
            poses = np.asarray(scene_mocap_poses, dtype=np.float64)
            if poses.shape != (len(values), len(self.scene_mocap_ids), 7):
                raise ValueError(
                    "scene_mocap_poses must have shape [T, configured bodies, 7]."
                )
            if not np.isfinite(poses).all():
                raise ValueError("scene_mocap_poses contains non-finite values.")
            quaternion_norms = np.linalg.norm(poses[..., 3:7], axis=-1)
            if not np.allclose(quaternion_norms, 1.0, rtol=0.0, atol=1.0e-5):
                raise ValueError("scene_mocap_poses contains non-unit quaternions.")
        elif self.scene_mocap_ids:
            raise ValueError(
                "scene_mocap_poses is required when scene mocap bodies are configured."
            )
        frame_pairs: list[tuple[tuple[int, int], ...]] | None = None
        if active_geom_pairs_by_frame is not None:
            if len(active_geom_pairs_by_frame) != len(values):
                raise ValueError("active_geom_pairs_by_frame must align with qpos.")
            allowed = frozenset(self.geom_pairs)
            frame_pairs = []
            for pairs in active_geom_pairs_by_frame:
                resolved = tuple(
                    (
                        _object_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, first),
                        _object_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, second),
                    )
                    for first, second in pairs
                )
                unknown = set(resolved) - allowed
                if unknown:
                    raise ValueError(
                        f"Per-frame pairs are absent from forbidden pairs: {unknown}."
                    )
                frame_pairs.append(resolved)
        output = values.copy()
        report = CollisionProjectionReport(frame_count=len(output))
        joint_shifts: list[float] = []
        site_drifts: list[float] = []
        worst_before = self._query_cutoff
        worst_after = self._query_cutoff

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)

        def objective(
            row: np.ndarray,
            seed: np.ndarray,
            pairs: Sequence[tuple[int, int]] | None,
            site_targets: np.ndarray,
        ) -> float:
            self._set_q(row)
            collision = sum(
                (
                    self.config.collision_weight
                    * max(self.config.clearance_m - distance, 0.0)
                )
                ** 2
                for _, _, distance in self._pair_distances(pairs)
            )
            preservation = sum(
                (
                    self.config.preserve_site_weight
                    * float(np.linalg.norm(self.data.site_xpos[site_id] - target))
                )
                ** 2
                for target, site_id in zip(site_targets, self.site_ids, strict=True)
            )
            regularization = self.config.regularization * float(
                np.sum((row[self.variable_columns] - seed[self.variable_columns]) ** 2)
            )
            return float(collision + preservation + regularization)

        for frame, row in enumerate(output):
            if poses is not None:
                self._set_scene_mocap(poses[frame])
            active_pairs = None if frame_pairs is None else frame_pairs[frame]
            if active_pairs is not None and not active_pairs:
                continue
            seed = row.copy()
            self._set_q(row)
            site_targets = np.asarray(
                [self.data.site_xpos[site_id].copy() for site_id in self.site_ids],
                dtype=np.float64,
            )
            before_pairs = self._pair_distances(active_pairs)
            before = min(distance for _, _, distance in before_pairs)
            worst_before = min(worst_before, before)
            violated_before = before < self.config.clearance_m - self.config.tolerance_m
            if violated_before:
                report.violating_frames_before += 1

            for _ in range(self.config.iterations):
                self._set_q(row)
                violating = [
                    pair
                    for pair in self._pair_distances(active_pairs)
                    if pair[2] < self.config.clearance_m - self.config.tolerance_m
                ]
                if not violating:
                    break
                rows: list[np.ndarray] = []
                errors: list[np.ndarray] = []
                for robot_geom, scene_geom, distance in violating:
                    gradient = self._distance_gradient(row, robot_geom, scene_geom)
                    if float(np.linalg.norm(gradient)) <= 1.0e-10:
                        continue
                    rows.append(self.config.collision_weight * gradient[None, :])
                    errors.append(
                        np.asarray(
                            [
                                self.config.collision_weight
                                * (self.config.clearance_m - distance)
                            ]
                        )
                    )
                self._set_q(row)
                for target, site_id in zip(site_targets, self.site_ids, strict=True):
                    jacp.fill(0.0)
                    jacr.fill(0.0)
                    mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
                    rows.append(
                        self.config.preserve_site_weight * jacp[:, self.variable_dofs]
                    )
                    errors.append(
                        self.config.preserve_site_weight
                        * (target - self.data.site_xpos[site_id])
                    )
                if not rows:
                    break
                jacobian = np.concatenate(rows, axis=0)
                error = np.concatenate(errors)
                normal = jacobian.T @ jacobian
                normal += (
                    self.config.damping**2 + self.config.regularization
                ) * np.eye(len(self.variable_columns))
                current = row[self.variable_columns]
                right = jacobian.T @ error + self.config.regularization * (
                    seed[self.variable_columns] - current
                )
                delta = np.linalg.solve(normal, right)
                norm = float(np.linalg.norm(delta))
                if norm > self.config.max_step:
                    delta *= self.config.max_step / norm
                current = current.copy()
                current_objective = objective(row, seed, active_pairs, site_targets)
                accepted = False
                for search_step in range(self.config.line_search_steps):
                    row[self.variable_columns] = np.clip(
                        current + 0.5**search_step * delta,
                        self.lower,
                        self.upper,
                    )
                    if (
                        objective(row, seed, active_pairs, site_targets)
                        < current_objective - 1.0e-16
                    ):
                        accepted = True
                        break
                if not accepted:
                    row[self.variable_columns] = current
                    break

            self._set_q(row)
            after = min(
                distance for _, _, distance in self._pair_distances(active_pairs)
            )
            worst_after = min(worst_after, after)
            violated_after = after < self.config.clearance_m - self.config.tolerance_m
            if violated_after:
                report.violating_frames_after += 1
                report.residual_frames.append(frame)
            elif violated_before:
                report.corrected_frames += 1
            joint_shifts.extend(np.abs(row - seed))
            for target, site_id in zip(site_targets, self.site_ids, strict=True):
                site_drifts.append(
                    float(np.linalg.norm(self.data.site_xpos[site_id] - target))
                )

        report.worst_clearance_before_m = float(worst_before)
        report.worst_clearance_after_m = float(worst_after)
        if joint_shifts:
            report.mean_joint_shift = float(np.mean(joint_shifts))
            report.max_joint_shift = float(np.max(joint_shifts))
        if site_drifts:
            report.mean_preserved_site_drift_m = float(np.mean(site_drifts))
            report.max_preserved_site_drift_m = float(np.max(site_drifts))
        return output, report


class MujocoRadialEscapeProjector:
    """Move one task site radially out of a nonlocal penetration basin.

    A hand wrapped around an object can place several links in a signed-distance
    local minimum: every infinitesimal joint perturbation makes at least one
    penetration worse.  This projector performs a bounded Cartesian retreat of
    the palm using only the authorized arm joints.  A local collision projector
    can then finish the surface correction, and a semantic contact projector can
    re-establish a measured contact on a finger.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        trajectory_joint_names: Sequence[str],
        variable_joint_names: Sequence[str],
        robot_geom_names: Sequence[str | int],
        scene_geom_name: str | int,
        target_site_name: str,
        scene_mocap_body_name: str,
        config: RadialEscapeProjectionConfig | None = None,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.config = config or RadialEscapeProjectionConfig()
        self.trajectory_joint_names = tuple(
            str(name) for name in trajectory_joint_names
        )
        self.variable_joint_names = tuple(str(name) for name in variable_joint_names)
        if not self.trajectory_joint_names or len(
            set(self.trajectory_joint_names)
        ) != len(self.trajectory_joint_names):
            raise ValueError("trajectory_joint_names must be non-empty and unique.")
        if not self.variable_joint_names or len(set(self.variable_joint_names)) != len(
            self.variable_joint_names
        ):
            raise ValueError("variable_joint_names must be non-empty and unique.")
        columns = {
            name: index for index, name in enumerate(self.trajectory_joint_names)
        }
        missing = set(self.variable_joint_names) - set(columns)
        if missing:
            raise ValueError(f"Variable joints are absent from trajectory: {missing}.")
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
        self.variable_columns = np.asarray(
            [columns[name] for name in self.variable_joint_names], dtype=np.int32
        )
        variable_joint_ids = self.joint_ids[self.variable_columns]
        unsupported = [
            self.variable_joint_names[index]
            for index, joint_id in enumerate(variable_joint_ids)
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
        self.variable_dofs = np.asarray(
            model.jnt_dofadr[variable_joint_ids], dtype=np.int32
        )
        self.lower = np.asarray(
            model.jnt_range[variable_joint_ids, 0], dtype=np.float64
        )
        self.upper = np.asarray(
            model.jnt_range[variable_joint_ids, 1], dtype=np.float64
        )
        self.robot_geom_ids = tuple(
            _object_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in robot_geom_names
        )
        if not self.robot_geom_ids:
            raise ValueError("robot_geom_names must be non-empty.")
        self.scene_geom_id = _object_id(
            model, mujoco.mjtObj.mjOBJ_GEOM, scene_geom_name
        )
        self.target_site_id = _object_id(
            model, mujoco.mjtObj.mjOBJ_SITE, target_site_name
        )
        body_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, scene_mocap_body_name)
        self.scene_mocap_id = int(model.body_mocapid[body_id])
        if self.scene_mocap_id < 0:
            raise ValueError(f"Body {scene_mocap_body_name!r} is not a mocap body.")
        self._query_cutoff = self.config.clearance_m + self.config.query_margin_m

    def _set_frame(self, row: np.ndarray, pose: np.ndarray) -> None:
        self.data.qpos[self.qpos_addresses] = row
        self.data.mocap_pos[self.scene_mocap_id] = pose[:3]
        self.data.mocap_quat[self.scene_mocap_id] = pose[3:7]
        mujoco.mj_forward(self.model, self.data)

    def _minimum_distance(self) -> float:
        segment = np.zeros(6, dtype=np.float64)
        return min(
            float(
                mujoco.mj_geomDistance(
                    self.model,
                    self.data,
                    geom_id,
                    self.scene_geom_id,
                    self._query_cutoff,
                    segment,
                )
            )
            for geom_id in self.robot_geom_ids
        )

    def _solve_site_target(
        self, seed: np.ndarray, pose: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        row = seed.copy()
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        for _ in range(self.config.ik_iterations):
            self._set_frame(row, pose)
            error = target - self.data.site_xpos[self.target_site_id]
            if float(np.linalg.norm(error)) <= self.config.position_tolerance_m:
                break
            jacp.fill(0.0)
            jacr.fill(0.0)
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.target_site_id)
            jacobian = jacp[:, self.variable_dofs]
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + self.config.damping**2 * np.eye(3),
                error,
            )
            norm = float(np.linalg.norm(delta))
            if norm > self.config.max_step:
                delta *= self.config.max_step / norm
            row[self.variable_columns] = np.clip(
                row[self.variable_columns] + delta,
                self.lower,
                self.upper,
            )
        return row

    def project(
        self,
        qpos: np.ndarray,
        *,
        scene_mocap_poses: np.ndarray,
    ) -> tuple[np.ndarray, CollisionProjectionReport]:
        """Return a bounded radial escape for each penetrating frame."""

        values = np.asarray(qpos, dtype=np.float64)
        poses = np.asarray(scene_mocap_poses, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.trajectory_joint_names):
            raise ValueError("qpos must have shape [T, trajectory joints].")
        if poses.shape != (len(values), 1, 7):
            raise ValueError("scene_mocap_poses must have shape [T, 1, 7].")
        if not np.isfinite(values).all() or not np.isfinite(poses).all():
            raise ValueError("qpos/scene_mocap_poses contain non-finite values.")
        if not np.allclose(
            np.linalg.norm(poses[..., 3:7], axis=-1),
            1.0,
            rtol=0.0,
            atol=1.0e-5,
        ):
            raise ValueError("scene_mocap_poses contains non-unit quaternions.")

        output = values.copy()
        report = CollisionProjectionReport(frame_count=len(output))
        shifts: list[float] = []
        worst_before = self._query_cutoff
        worst_after = self._query_cutoff
        threshold = self.config.clearance_m - self.config.tolerance_m
        retreat_distances = np.arange(
            self.config.retreat_step_m,
            self.config.maximum_retreat_m + 0.5 * self.config.retreat_step_m,
            self.config.retreat_step_m,
        )
        for frame, row in enumerate(output):
            seed = row.copy()
            self._set_frame(row, poses[frame, 0])
            before = self._minimum_distance()
            worst_before = min(worst_before, before)
            if before < threshold:
                report.violating_frames_before += 1
                site_origin = self.data.site_xpos[self.target_site_id].copy()
                direction = site_origin - self.data.geom_xpos[self.scene_geom_id]
                norm = float(np.linalg.norm(direction))
                if norm > 1.0e-10:
                    direction /= norm
                    best = seed
                    best_distance = before
                    for retreat in retreat_distances:
                        candidate = self._solve_site_target(
                            seed,
                            poses[frame, 0],
                            site_origin + retreat * direction,
                        )
                        self._set_frame(candidate, poses[frame, 0])
                        distance = self._minimum_distance()
                        if distance > best_distance:
                            best = candidate
                            best_distance = distance
                        if distance >= threshold:
                            break
                    row[:] = best
            self._set_frame(row, poses[frame, 0])
            after = self._minimum_distance()
            worst_after = min(worst_after, after)
            if after < threshold:
                report.violating_frames_after += 1
                report.residual_frames.append(frame)
            elif before < threshold:
                report.corrected_frames += 1
            shifts.extend(np.abs(row - seed))

        report.worst_clearance_before_m = float(worst_before)
        report.worst_clearance_after_m = float(worst_after)
        if shifts:
            report.mean_joint_shift = float(np.mean(shifts))
            report.max_joint_shift = float(np.max(shifts))
        return output, report


class MujocoSceneCollisionClosureProjector:
    """Retain maximal finger closure without penetrating a moving object."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        trajectory_joint_names: Sequence[str],
        closure_joint_names: Sequence[str],
        robot_geom_names: Sequence[str | int],
        scene_geom_name: str | int,
        scene_mocap_body_name: str,
        open_joint_positions: Sequence[float] | None = None,
        config: SceneCollisionClosureProjectionConfig | None = None,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.config = config or SceneCollisionClosureProjectionConfig()
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
                _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.trajectory_joint_names
            ],
            dtype=np.int32,
        )
        self.qpos_addresses = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int32)
        closure_joint_ids = joint_ids[self.closure_columns]
        lower = np.asarray(model.jnt_range[closure_joint_ids, 0], dtype=np.float64)
        upper = np.asarray(model.jnt_range[closure_joint_ids, 1], dtype=np.float64)
        if open_joint_positions is None:
            self.open_positions = np.clip(np.zeros(len(lower)), lower, upper)
        else:
            open_values = np.asarray(open_joint_positions, dtype=np.float64)
            if open_values.shape != lower.shape or not np.isfinite(open_values).all():
                raise ValueError("open_joint_positions must align with closure joints.")
            self.open_positions = np.clip(open_values, lower, upper)
        self.robot_geom_ids = tuple(
            _object_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in robot_geom_names
        )
        if not self.robot_geom_ids:
            raise ValueError("robot_geom_names must be non-empty.")
        self.scene_geom_id = _object_id(
            model, mujoco.mjtObj.mjOBJ_GEOM, scene_geom_name
        )
        body_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, scene_mocap_body_name)
        self.scene_mocap_id = int(model.body_mocapid[body_id])
        if self.scene_mocap_id < 0:
            raise ValueError(f"Body {scene_mocap_body_name!r} is not a mocap body.")
        self._query_cutoff = self.config.clearance_m + self.config.query_margin_m

    def _minimum_distance(self, row: np.ndarray, pose: np.ndarray) -> float:
        self.data.qpos[self.qpos_addresses] = row
        self.data.mocap_pos[self.scene_mocap_id] = pose[:3]
        self.data.mocap_quat[self.scene_mocap_id] = pose[3:7]
        mujoco.mj_forward(self.model, self.data)
        segment = np.zeros(6, dtype=np.float64)
        return min(
            float(
                mujoco.mj_geomDistance(
                    self.model,
                    self.data,
                    geom_id,
                    self.scene_geom_id,
                    self._query_cutoff,
                    segment,
                )
            )
            for geom_id in self.robot_geom_ids
        )

    def project(
        self,
        qpos: np.ndarray,
        *,
        scene_mocap_poses: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, CollisionProjectionReport]:
        """Return qpos, retained closure scales, and a fail-closed report."""

        values = np.asarray(qpos, dtype=np.float64)
        poses = np.asarray(scene_mocap_poses, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.trajectory_joint_names):
            raise ValueError("qpos must have shape [T, trajectory joints].")
        if poses.shape != (len(values), 1, 7):
            raise ValueError("scene_mocap_poses must have shape [T, 1, 7].")
        if not np.isfinite(values).all() or not np.isfinite(poses).all():
            raise ValueError("qpos/scene_mocap_poses contain non-finite values.")
        if not np.allclose(
            np.linalg.norm(poses[..., 3:7], axis=-1),
            1.0,
            rtol=0.0,
            atol=1.0e-5,
        ):
            raise ValueError("scene_mocap_poses contains non-unit quaternions.")

        output = values.copy()
        scales = np.ones(len(output), dtype=np.float64)
        report = CollisionProjectionReport(frame_count=len(output))
        shifts: list[float] = []
        worst_before = self._query_cutoff
        worst_after = self._query_cutoff
        threshold = self.config.clearance_m - self.config.tolerance_m
        for frame, row in enumerate(output):
            target = row.copy()
            before = self._minimum_distance(target, poses[frame, 0])
            worst_before = min(worst_before, before)
            if before < threshold:
                report.violating_frames_before += 1
                open_row = target.copy()
                open_row[self.closure_columns] = self.open_positions
                open_distance = self._minimum_distance(open_row, poses[frame, 0])
                if open_distance >= threshold:
                    low = 0.0
                    high = 1.0
                    for _ in range(self.config.search_iterations):
                        middle = 0.5 * (low + high)
                        candidate = open_row.copy()
                        candidate[self.closure_columns] = (
                            self.open_positions
                            + middle
                            * (target[self.closure_columns] - self.open_positions)
                        )
                        if (
                            self._minimum_distance(candidate, poses[frame, 0])
                            < threshold
                        ):
                            high = middle
                        else:
                            low = middle
                    row[:] = open_row
                    row[self.closure_columns] = self.open_positions + low * (
                        target[self.closure_columns] - self.open_positions
                    )
                    scales[frame] = low
                elif open_distance > before:
                    row[:] = open_row
                    scales[frame] = 0.0
            after = self._minimum_distance(row, poses[frame, 0])
            worst_after = min(worst_after, after)
            if after < threshold:
                report.violating_frames_after += 1
                report.residual_frames.append(frame)
            elif before < threshold:
                report.corrected_frames += 1
            shifts.extend(np.abs(row - target))
        report.worst_clearance_before_m = float(worst_before)
        report.worst_clearance_after_m = float(worst_after)
        if shifts:
            report.mean_joint_shift = float(np.mean(shifts))
            report.max_joint_shift = float(np.max(shifts))
        return output, scales, report


__all__ = [
    "CollisionProjectionConfig",
    "CollisionProjectionReport",
    "MujocoCollisionProjector",
    "MujocoRadialEscapeProjector",
    "MujocoSceneCollisionClosureProjector",
    "RadialEscapeProjectionConfig",
    "SceneCollisionClosureProjectionConfig",
]
