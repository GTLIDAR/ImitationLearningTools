"""Dependency-light human keypoint to robot joint retargeting.

This module provides the first retargeting stage for Dexterous Manipulation:
it turns named 3-D human landmarks into bounded robot joint targets.  It does
not claim to replace OmniRetarget's mesh-constrained optimisation.  The output
is a stable, named trajectory that can be passed to a later robot-model IK
stage or used for an initial hand-only tracking environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from iltools.core.trajectory import Trajectory


def _copy_optional_array(value: np.ndarray | None) -> np.ndarray | None:
    return None if value is None else np.array(value, copy=True)


def transform_keypoint_trajectory(
    trajectory: Trajectory,
    *,
    rotation: np.ndarray,
    translation: np.ndarray,
    scale: float = 1.0,
) -> Trajectory:
    """Apply an explicit world-to-robot transform to 3-D keypoints."""

    if "keypoints" not in trajectory.observations:
        raise KeyError("trajectory observations must contain 'keypoints'.")
    rotation_array = np.asarray(rotation, dtype=np.float64)
    translation_array = np.asarray(translation, dtype=np.float64)
    if rotation_array.shape != (3, 3):
        raise ValueError("rotation must have shape [3, 3].")
    if translation_array.shape != (3,):
        raise ValueError("translation must have shape [3].")
    if (
        not np.isfinite(rotation_array).all()
        or not np.isfinite(translation_array).all()
    ):
        raise ValueError("rotation and translation must be finite.")
    if not np.allclose(rotation_array.T @ rotation_array, np.eye(3), atol=1.0e-5):
        raise ValueError("rotation must be orthonormal.")
    if np.linalg.det(rotation_array) <= 0.0:
        raise ValueError("rotation must preserve handedness.")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive.")

    points = np.asarray(trajectory.observations["keypoints"], dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("keypoints must have shape [frames, keypoints, 3].")
    observations = {
        key: np.array(value, copy=True)
        for key, value in trajectory.observations.items()
    }
    observations["keypoints"] = (points @ rotation_array.T) * float(
        scale
    ) + translation_array
    infos = dict(trajectory.infos or {})
    infos["coordinate_frame"] = "robot"
    infos["keypoint_transform"] = {
        "rotation": rotation_array.tolist(),
        "translation": translation_array.tolist(),
        "scale": float(scale),
    }
    return Trajectory(
        observations=observations,
        actions=(
            None
            if trajectory.actions is None
            else {
                key: np.array(value, copy=True)
                for key, value in trajectory.actions.items()
            }
        ),
        rewards=_copy_optional_array(trajectory.rewards),
        infos=infos,
        dt=trajectory.dt,
    )


@dataclass(frozen=True, slots=True)
class JointMapSpec:
    """Map one named source joint angle to one bounded target joint."""

    target_name: str
    source_name: str
    lower: float
    upper: float
    scale: float = 1.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        if not self.target_name or not self.source_name:
            raise ValueError("Joint map names must be non-empty.")
        if not np.isfinite((self.lower, self.upper, self.scale, self.offset)).all():
            raise ValueError("Joint map parameters must be finite.")
        if self.lower > self.upper:
            raise ValueError("Joint map lower bound must not exceed upper bound.")


@dataclass(frozen=True, slots=True)
class KeypointJointSpec:
    """Derive a target hinge angle from three named 3-D landmarks.

    The angle is measured between ``parent - joint`` and ``child - joint``.
    ``scale`` and ``offset`` adapt the human hinge angle to a robot joint, and
    the result is clipped to the robot joint limits.  This is intentionally
    explicit: a hand model can define one spec per actuator without relying on
    an implicit joint-order assumption.
    """

    target_name: str
    parent: str
    joint: str
    child: str
    lower: float
    upper: float
    scale: float = 1.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        if not self.target_name or not self.parent or not self.joint or not self.child:
            raise ValueError("Keypoint joint names must be non-empty.")
        if len({self.parent, self.joint, self.child}) != 3:
            raise ValueError("A keypoint joint spec needs three distinct landmarks.")
        if not np.isfinite((self.lower, self.upper, self.scale, self.offset)).all():
            raise ValueError("Keypoint retarget parameters must be finite.")
        if self.lower > self.upper:
            raise ValueError("Keypoint lower bound must not exceed upper bound.")


class JointMapRetargeter:
    """Retarget named joint angles with scale, offset, and limit clipping."""

    def __init__(
        self,
        source_joint_names: Sequence[str],
        target_joint_names: Sequence[str],
        specs: Sequence[JointMapSpec],
        *,
        default_position: float = 0.0,
        source_key: str = "joint_position",
        target_key: str = "qpos",
    ) -> None:
        self.source_joint_names = tuple(source_joint_names)
        self.target_joint_names = tuple(target_joint_names)
        self.specs = tuple(specs)
        self.source_key = source_key
        self.target_key = target_key
        self.default_position = float(default_position)
        self._validate_names()
        self._source_index = {name: i for i, name in enumerate(self.source_joint_names)}
        self._target_index = {name: i for i, name in enumerate(self.target_joint_names)}

    def _validate_names(self) -> None:
        if len(set(self.source_joint_names)) != len(self.source_joint_names):
            raise ValueError("source_joint_names must be unique.")
        if len(set(self.target_joint_names)) != len(self.target_joint_names):
            raise ValueError("target_joint_names must be unique.")
        seen: set[str] = set()
        for spec in self.specs:
            if spec.source_name not in self.source_joint_names:
                raise KeyError(f"Unknown source joint in map: {spec.source_name!r}.")
            if spec.target_name not in self.target_joint_names:
                raise KeyError(f"Unknown target joint in map: {spec.target_name!r}.")
            if spec.target_name in seen:
                raise ValueError(
                    f"Target joint is mapped more than once: {spec.target_name!r}."
                )
            seen.add(spec.target_name)

    def retarget(self, trajectory: Trajectory) -> Trajectory:
        source = np.asarray(trajectory.observations[self.source_key], dtype=np.float64)
        expected = (len(source), len(self.source_joint_names))
        if source.shape != expected:
            raise ValueError(
                f"{self.source_key!r} must have shape {expected}, got {source.shape}."
            )
        if not np.isfinite(source).all():
            raise ValueError(f"{self.source_key!r} contains non-finite values.")

        qpos = np.full(
            (len(source), len(self.target_joint_names)),
            self.default_position,
            dtype=np.float64,
        )
        for spec in self.specs:
            source_values = source[:, self._source_index[spec.source_name]]
            values = np.clip(
                spec.scale * source_values + spec.offset,
                spec.lower,
                spec.upper,
            )
            qpos[:, self._target_index[spec.target_name]] = values
        return _retargeted_trajectory(
            trajectory,
            qpos,
            target_key=self.target_key,
            target_joint_names=self.target_joint_names,
            method="joint_map",
        )


class KeypointRetargeter:
    """Retarget named 3-D landmarks to bounded robot hinge positions."""

    def __init__(
        self,
        source_keypoint_names: Sequence[str],
        specs: Sequence[KeypointJointSpec],
        *,
        source_key: str = "keypoints",
        target_key: str = "qpos",
    ) -> None:
        self.source_keypoint_names = tuple(source_keypoint_names)
        self.specs = tuple(specs)
        self.source_key = source_key
        self.target_key = target_key
        if len(set(self.source_keypoint_names)) != len(self.source_keypoint_names):
            raise ValueError("source_keypoint_names must be unique.")
        if not self.source_keypoint_names:
            raise ValueError("At least one source keypoint is required.")
        if not self.specs:
            raise ValueError("At least one target joint spec is required.")
        seen_targets: set[str] = set()
        for spec in self.specs:
            if spec.target_name in seen_targets:
                raise ValueError(
                    f"Target joint is mapped more than once: {spec.target_name!r}."
                )
            seen_targets.add(spec.target_name)
            for name in (spec.parent, spec.joint, spec.child):
                if name not in self.source_keypoint_names:
                    raise KeyError(f"Unknown source keypoint in spec: {name!r}.")
        self._source_index = {
            name: i for i, name in enumerate(self.source_keypoint_names)
        }

    def retarget(self, trajectory: Trajectory) -> Trajectory:
        points = np.asarray(trajectory.observations[self.source_key], dtype=np.float64)
        expected = (len(points), len(self.source_keypoint_names), 3)
        if points.shape != expected:
            raise ValueError(
                f"{self.source_key!r} must have shape {expected}, got {points.shape}."
            )
        if not np.isfinite(points).all():
            raise ValueError(f"{self.source_key!r} contains non-finite values.")

        qpos = np.empty((len(points), len(self.specs)), dtype=np.float64)
        for target_index, spec in enumerate(self.specs):
            parent = points[:, self._source_index[spec.parent]]
            joint = points[:, self._source_index[spec.joint]]
            child = points[:, self._source_index[spec.child]]
            first = parent - joint
            second = child - joint
            first_norm = np.linalg.norm(first, axis=-1)
            second_norm = np.linalg.norm(second, axis=-1)
            if np.any(first_norm <= 1.0e-8) or np.any(second_norm <= 1.0e-8):
                raise ValueError(
                    "Keypoint retargeting found a zero-length landmark segment "
                    f"for target joint {spec.target_name!r}."
                )
            cosine = np.sum(first * second, axis=-1) / (first_norm * second_norm)
            angle = np.arccos(np.clip(cosine, -1.0, 1.0))
            qpos[:, target_index] = np.clip(
                spec.scale * angle + spec.offset,
                spec.lower,
                spec.upper,
            )

        return _retargeted_trajectory(
            trajectory,
            qpos,
            target_key=self.target_key,
            target_joint_names=tuple(spec.target_name for spec in self.specs),
            method="keypoint_angle",
        )


def _retargeted_trajectory(
    source: Trajectory,
    qpos: np.ndarray,
    *,
    target_key: str,
    target_joint_names: Sequence[str],
    method: str,
) -> Trajectory:
    observations = {
        key: np.array(value, copy=True) for key, value in source.observations.items()
    }
    observations[target_key] = qpos
    if source.dt is not None:
        if source.dt <= 0.0:
            raise ValueError("Trajectory dt must be positive when deriving qvel.")
        qvel = np.zeros_like(qpos)
        if len(qpos) > 1:
            qvel[1:] = np.diff(qpos, axis=0) / float(source.dt)
            qvel[0] = qvel[1]
        observations["qvel"] = qvel

    infos = dict(source.infos or {})
    infos["retarget"] = {
        "method": method,
        "target_joint_names": list(target_joint_names),
        "source_dt": source.dt,
    }
    return Trajectory(
        observations=observations,
        actions=(
            None
            if source.actions is None
            else {
                key: np.array(value, copy=True) for key, value in source.actions.items()
            }
        ),
        rewards=_copy_optional_array(source.rewards),
        infos=infos,
        dt=source.dt,
    )


def save_joint_reference_npz(
    trajectory: Trajectory,
    output_path: str | Path,
    *,
    joint_names: Sequence[str],
    fps: float,
) -> Path:
    """Save a named joint trajectory for the Vega/Wuji tracking task."""

    output = Path(output_path)
    names = tuple(joint_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("joint_names must be non-empty and unique.")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive.")

    if "qpos" not in trajectory.observations:
        raise KeyError("trajectory observations must contain 'qpos'.")
    qpos = np.asarray(trajectory.observations["qpos"], dtype=np.float64)
    expected = (len(qpos), len(names))
    if qpos.shape != expected:
        raise ValueError(f"qpos must have shape {expected}, got {qpos.shape}.")
    if not np.isfinite(qpos).all():
        raise ValueError("qpos contains non-finite values.")

    qvel_value = trajectory.observations.get("qvel")
    if qvel_value is None:
        qvel = np.zeros_like(qpos)
        if len(qpos) > 1:
            qvel[1:] = np.diff(qpos, axis=0) * float(fps)
            qvel[0] = qvel[1]
    else:
        qvel = np.asarray(qvel_value, dtype=np.float64)
        if qvel.shape != expected:
            raise ValueError(f"qvel must have shape {expected}, got {qvel.shape}.")
        if not np.isfinite(qvel).all():
            raise ValueError("qvel contains non-finite values.")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        qpos=qpos,
        qvel=qvel,
        joint_names=np.asarray(names),
        fps=np.asarray(float(fps), dtype=np.float32),
    )
    return output.resolve()


def merge_joint_trajectories(
    trajectories: Sequence[Trajectory],
    *,
    joint_names: Sequence[str] | None = None,
    default_position: float = 0.0,
) -> Trajectory:
    """Merge arm and hand retarget outputs into one named joint trajectory."""

    sources = tuple(trajectories)
    if not sources:
        raise ValueError("At least one joint trajectory is required.")
    if not np.isfinite(default_position):
        raise ValueError("default_position must be finite.")

    source_names: list[tuple[str, ...]] = []
    for trajectory in sources:
        retarget_info = (trajectory.infos or {}).get("retarget", {})
        names = tuple(retarget_info.get("target_joint_names", ()))
        if not names or len(set(names)) != len(names):
            raise ValueError(
                "Each trajectory must record unique retarget target_joint_names."
            )
        if "qpos" not in trajectory.observations:
            raise KeyError("Each trajectory must contain qpos observations.")
        qpos = np.asarray(trajectory.observations["qpos"])
        if qpos.ndim != 2 or qpos.shape[1] != len(names):
            raise ValueError("Each qpos array must match its target joint names.")
        source_names.append(names)

    frame_count = len(np.asarray(sources[0].observations["qpos"]))
    dt = sources[0].dt
    if any(
        len(np.asarray(item.observations["qpos"])) != frame_count for item in sources
    ):
        raise ValueError("Joint trajectories must have the same frame count.")
    if any(item.dt != dt for item in sources):
        raise ValueError("Joint trajectories must use the same dt.")

    output_names = tuple(
        joint_names or (name for names in source_names for name in names)
    )
    if not output_names or len(set(output_names)) != len(output_names):
        raise ValueError("joint_names must be non-empty and unique.")
    output_index = {name: index for index, name in enumerate(output_names)}
    qpos = np.full(
        (frame_count, len(output_names)),
        float(default_position),
        dtype=np.float64,
    )
    written = np.zeros(len(output_names), dtype=bool)
    for trajectory, names in zip(sources, source_names, strict=True):
        values = np.asarray(trajectory.observations["qpos"], dtype=np.float64)
        for source_index, name in enumerate(names):
            if name not in output_index:
                raise KeyError(f"Joint {name!r} is missing from output joint_names.")
            target_index = output_index[name]
            if written[target_index] and not np.allclose(
                qpos[:, target_index], values[:, source_index]
            ):
                raise ValueError(f"Conflicting joint targets for {name!r}.")
            qpos[:, target_index] = values[:, source_index]
            written[target_index] = True
    if not np.isfinite(qpos).all():
        raise ValueError("Merged qpos contains non-finite values.")

    observations = {"qpos": qpos}
    if dt is not None:
        if dt <= 0.0:
            raise ValueError("Trajectory dt must be positive when deriving qvel.")
        qvel = np.zeros_like(qpos)
        if frame_count > 1:
            qvel[1:] = np.diff(qpos, axis=0) / float(dt)
            qvel[0] = qvel[1]
        observations["qvel"] = qvel
    return Trajectory(
        observations=observations,
        infos={
            "retarget": {
                "method": "joint_merge",
                "target_joint_names": list(output_names),
                "source_methods": [
                    (item.infos or {}).get("retarget", {}).get("method")
                    for item in sources
                ],
            }
        },
        dt=dt,
    )


__all__ = [
    "JointMapRetargeter",
    "JointMapSpec",
    "KeypointJointSpec",
    "KeypointRetargeter",
    "merge_joint_trajectories",
    "save_joint_reference_npz",
    "transform_keypoint_trajectory",
]
