"""Resolve retargeting penetration into surface contact by soft replay.

A kinematic retarget places the robot hand where the human hand was. Because
the robot hand is a different size and shape, it usually ends up overlapping
the object. There are two ways to answer that.

The conservative answer pushes the hand away until nothing overlaps. It is
safe, and it destroys the task: a hand held clear of the object can never
manipulate it, and no contact geometry can be recovered from it.

This module takes the other answer, following DexMachina's functional
retargeting: replay the retargeted joints as *soft* position targets in
simulation while the object is held fixed, and let contact push the hand out
of the object. The hand settles on the surface instead of inside it or far
from it. What comes back is the achieved joint trajectory, which is the
closest non-penetrating pose to the retarget rather than an arbitrary
standoff.

Gravity is off during settling. The goal is a geometric correction, and a
sagging arm would move the hand for reasons that have nothing to do with
contact.

The object never moves. It is driven by mocap, so the robot yields and the
object does not, which is what makes the result a statement about the robot's
pose rather than about the object's dynamics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import mujoco as _mujoco
import numpy as np

mujoco: Any = _mujoco


@dataclass(frozen=True)
class ContactSettlingConfig:
    """How hard and how long to press the retarget against the object."""

    settle_steps: int = 60
    """Physics steps for each frame. More steps means a deeper settle."""

    stiffness_scale: float = 1.0
    """Multiplier on the model's own position-actuator gains."""

    damping_scale: float = 1.0
    """Multiplier on the model's own actuator damping."""

    disable_gravity: bool = True
    """Keep gravity out of a purely geometric correction."""

    def __post_init__(self) -> None:
        if int(self.settle_steps) < 1:
            raise ValueError("settle_steps must be at least one.")
        for name in ("stiffness_scale", "damping_scale"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")


@dataclass
class ContactSettlingReport:
    """How far the settle moved the pose, and what penetration remains."""

    frame_count: int = 0
    worst_penetration_before_m: float = 0.0
    worst_penetration_after_m: float = 0.0
    frames_penetrating_before: int = 0
    frames_penetrating_after: int = 0
    mean_joint_shift_rad: float = 0.0
    max_joint_shift_rad: float = 0.0
    residual_penetration_frames: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "worst_penetration_before_m": self.worst_penetration_before_m,
            "worst_penetration_after_m": self.worst_penetration_after_m,
            "frames_penetrating_before": self.frames_penetrating_before,
            "frames_penetrating_after": self.frames_penetrating_after,
            "mean_joint_shift_rad": self.mean_joint_shift_rad,
            "max_joint_shift_rad": self.max_joint_shift_rad,
            "residual_penetration_frames": list(self.residual_penetration_frames),
        }


def _actuator_for_joints(model: Any, joint_names: Sequence[str]) -> list[int]:
    """Map each joint to the position actuator that drives it."""

    actuator_of_joint: dict[int, int] = {}
    for actuator in range(model.nu):
        transmission = int(model.actuator_trntype[actuator])
        if transmission != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        actuator_of_joint[int(model.actuator_trnid[actuator, 0])] = actuator

    actuators: list[int] = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
        if joint_id < 0:
            raise ValueError(f"The model has no joint named {str(name)!r}.")
        if int(joint_id) not in actuator_of_joint:
            raise ValueError(f"Joint {str(name)!r} has no position actuator.")
        actuators.append(actuator_of_joint[int(joint_id)])
    return actuators


def _worst_penetration(
    model: Any,
    data: Any,
    robot_geoms: Sequence[int],
    object_geoms: Sequence[int],
    cutoff: float,
) -> float:
    segment = np.zeros(6, dtype=np.float64)
    worst = cutoff
    for robot_geom in robot_geoms:
        for object_geom in object_geoms:
            distance = float(
                mujoco.mj_geomDistance(
                    model, data, int(robot_geom), int(object_geom), cutoff, segment
                )
            )
            worst = min(worst, distance)
    return worst


def settle_contact_trajectory(
    model: Any,
    data: Any,
    *,
    qpos: np.ndarray,
    joint_names: Sequence[str],
    object_geom_names: Sequence[str],
    robot_geom_names: Sequence[str],
    object_mocap_poses: np.ndarray | None = None,
    object_mocap_body_names: Sequence[str] | None = None,
    mocap_poses: Mapping[str, np.ndarray] | None = None,
    config: ContactSettlingConfig | None = None,
) -> tuple[np.ndarray, ContactSettlingReport]:
    """Press a retargeted trajectory onto the object and return what it achieves.

    Args:
        model: MuJoCo model holding the robot and every tracked object.
        data: Matching MuJoCo data.
        qpos: Retargeted joint trajectory with shape ``[T, J]``.
        joint_names: The joint driven by each column of ``qpos``.
        object_geom_names: Object collision geoms to resolve against.
        robot_geom_names: Robot geoms to measure penetration on.
        object_mocap_poses: Object poses ``[T, B, 7]`` as XYZ+WXYZ.
        object_mocap_body_names: Mocap body for each tracked object.
        mocap_poses: Any further mocap bodies to drive, as name to ``[T, 7]``
            XYZ+WXYZ poses. Use it to hold a floating hand's root: a mocap
            body has infinite mass, so the wrist stays exactly on the
            retarget while contact moves the fingers alone.
        config: Settling behaviour.

    Returns:
        The achieved joint trajectory with the same shape as ``qpos``, and a
        report comparing penetration before and after.
    """

    settings = config or ContactSettlingConfig()
    target = np.asarray(qpos, dtype=np.float64)
    if target.ndim != 2:
        raise ValueError("qpos must have shape [frames, joints].")
    if target.shape[1] != len(joint_names):
        raise ValueError("qpos columns and joint_names must correspond.")
    frame_count = int(target.shape[0])

    addresses = np.asarray(
        [
            int(
                model.jnt_qposadr[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
                ]
            )
            for name in joint_names
        ],
        dtype=np.int32,
    )
    actuators = np.asarray(_actuator_for_joints(model, joint_names), dtype=np.int32)

    def geom_ids(names: Sequence[str]) -> list[int]:
        resolved = []
        for name in names:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, str(name))
            if geom_id < 0:
                raise ValueError(f"The model has no geom named {str(name)!r}.")
            resolved.append(int(geom_id))
        return resolved

    object_geoms = geom_ids(object_geom_names)
    robot_geoms = geom_ids(robot_geom_names)

    mocap_ids: list[int] = []
    poses = None
    if object_mocap_poses is not None:
        poses = np.asarray(object_mocap_poses, dtype=np.float64)
        if poses.shape != (frame_count, len(object_geoms), 7):
            raise ValueError("object_mocap_poses must have shape [T, objects, 7].")
        if object_mocap_body_names is None or len(object_mocap_body_names) != len(
            object_geoms
        ):
            raise ValueError("object_mocap_body_names must name one body per object.")
        for name in object_mocap_body_names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name))
            if body_id < 0:
                raise ValueError(f"The model has no body named {str(name)!r}.")
            mocap_id = int(model.body_mocapid[body_id])
            if mocap_id < 0:
                raise ValueError(f"Body {str(name)!r} is not a mocap body.")
            mocap_ids.append(mocap_id)

    extra_mocap: list[tuple[int, np.ndarray]] = []
    for name, values in (mocap_poses or {}).items():
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (frame_count, 7):
            raise ValueError(
                f"mocap_poses[{name!r}] must have shape [{frame_count}, 7], "
                f"got {array.shape}."
            )
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name))
        if body_id < 0:
            raise ValueError(f"The model has no body named {str(name)!r}.")
        mocap_id = int(model.body_mocapid[body_id])
        if mocap_id < 0:
            raise ValueError(f"Body {str(name)!r} is not a mocap body.")
        extra_mocap.append((mocap_id, array))

    saved_gravity = np.array(model.opt.gravity, dtype=np.float64)
    saved_gain = np.array(model.actuator_gainprm, dtype=np.float64)
    saved_bias = np.array(model.actuator_biasprm, dtype=np.float64)
    if settings.disable_gravity:
        model.opt.gravity[:] = 0.0
    if settings.stiffness_scale != 1.0 or settings.damping_scale != 1.0:
        for actuator in actuators:
            model.actuator_gainprm[actuator, 0] *= settings.stiffness_scale
            # A MuJoCo position actuator stores -kp in biasprm[1] and -kv in
            # biasprm[2]; scale each with its matching term.
            model.actuator_biasprm[actuator, 1] *= settings.stiffness_scale
            model.actuator_biasprm[actuator, 2] *= settings.damping_scale

    achieved = np.empty_like(target)
    report = ContactSettlingReport(frame_count=frame_count)
    cutoff = 0.05
    shifts: list[float] = []
    try:
        for frame in range(frame_count):
            # Every frame starts at its own retarget, never at the previous
            # settled pose. Carrying the previous pose forward would let the
            # actuators lag behind a fast target and record that lag as if it
            # were a contact correction.
            mujoco.mj_resetData(model, data)
            data.qpos[addresses] = target[frame]
            data.qvel[:] = 0.0
            if poses is not None:
                for index, mocap_id in enumerate(mocap_ids):
                    data.mocap_pos[mocap_id] = poses[frame, index, :3]
                    data.mocap_quat[mocap_id] = poses[frame, index, 3:7]
            for mocap_id, array in extra_mocap:
                data.mocap_pos[mocap_id] = array[frame, :3]
                data.mocap_quat[mocap_id] = array[frame, 3:7]
            data.ctrl[actuators] = target[frame]

            mujoco.mj_forward(model, data)
            before = _worst_penetration(model, data, robot_geoms, object_geoms, cutoff)

            for _ in range(int(settings.settle_steps)):
                mujoco.mj_step(model, data)

            mujoco.mj_forward(model, data)
            after = _worst_penetration(model, data, robot_geoms, object_geoms, cutoff)

            achieved[frame] = data.qpos[addresses]
            shift = float(np.abs(achieved[frame] - target[frame]).max())
            shifts.append(shift)

            report.worst_penetration_before_m = min(
                report.worst_penetration_before_m, before
            )
            report.worst_penetration_after_m = min(
                report.worst_penetration_after_m, after
            )
            if before < 0.0:
                report.frames_penetrating_before += 1
            if after < 0.0:
                report.frames_penetrating_after += 1
                report.residual_penetration_frames.append(frame)
    finally:
        model.opt.gravity[:] = saved_gravity
        model.actuator_gainprm[:] = saved_gain
        model.actuator_biasprm[:] = saved_bias

    if shifts:
        report.mean_joint_shift_rad = float(np.mean(shifts))
        report.max_joint_shift_rad = float(np.max(shifts))
    return achieved, report


__all__ = [
    "ContactSettlingConfig",
    "ContactSettlingReport",
    "settle_contact_trajectory",
]
