"""Recover exact contact geometry from a retargeted trajectory.

A SOMA-style source records *when* each hand touches an object, as one binary
flag per hand per frame, but not *where*. The training contract needs exact
geometry: a point on the robot link, a point on the object, and a unit normal
for each. This module recovers that geometry from the solved kinematics
instead of inventing it.

For each frame the recovered contact comes from a MuJoCo threshold distance
query between one fingertip geom and one object geom. The query returns the
signed distance and the witness segment, so the contact point on each body is
measured, not assumed. The normals follow the separation axis: the link normal
points from the link witness point toward the object, and the object normal
points back.

The module is fail-closed in both directions.

* A frame the source marks active, where no fingertip reaches the object
  within ``contact_tolerance_m``, stays **inactive**. It is counted as
  unreached and reported. A binary flag is never sufficient evidence of
  contact on its own.
* A fingertip that penetrates the object deeper than ``max_penetration_m``
  raises. Penetration is a retargeting defect, and emitting it as contact
  would certify a physically impossible reference.

The result is therefore a lower bound on the true contact set. It never
reports a contact that the geometry does not support.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import mujoco as _mujoco
import numpy as np

from iltools.core.dexterous_reference import ContactSequence

mujoco: Any = _mujoco


@dataclass(frozen=True)
class ContactRecoveryConfig:
    """Thresholds for turning a distance query into a contact record."""

    contact_tolerance_m: float = 0.002
    """Separation at or below which a fingertip counts as touching."""

    max_penetration_m: float = 0.002
    """Penetration deeper than this is a defect and stops recovery."""

    query_cutoff_m: float = 0.05
    """``mj_geomDistance`` threshold. Pairs beyond it return the cutoff."""

    minimum_normal_segment_m: float = 1.0e-9
    """Witness segments shorter than this cannot define a direction."""

    def __post_init__(self) -> None:
        for name in (
            "contact_tolerance_m",
            "max_penetration_m",
            "query_cutoff_m",
            "minimum_normal_segment_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.query_cutoff_m <= self.contact_tolerance_m:
            raise ValueError(
                "query_cutoff_m must exceed contact_tolerance_m, otherwise a "
                "touching pair cannot be told apart from a distant one."
            )


@dataclass
class ContactRecoveryReport:
    """What the recovery measured, including everything it refused."""

    frame_count: int = 0
    source_active_side_frames: int = 0
    recovered_side_frames: int = 0
    recovered_contacts: int = 0
    unreached_side_frames: int = 0
    degenerate_normal_slots: int = 0
    minimum_distance_m: float = float("inf")
    per_side_recovered: dict[str, int] = field(default_factory=dict)
    per_side_unreached: dict[str, int] = field(default_factory=dict)
    unreached_frames: dict[str, list[int]] = field(default_factory=dict)

    @property
    def recovery_rate(self) -> float:
        """Fraction of source-active hand frames that produced geometry."""

        if self.source_active_side_frames == 0:
            return 0.0
        return self.recovered_side_frames / self.source_active_side_frames

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "source_active_side_frames": self.source_active_side_frames,
            "recovered_side_frames": self.recovered_side_frames,
            "recovered_contacts": self.recovered_contacts,
            "unreached_side_frames": self.unreached_side_frames,
            "degenerate_normal_slots": self.degenerate_normal_slots,
            "minimum_distance_m": (
                None
                if not np.isfinite(self.minimum_distance_m)
                else float(self.minimum_distance_m)
            ),
            "recovery_rate": self.recovery_rate,
            "per_side_recovered": dict(self.per_side_recovered),
            "per_side_unreached": dict(self.per_side_unreached),
            "unreached_frames": {
                side: list(frames) for side, frames in self.unreached_frames.items()
            },
        }


def _geom_id(model: Any, name: str | int) -> int:
    if isinstance(name, (int, np.integer)):
        geom_id = int(name)
        if not 0 <= geom_id < int(model.ngeom):
            raise ValueError(f"The model has no geom id {geom_id}.")
        return geom_id
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id < 0:
        raise ValueError(f"The model has no geom named {name!r}.")
    return int(geom_id)


def recover_contact_sequence(
    model: Any,
    data: Any,
    *,
    qpos: np.ndarray,
    qpos_addresses: Sequence[int],
    object_geom_names: Sequence[str | int],
    fingertip_geom_names: Mapping[str, Sequence[str | int]],
    contact_link_names: Mapping[str, Sequence[str]],
    source_active: np.ndarray,
    object_mocap_poses: np.ndarray | None = None,
    object_mocap_body_names: Sequence[str] | None = None,
    config: ContactRecoveryConfig | None = None,
) -> tuple[ContactSequence, ContactRecoveryReport]:
    """Measure contact geometry for a retargeted trajectory.

    Args:
        model: A MuJoCo model holding the robot and every tracked object.
        data: Matching MuJoCo data.
        qpos: Retargeted joint trajectory with shape ``[T, J]``.
        qpos_addresses: ``model.qpos`` address for each column of ``qpos``.
        object_geom_names: One collision geom name for each tracked object.
        fingertip_geom_names: Per side, the geom name of each contact slot.
        contact_link_names: Per side, the link name of each contact slot.
        source_active: Source per-hand activity with shape ``[T, S]``.
        object_mocap_poses: Object poses ``[T, B, 7]`` as XYZ+WXYZ, if the
            objects are mocap bodies that move over the trajectory.
        object_mocap_body_names: Mocap body name for each tracked object.
        config: Recovery thresholds.

    Returns:
        The recovered :class:`ContactSequence` and a report of what it
        refused. Slots stay inactive with zero geometry where no contact was
        measured.
    """

    settings = config or ContactRecoveryConfig()
    sides = tuple(str(side) for side in fingertip_geom_names)
    if not sides:
        raise ValueError("fingertip_geom_names must name at least one hand side.")
    if tuple(contact_link_names) != sides:
        raise ValueError("contact_link_names must use the same sides, in order.")

    joint_trajectory = np.asarray(qpos, dtype=np.float64)
    if joint_trajectory.ndim != 2:
        raise ValueError("qpos must have shape [frames, joints].")
    frame_count = int(joint_trajectory.shape[0])
    addresses = np.asarray(qpos_addresses, dtype=np.int32)
    if addresses.shape[0] != joint_trajectory.shape[1]:
        raise ValueError("qpos_addresses must have one entry for each qpos column.")

    activity = np.asarray(source_active)
    if activity.shape != (frame_count, len(sides)):
        raise ValueError("source_active must have shape [frames, hand sides].")
    activity = activity.astype(bool)

    slot_count = max(len(fingertip_geom_names[side]) for side in sides)
    for side in sides:
        if len(fingertip_geom_names[side]) != len(contact_link_names[side]):
            raise ValueError(f"{side} geom and link name counts differ.")

    object_geoms = [_geom_id(model, name) for name in object_geom_names]
    if not object_geoms:
        raise ValueError("object_geom_names must name at least one object geom.")
    side_geoms = {
        side: [_geom_id(model, name) for name in fingertip_geom_names[side]]
        for side in sides
    }

    mocap_indices: list[int] = []
    if object_mocap_poses is not None:
        poses = np.asarray(object_mocap_poses, dtype=np.float64)
        if poses.shape != (frame_count, len(object_geoms), 7):
            raise ValueError("object_mocap_poses must have shape [T, objects, 7].")
        if object_mocap_body_names is None or len(object_mocap_body_names) != len(
            object_geoms
        ):
            raise ValueError(
                "object_mocap_body_names must name one body for each object."
            )
        for name in object_mocap_body_names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name))
            if body_id < 0:
                raise ValueError(f"The model has no body named {str(name)!r}.")
            mocap_id = int(model.body_mocapid[body_id])
            if mocap_id < 0:
                raise ValueError(f"Body {str(name)!r} is not a mocap body.")
            mocap_indices.append(mocap_id)
    else:
        poses = None

    shape = (frame_count, len(sides), slot_count)
    link_positions = np.zeros((*shape, 3), dtype=np.float32)
    link_normals = np.zeros((*shape, 3), dtype=np.float32)
    object_positions = np.zeros((*shape, 3), dtype=np.float32)
    object_normals = np.zeros((*shape, 3), dtype=np.float32)
    object_indices = np.full(shape, -1, dtype=np.int32)
    active = np.zeros(shape, dtype=np.bool_)

    names = np.zeros((len(sides), slot_count), dtype=object)
    for side_index, side in enumerate(sides):
        for slot, link in enumerate(contact_link_names[side]):
            names[side_index, slot] = str(link)
        for slot in range(len(contact_link_names[side]), slot_count):
            names[side_index, slot] = ""

    report = ContactRecoveryReport(frame_count=frame_count)
    report.per_side_recovered = {side: 0 for side in sides}
    report.per_side_unreached = {side: 0 for side in sides}
    report.unreached_frames = {side: [] for side in sides}

    fromto = np.zeros(6, dtype=np.float64)
    for frame in range(frame_count):
        data.qpos[addresses] = joint_trajectory[frame]
        if poses is not None:
            for body_index, mocap_id in enumerate(mocap_indices):
                data.mocap_pos[mocap_id] = poses[frame, body_index, :3]
                data.mocap_quat[mocap_id] = poses[frame, body_index, 3:7]
        mujoco.mj_forward(model, data)

        for side_index, side in enumerate(sides):
            if not bool(activity[frame, side_index]):
                continue
            report.source_active_side_frames += 1
            recovered_here = 0
            for slot, geom in enumerate(side_geoms[side]):
                best: tuple[float, int, np.ndarray] | None = None
                for object_index, object_geom in enumerate(object_geoms):
                    distance = float(
                        mujoco.mj_geomDistance(
                            model,
                            data,
                            geom,
                            object_geom,
                            settings.query_cutoff_m,
                            fromto,
                        )
                    )
                    if distance < -settings.max_penetration_m:
                        raise ValueError(
                            f"Fingertip {contact_link_names[side][slot]!r} "
                            f"penetrates object {object_index} by "
                            f"{-distance:.5f} m at frame {frame}, deeper than "
                            f"the {settings.max_penetration_m:.5f} m limit. "
                            "A penetrating trajectory cannot certify contact."
                        )
                    if best is None or distance < best[0]:
                        best = (distance, object_index, fromto.copy())
                assert best is not None
                distance, object_index, segment = best
                report.minimum_distance_m = min(report.minimum_distance_m, distance)
                if distance > settings.contact_tolerance_m:
                    continue

                direction = segment[3:] - segment[:3]
                length = float(np.linalg.norm(direction))
                if length < settings.minimum_normal_segment_m:
                    # Exactly coincident witness points give no direction. Fall
                    # back to the geom-centre axis, which stays well defined
                    # for a fingertip against a much larger object.
                    direction = np.asarray(
                        data.geom_xpos[object_geoms[object_index]]
                        - data.geom_xpos[geom],
                        dtype=np.float64,
                    )
                    length = float(np.linalg.norm(direction))
                    report.degenerate_normal_slots += 1
                    if length < settings.minimum_normal_segment_m:
                        raise ValueError(
                            "Cannot define a contact normal at frame "
                            f"{frame} for {contact_link_names[side][slot]!r}: "
                            "the fingertip and object centres coincide."
                        )
                unit = direction / length

                link_positions[frame, side_index, slot] = segment[:3]
                object_positions[frame, side_index, slot] = segment[3:]
                link_normals[frame, side_index, slot] = unit
                object_normals[frame, side_index, slot] = -unit
                object_indices[frame, side_index, slot] = object_index
                active[frame, side_index, slot] = True
                recovered_here += 1

            if recovered_here:
                report.recovered_side_frames += 1
                report.recovered_contacts += recovered_here
                report.per_side_recovered[side] += 1
            else:
                report.unreached_side_frames += 1
                report.per_side_unreached[side] += 1
                report.unreached_frames[side].append(frame)

    sequence = ContactSequence(
        hand_sides=sides,
        link_names=names.astype(np.str_),
        link_positions_w=link_positions,
        link_normals_w=link_normals,
        object_positions_w=object_positions,
        object_normals_w=object_normals,
        object_indices=object_indices,
        active=active,
    )
    return sequence, report


__all__ = [
    "ContactRecoveryConfig",
    "ContactRecoveryReport",
    "recover_contact_sequence",
]
