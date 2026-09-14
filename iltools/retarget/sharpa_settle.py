"""Settle a retargeted Sharpa row onto the object surface.

The Pink retarget solves fingertip inverse kinematics. Nothing in that step
knows the object exists, so the fingers finish inside it: on the released
ARCTIC box-grab sequence they are 6.5 mm deep on average and 35 mm at worst,
and every penetrating link is a finger, never the palm.

Isaac has to answer that penetration on the first physics step. PhysX absorbs
it. Newton, which is MuJoCo Warp, answers with an impulse an order of
magnitude larger, throws both hands and the object off the Reference, and the
episode terminates before the policy acts.

This module presses the retarget onto the object in MuJoCo instead, following
DexMachina's functional retargeting, and hands back the achieved pose. The
wrist is a mocap body, so it has infinite mass and stays exactly on the
retargeted trajectory; only the fingers move. That keeps the wrist tracking
terms unchanged and confines the correction to the links that are actually
wrong.

Settling in MuJoCo is deliberate: Newton is MuJoCo Warp, so the contact model
that resolved the pose is the one that will simulate it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from iltools.retarget.convex_parts import (
    ConvexDecompositionConfig,
    convex_part_paths,
)
from iltools.retarget.contact_settling import (
    ContactSettlingConfig,
    ContactSettlingReport,
    settle_contact_trajectory,
)

WRIST_BODY = {"left": "left_root", "right": "right_root"}
OBJECT_BODY = "object"
OBJECT_GEOM_PREFIX = "object_geom"


@dataclass(frozen=True)
class SharpaSettleAssets:
    """Where the geometry for one settle comes from."""

    left_mjcf_path: Path
    right_mjcf_path: Path
    object_mesh_path: Path
    decomposition: ConvexDecompositionConfig | None = None


def build_settle_model(assets: SharpaSettleAssets) -> tuple[Any, Any]:
    """Return a MuJoCo model of both hands and the object, wrists on mocap.

    The object and both wrists are mocap bodies. A mocap body is kinematic, so
    the object never yields and the wrist never recoils; the only thing free to
    move is the finger chain, which is the thing the settle is meant to correct.
    """

    import mujoco

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=OBJECT_BODY, mocap=True)
    # One geom per convex part. A single mesh geom would be its convex hull,
    # which fills the cavity of a container and turns a hand reaching inside
    # into deep false penetration.
    parts = convex_part_paths(assets.object_mesh_path, config=assets.decomposition)
    for index, part in enumerate(parts):
        mesh_name = f"object_part_{index:03d}"
        spec.add_mesh(name=mesh_name, file=str(part))
        body.add_geom(
            name=f"{OBJECT_GEOM_PREFIX}_{index:03d}",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
        )
    for side, path in (
        ("left", assets.left_mjcf_path),
        ("right", assets.right_mjcf_path),
    ):
        child = mujoco.MjSpec.from_file(str(Path(path).resolve()))
        root = spec.worldbody.add_body(name=WRIST_BODY[side], mocap=True)
        # Both hand MJCFs share mesh names, so every attached name is prefixed.
        root.add_frame().attach_body(child.worldbody.first_body(), f"{side}_", "")
    model = spec.compile()
    return model, mujoco.MjData(model)


def object_geom_names(model: Any) -> tuple[str, ...]:
    """Return every collision geom that belongs to the object."""

    import mujoco

    object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY)
    return tuple(
        str(model.geom(geom_id).name)
        for geom_id in range(model.ngeom)
        if int(model.geom(geom_id).bodyid[0]) == object_body
        and model.geom(geom_id).name
    )


def robot_geom_names(model: Any) -> tuple[str, ...]:
    """Return every collision geom that belongs to a hand, not to the object."""

    import mujoco

    object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY)
    names = []
    for geom_id in range(model.ngeom):
        if int(model.geom(geom_id).bodyid[0]) == object_body:
            continue
        name = model.geom(geom_id).name
        if name:
            names.append(str(name))
    return tuple(names)


def settle_sharpa_row(
    row: Mapping[str, Any],
    *,
    assets: SharpaSettleAssets,
    config: ContactSettlingConfig | None = None,
    object_body_index: int = 0,
) -> tuple[dict[str, Any], ContactSettlingReport]:
    """Return the row with settled finger joints, and what the settle achieved.

    Only ``robot_<side>_finger_joints`` changes. Wrist poses are inputs to the
    settle, never outputs, so every wrist-referenced quantity in the row stays
    exactly as the retarget left it.
    """

    model, data = build_settle_model(assets)
    settled = dict(row)

    object_position = np.asarray(row["object_body_position"], dtype=np.float64)
    object_wxyz = np.asarray(row["object_body_wxyz"], dtype=np.float64)
    frame_count = int(object_position.shape[0])
    object_pose = np.concatenate(
        (
            object_position[:, object_body_index, :],
            object_wxyz[:, object_body_index, :],
        ),
        axis=-1,
    )

    joint_names: list[str] = []
    targets: list[np.ndarray] = []
    slices: dict[str, slice] = {}
    mocap: dict[str, np.ndarray] = {OBJECT_BODY: object_pose}
    for side in ("left", "right"):
        names = [str(name) for name in row[f"{side}_robot_finger_joint_names"]]
        angles = np.asarray(row[f"robot_{side}_finger_joints"], dtype=np.float64)
        if angles.shape != (frame_count, len(names)):
            raise ValueError(
                f"robot_{side}_finger_joints must have shape "
                f"[{frame_count}, {len(names)}], got {angles.shape}."
            )
        slices[side] = slice(len(joint_names), len(joint_names) + len(names))
        # Both hand MJCFs reuse the same mesh names, so each is attached under
        # a side prefix. The joint names already carry the side, so the model
        # name is the doubled form: left_left_thumb_CMC_FE.
        joint_names.extend(f"{side}_{name}" for name in names)
        targets.append(angles)
        mocap[WRIST_BODY[side]] = np.concatenate(
            (
                np.asarray(row[f"robot_{side}_wrist_position"], dtype=np.float64),
                np.asarray(row[f"robot_{side}_wrist_wxyz"], dtype=np.float64),
            ),
            axis=-1,
        )

    achieved, report = settle_contact_trajectory(
        model,
        data,
        qpos=np.concatenate(targets, axis=-1),
        joint_names=joint_names,
        object_geom_names=object_geom_names(model),
        robot_geom_names=robot_geom_names(model),
        mocap_poses=mocap,
        config=config,
    )
    for side in ("left", "right"):
        settled[f"robot_{side}_finger_joints"] = achieved[:, slices[side]].tolist()
    settled["retarget_contact_settled"] = True
    settled["retarget_contact_settle_report"] = report.as_dict()
    return settled, report


__all__ = [
    "OBJECT_BODY",
    "OBJECT_GEOM_PREFIX",
    "object_geom_names",
    "SharpaSettleAssets",
    "build_settle_model",
    "robot_geom_names",
    "settle_sharpa_row",
]
