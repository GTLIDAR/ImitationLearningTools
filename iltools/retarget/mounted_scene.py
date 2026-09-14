"""Recompute contact geometry and collision evidence on a mounted scene."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from iltools.core import ContactSequence, DexterousReference


def project_contact_constraints(
    model: mujoco.MjModel,
    reference: DexterousReference,
    *,
    penetration_tolerance_m: float = 0.0005,
    iterations: int = 100,
    neighbor_restarts: int = 2,
    variable_joint_names: Sequence[str] | None = None,
    joint_change_limits: Mapping[str, float] | None = None,
    initial_qpos: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Project actual penetrating pairs with their relative-point Jacobians.

    All scalar robot joints may move. Object and support poses stay fixed.
    Model contact filtering applies, including the vendor's adjacent-link
    exclusions. This is a geometric repair; the caller must recompute poses,
    velocities, source tracking error, and contact labels afterward.
    """
    data = mujoco.MjData(model)
    joints = [model.joint(n).id for n in reference.joint_names]
    addresses = model.jnt_qposadr[joints]
    variable_columns = np.asarray(
        [reference.joint_names.index(n) for n in variable_joint_names]
        if variable_joint_names is not None
        else list(range(len(joints)))
    )
    dofs = model.jnt_dofadr[np.asarray(joints)[variable_columns]]
    bounds = model.jnt_range[joints]
    object_id = model.body("object").id
    mocap = model.body_mocapid[object_id]
    if mocap < 0 or model.nq != len(joints):
        raise ValueError(
            "Contact projection needs a fixed scalar-joint robot and mocap object."
        )
    output = np.asarray(
        reference.qpos if initial_qpos is None else initial_qpos, dtype=np.float64
    ).copy()
    if output.shape != reference.qpos.shape or not np.isfinite(output).all():
        raise ValueError(
            "Projection seeds must match the finite Reference joint array."
        )
    if np.any(output < bounds[:, 0] - 1e-5) or np.any(output > bounds[:, 1] + 1e-5):
        raise ValueError("Source joint positions exceed the mounted model limits.")
    output = np.clip(output, bounds[:, 0], bounds[:, 1])
    before, after, counts = [], [], []
    jac_a, jac_b = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    jac_rot = np.zeros_like(jac_a)

    def set_row(row):
        data.qpos[addresses] = row
        mujoco.mj_forward(model, data)

    def objective():
        return sum(
            max(-float(c.dist) - penetration_tolerance_m, 0) ** 2
            for c in data.contact
            if model.geom_bodyid[c.geom1] != object_id
            or model.geom_bodyid[c.geom2] != object_id
        )

    def depth():
        return max([0.0] + [-float(c.dist) for c in data.contact])

    for t, row in enumerate(output):
        frame_bounds = bounds.copy()
        for name, limit in (joint_change_limits or {}).items():
            if not np.isfinite(limit) or limit <= 0:
                raise ValueError("Joint correction limits must be positive.")
            column = reference.joint_names.index(name)
            frame_bounds[column, 0] = max(
                frame_bounds[column, 0], reference.qpos[t, column] - limit
            )
            frame_bounds[column, 1] = min(
                frame_bounds[column, 1], reference.qpos[t, column] + limit
            )
        row[:] = np.clip(row, frame_bounds[:, 0], frame_bounds[:, 1])
        data.mocap_pos[mocap] = reference.object_poses_w[t, 0, :3]
        data.mocap_quat[mocap] = reference.object_poses_w[t, 0, 3:]
        set_row(row)
        before.append(depth())
        count = 0
        for count in range(iterations):
            value = objective()
            if value < 1e-12:
                break
            gradients, errors = [], []
            for contact in data.contact:
                error = -float(contact.dist) - penetration_tolerance_m
                if error <= 0:
                    continue
                body_a, body_b = (
                    int(model.geom_bodyid[contact.geom1]),
                    int(model.geom_bodyid[contact.geom2]),
                )
                mujoco.mj_jac(model, data, jac_a, jac_rot, contact.pos, body_a)
                mujoco.mj_jac(model, data, jac_b, jac_rot, contact.pos, body_b)
                gradient = contact.frame[:3] @ (jac_b - jac_a)[:, dofs]
                if np.linalg.norm(gradient) < 1e-10:
                    continue
                gradients.append(gradient)
                errors.append(error + 1e-5)
            if not gradients:
                break
            jac = np.asarray(gradients)
            # Bound the solve itself. Clipping an unconstrained solution can
            # discard all useful motion when several finger joints hit limits.
            step = lsq_linear(
                np.vstack((jac, 0.003 * np.eye(len(dofs)))),
                np.r_[errors, np.zeros(len(dofs))],
                bounds=(
                    np.maximum(
                        frame_bounds[variable_columns, 0] - row[variable_columns], -0.15
                    ),
                    np.minimum(
                        frame_bounds[variable_columns, 1] - row[variable_columns], 0.15
                    ),
                ),
                tol=1e-7,
                max_iter=50,
            ).x
            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
                candidate = row.copy()
                candidate[variable_columns] = np.clip(
                    row[variable_columns] + scale * step,
                    frame_bounds[variable_columns, 0],
                    frame_bounds[variable_columns, 1],
                )
                set_row(candidate)
                if objective() < value:
                    row[:] = candidate
                    accepted = True
                    break
            if not accepted:
                set_row(row)
                break
        after.append(depth())
        counts.append(count + 1)
    restarted = []
    for _ in range(neighbor_restarts):
        good = np.flatnonzero(np.asarray(after) <= penetration_tolerance_m + 1e-5)
        bad = np.flatnonzero(np.asarray(after) > penetration_tolerance_m + 1e-5)
        if not len(bad) or not len(good):
            break
        seeded = output.copy()
        for column in variable_columns:
            seeded[bad, column] = np.interp(bad, good, output[good, column])
        candidate, candidate_report = project_contact_constraints(
            model,
            reference,
            initial_qpos=seeded,
            penetration_tolerance_m=penetration_tolerance_m,
            iterations=iterations,
            neighbor_restarts=0,
            variable_joint_names=variable_joint_names,
            joint_change_limits=joint_change_limits,
        )
        for frame in bad:
            if candidate_report["penetration_after_m"][frame] < after[frame]:
                output[frame] = candidate[frame]
                after[frame] = candidate_report["penetration_after_m"][frame]
                restarted.append(int(frame))
    return output, {
        "method": "native_contact_relative_point_jacobian",
        "penetration_tolerance_m": penetration_tolerance_m,
        "maximum_penetration_before_m": max(before),
        "maximum_penetration_after_m": max(after),
        "penetration_after_m": after,
        "iterations": counts,
        "neighbor_seeded_frames": sorted(set(restarted)),
        "variable_joint_names": [reference.joint_names[i] for i in variable_columns],
        "joint_change_limits": dict(joint_change_limits or {}),
        "maximum_joint_change_rad": float(np.max(np.abs(output - reference.qpos))),
    }


def contact_geometry_and_clearance(
    model: mujoco.MjModel,
    reference: DexterousReference,
    contact_body_names: Mapping[str, Sequence[str]],
    *,
    object_body_name: str = "object",
    support_body_names: Sequence[str] = ("stand",),
    contact_distance_m: float = 0.01,
) -> tuple[ContactSequence, dict[str, np.ndarray]]:
    """Return robot contact witnesses and signed penetration depths per frame.

    The model must place its robot at ``fixed_root_pose_w`` and expose one
    mocap object. It must use the runtime's explicit convex object pieces.
    Positive gaps beyond the contact threshold are capped at that threshold;
    the penetration arrays report zero when no penetrating pair was found.
    """
    if reference.robot_layout != "fixed_base":
        raise ValueError("Mounted scene audit requires a fixed-base Reference.")
    sides = ("left", "right")
    names = np.asarray([contact_body_names[side] for side in sides])
    if names.ndim != 2 or names.shape[0] != 2:
        raise ValueError("Contact body lists must have the same length on both sides.")
    data = mujoco.MjData(model)
    addresses = [
        int(model.jnt_qposadr[model.joint(name).id]) for name in reference.joint_names
    ]
    object_id = model.body(object_body_name).id
    mocap_id = int(model.body_mocapid[object_id])
    if mocap_id < 0:
        raise ValueError("The audit object must be a mocap body.")
    object_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == object_id]
    support_ids = {model.body(name).id for name in support_body_names}
    support_geoms = {
        g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in support_ids
    }
    robot_geoms = {
        g
        for g in range(model.ngeom)
        if int(model.geom_bodyid[g]) not in support_ids | {object_id, 0}
        and (model.geom_contype[g] or model.geom_conaffinity[g])
    }
    geom_groups = []
    for side_names in names:
        groups = []
        for name in side_names:
            body_id = model.body(str(name)).id
            groups.append([g for g in robot_geoms if model.geom_bodyid[g] == body_id])
        if any(not group for group in groups):
            raise ValueError("Every contact body must own collision geometry.")
        geom_groups.append(groups)
    shape = (reference.frame_count, *names.shape)
    link_points = np.zeros((*shape, 3), dtype=np.float32)
    object_points = np.zeros_like(link_points)
    normals = np.zeros_like(link_points)
    active = np.zeros(shape, dtype=bool)
    distances = np.full(shape, contact_distance_m, dtype=np.float32)
    diagnostics = {
        key: np.zeros(reference.frame_count, dtype=np.float32)
        for key in (
            "self_penetration_m",
            "robot_support_penetration_m",
            "robot_object_penetration_m",
            "object_support_penetration_m",
        )
    }
    segment = np.zeros(6)
    for t in range(reference.frame_count):
        data.qpos[addresses] = reference.qpos[t]
        data.mocap_pos[mocap_id] = reference.object_poses_w[t, 0, :3]
        data.mocap_quat[mocap_id] = reference.object_poses_w[t, 0, 3:]
        mujoco.mj_forward(model, data)
        for c in data.contact:
            a, b = int(c.geom1), int(c.geom2)
            if a in robot_geoms and b in robot_geoms:
                key = "self_penetration_m"
            elif (a in robot_geoms and b in support_geoms) or (
                b in robot_geoms and a in support_geoms
            ):
                key = "robot_support_penetration_m"
            elif (a in robot_geoms and b in object_geoms) or (
                b in robot_geoms and a in object_geoms
            ):
                key = "robot_object_penetration_m"
            elif (a in support_geoms and b in object_geoms) or (
                b in support_geoms and a in object_geoms
            ):
                key = "object_support_penetration_m"
            else:
                continue
            diagnostics[key][t] = max(diagnostics[key][t], -float(c.dist))
        for side, groups in enumerate(geom_groups):
            for link, geoms in enumerate(groups):
                closest = contact_distance_m
                witness = None
                for a in geoms:
                    for b in object_geoms:
                        distance = mujoco.mj_geomDistance(
                            model, data, a, b, contact_distance_m, segment
                        )
                        if distance < closest:
                            closest, witness = distance, segment.copy()
                if witness is None:
                    # Finite, explicit inactive geometry. It never enters
                    # the contact-wrench computation while active is false.
                    point = data.xpos[model.body(str(names[side, link])).id]
                    link_points[t, side, link] = point
                    object_points[t, side, link] = point
                    normals[t, side, link] = [0, 0, 1]
                    continue
                link_points[t, side, link] = witness[:3]
                object_points[t, side, link] = witness[3:]
                normal = witness[3:] - witness[:3]
                # Signed witness distance reverses in penetration.
                if closest < 0:
                    normal *= -1
                length = np.linalg.norm(normal)
                normals[t, side, link] = (
                    normal / length if length > 1e-10 else [0, 0, 1]
                )
                distances[t, side, link] = closest
                active[t, side, link] = True
    contacts = ContactSequence(
        hand_sides=sides,
        link_names=names,
        link_positions_w=link_points,
        link_normals_w=normals,
        object_positions_w=object_points,
        object_normals_w=-normals,
        object_indices=np.zeros(shape, dtype=np.int32),
        active=active,
    )
    diagnostics["contact_signed_distance_m"] = distances
    return contacts, diagnostics
