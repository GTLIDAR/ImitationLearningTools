"""Add fixed-base arm state to a processed two-hand Reference.

Hand retargeting stays upstream. This stage solves the mounting kinematics,
recomputes achieved hand geometry, and records reachability and speed labels.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from iltools.core import DexterousReference, Trajectory, sha256_file
from iltools.retarget.dual_hand import MujocoDualHandRetargeter


@dataclass(frozen=True)
class MountedHandConfig:
    arm_joint_names: tuple[str, ...]
    left_wrist_site: str
    right_wrist_site: str
    fixed_root_pose_w: tuple[float, ...]
    robot_name: str
    initial_arm_qpos: tuple[float, ...] = ()
    iterations: int = 120
    position_tolerance_m: float = 0.025
    orientation_tolerance_rad: float = 0.15
    arm_speed_limit_rad_s: float = 1.5


def _rotation(pose: np.ndarray) -> Rotation:
    return Rotation.from_quat(np.asarray(pose)[..., 3:7], scalar_first=True)


def _robot_poses(poses: np.ndarray, root: np.ndarray) -> np.ndarray:
    inverse = _rotation(root).inv()
    return np.concatenate(
        (
            inverse.apply(poses[:, :3] - root[:3]),
            (inverse * _rotation(poses)).as_quat(scalar_first=True),
        ),
        axis=-1,
    )


def retarget_mounted_hands(
    reference: DexterousReference,
    model_path: str | Path,
    cfg: MountedHandConfig,
) -> tuple[DexterousReference, dict[str, np.ndarray]]:
    """Solve sequential wrist IK with ILTools and return an audited Reference.

    Feasible labels describe kinematic reachability and recorded joint speed.
    They do not certify self collision, support clearance, or dynamic tracking.
    The caller must perform those checks against the actual runtime assembly.
    """
    if reference.robot_layout != "dual_floating_hand":
        raise ValueError("Expected a dual_floating_hand source Reference.")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if model.nq != model.nv:
        raise ValueError("Mounted IK expects only scalar joints and a fixed root.")
    # IK is purely kinematic; avoid collision detection in every optimizer step.
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    names = tuple(model.joint(i).name for i in range(model.njnt))
    root = np.asarray(cfg.fixed_root_pose_w, dtype=np.float64)
    if root.shape != (7,) or not np.isclose(np.linalg.norm(root[3:]), 1):
        raise ValueError("The fixed base must be XYZ plus a unit WXYZ quaternion.")
    source_index = {name: i for i, name in enumerate(reference.joint_names)}
    initial_qpos = model.qpos0.copy()
    if cfg.initial_arm_qpos:
        if len(cfg.initial_arm_qpos) != len(cfg.arm_joint_names):
            raise ValueError(
                "Initial arm positions must align with the arm joint names."
            )
        initial_qpos[[names.index(n) for n in cfg.arm_joint_names]] = (
            cfg.initial_arm_qpos
        )
    trajectory = Trajectory(
        observations={
            "left_wrist_pose_w": _robot_poses(reference.left_wrist_pose_w, root),
            "right_wrist_pose_w": _robot_poses(reference.right_wrist_pose_w, root),
            "left_finger_qpos": reference.qpos[
                :, [source_index[n] for n in reference.left_joint_names]
            ],
            "right_finger_qpos": reference.qpos[
                :, [source_index[n] for n in reference.right_joint_names]
            ],
        },
        infos={"coordinate_frame": "robot"},
        dt=1 / reference.fps,
    )
    retargeter = MujocoDualHandRetargeter(
        model,
        target_joint_names=names,
        arm_joint_names=cfg.arm_joint_names,
        left_finger_joint_names=reference.left_joint_names,
        right_finger_joint_names=reference.right_joint_names,
        left_wrist_site=cfg.left_wrist_site,
        right_wrist_site=cfg.right_wrist_site,
        iterations=cfg.iterations,
        damping=0.01,
        max_step=0.25,
        orientation_weight=0.03,
        tolerance=1e-4,
        initial_qpos=initial_qpos,
        restart_position_error_m=cfg.position_tolerance_m,
    )
    result = retargeter.retarget(trajectory)
    qpos = np.asarray(result.observations["qpos"])
    qvel = np.gradient(qpos, 1 / reference.fps, axis=0)
    data = mujoco.MjData(model)
    root_rotation = _rotation(root)
    wrists = {}
    frames = {}
    diagnostics = {}
    for side in ("left", "right"):
        site_id = model.site(getattr(cfg, f"{side}_wrist_site")).id
        wrist = np.empty((reference.frame_count, 7))
        for t, q in enumerate(qpos):
            data.qpos[:] = q
            mujoco.mj_kinematics(model, data)
            wrist[t, :3] = root_rotation.apply(data.site_xpos[site_id]) + root[:3]
            wrist[t, 3:] = (
                root_rotation
                * Rotation.from_matrix(data.site_xmat[site_id].reshape(3, 3))
            ).as_quat(scalar_first=True)
        target = getattr(reference, f"{side}_wrist_pose_w")
        # Pink's frame list also contains joint/site frames absent from the
        # MuJoCo body list. Their local hand geometry is unchanged by arm IK.
        finger_names = getattr(reference, f"{side}_joint_names")
        difference = (
            qpos[:, [names.index(n) for n in finger_names]]
            - reference.qpos[:, [source_index[n] for n in finger_names]]
        )
        if np.abs(difference).max() > 1e-5:
            raise ValueError("Source finger positions exceed the mounted model limits.")
        hand_frames = getattr(reference, f"{side}_hand_frame_poses_w").copy()
        delta = _rotation(wrist) * _rotation(target).inv()
        for t in range(reference.frame_count):
            hand_frames[t, :, :3] = (
                delta[t].apply(hand_frames[t, :, :3] - target[t, :3]) + wrist[t, :3]
            )
            hand_frames[t, :, 3:] = (
                delta[t] * Rotation.from_quat(hand_frames[t, :, 3:], scalar_first=True)
            ).as_quat(scalar_first=True)
        diagnostics[f"{side}_position_error_m"] = np.linalg.norm(
            wrist[:, :3] - target[:, :3], axis=-1
        )
        diagnostics[f"{side}_orientation_error_rad"] = (
            _rotation(wrist) * _rotation(target).inv()
        ).magnitude()
        wrists[side], frames[side] = wrist, hand_frames
    arm_indices = [names.index(name) for name in cfg.arm_joint_names]
    diagnostics["arm_max_speed_rad_s"] = np.max(np.abs(qvel[:, arm_indices]), axis=-1)
    feasible = diagnostics["arm_max_speed_rad_s"] <= cfg.arm_speed_limit_rad_s
    for side in ("left", "right"):
        feasible &= diagnostics[f"{side}_position_error_m"] <= cfg.position_tolerance_m
        feasible &= (
            diagnostics[f"{side}_orientation_error_rad"]
            <= cfg.orientation_tolerance_rad
        )
    diagnostics["kinematically_feasible"] = feasible
    # Transform hand-side contact points with the achieved wrist. Object-side
    # geometry stays on the measured object trajectory.
    contacts = reference.contacts
    if contacts is not None:
        positions = contacts.link_positions_w.copy()
        normals = contacts.link_normals_w.copy()
        for slot, side in enumerate(contacts.hand_sides):
            target = getattr(reference, f"{side}_wrist_pose_w")
            delta = _rotation(wrists[side]) * _rotation(target).inv()
            for t in range(reference.frame_count):
                positions[t, slot] = (
                    delta[t].apply(positions[t, slot] - target[t, :3])
                    + wrists[side][t, :3]
                )
                normals[t, slot] = delta[t].apply(normals[t, slot])
        contacts = replace(contacts, link_positions_w=positions, link_normals_w=normals)
    metadata = dict(reference.metadata)
    metadata["mounted_retarget"] = {
        "method": "iltools_mujoco_dual_wrist_se3",
        "model_sha256": sha256_file(model_path),
        "source_reference_sha256": sha256_file(reference.source_path)
        if reference.source_path
        else None,
        "position_tolerance_m": cfg.position_tolerance_m,
        "orientation_tolerance_rad": cfg.orientation_tolerance_rad,
        "arm_speed_limit_rad_s": cfg.arm_speed_limit_rad_s,
        "kinematically_feasible_frames": np.flatnonzero(feasible).tolist(),
        "qualification": "kinematics only; collision and runtime qualification pending",
    }
    output = replace(
        reference,
        robot_name=cfg.robot_name,
        robot_layout="fixed_base",
        joint_names=names,
        qpos=qpos,
        qvel=qvel,
        fixed_root_pose_w=root,
        left_wrist_pose_w=wrists["left"],
        right_wrist_pose_w=wrists["right"],
        left_wrist_twist_w=None,
        right_wrist_twist_w=None,
        left_hand_frame_poses_w=frames["left"],
        right_hand_frame_poses_w=frames["right"],
        contacts=contacts,
        metadata=metadata,
        training_qualification=None,
        source_path=None,
    )
    return output, diagnostics


def refresh_mounted_geometry(
    reference: DexterousReference,
    model_path: str | Path,
    qpos: np.ndarray,
) -> DexterousReference:
    """Recompute body poses and velocities; invalidate contacts after projection.

    The output frame lists contain MuJoCo hand bodies. Pink-only joint/site
    frames are removed explicitly, since they are not runtime body frames.
    Object geometry and motion are unchanged.
    """
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    addresses = [model.jnt_qposadr[model.joint(n).id] for n in reference.joint_names]
    root = reference.fixed_root_pose_w
    rotation = _rotation(root)
    poses = np.zeros((reference.frame_count, model.nbody, 7))
    for t, row in enumerate(qpos):
        data.qpos[addresses] = row
        mujoco.mj_kinematics(model, data)
        poses[t, :, :3] = rotation.apply(data.xpos) + root[:3]
        poses[t, :, 3:] = (
            rotation * Rotation.from_quat(data.xquat, scalar_first=True)
        ).as_quat(scalar_first=True)
    changes = {"qpos": qpos, "qvel": np.gradient(qpos, 1 / reference.fps, axis=0)}
    for side in ("left", "right"):
        names = tuple(
            model.body(i).name
            for i in range(1, model.nbody)
            if model.body(i).name.startswith(f"{side}_")
        )
        changes[f"{side}_hand_frame_names"] = names
        changes[f"{side}_hand_frame_poses_w"] = poses[
            :, [model.body(n).id for n in names]
        ]
        changes[f"{side}_wrist_pose_w"] = poses[
            :, model.body(getattr(reference, f"{side}_wrist_frame_name")).id
        ]
        changes[f"{side}_wrist_twist_w"] = None
    # Finger projection invalidates the old contact points. ARCTIC stores
    # MANO link names, not robot links. Recompute contacts against the actual
    # robot and object collision meshes before publishing training data.
    changes["contacts"] = None
    changes["metadata"] = {**reference.metadata, "contact_geometry_pending": True}
    changes["source_path"] = None
    changes["training_qualification"] = None
    return replace(reference, **changes)


def project_mounted_self_collisions(
    reference: DexterousReference,
    model_path: str | Path,
) -> tuple[DexterousReference, dict]:
    """Apply ILTools' maximal-safe-closure search separately to each hand."""
    from iltools.retarget.self_collision import MujocoSelfCollisionClosureProjector

    model = mujoco.MjModel.from_xml_path(str(model_path))
    qpos = reference.qpos.copy()
    reports = {}
    for side in ("left", "right"):
        geoms = [
            i
            for i in range(model.ngeom)
            if model.body(int(model.geom_bodyid[i])).name.startswith(f"{side}_")
            and (model.geom_contype[i] or model.geom_conaffinity[i])
        ]
        projector = MujocoSelfCollisionClosureProjector(
            model,
            trajectory_joint_names=reference.joint_names,
            closure_joint_names=getattr(reference, f"{side}_joint_names"),
            robot_geom_names=geoms,
        )
        qpos, scales, report = projector.project(qpos)
        reports[side] = {**report.as_dict(), "closure_scales": scales.tolist()}
        if not report.qualified:
            raise ValueError(
                f"{side} hand self-collision projection failed: {report.as_dict()}"
            )
    output = refresh_mounted_geometry(reference, model_path, qpos)
    output.metadata = {**output.metadata, "mounted_self_collision_projection": reports}
    return output, reports


def crop_mounted_reference(
    reference: DexterousReference, start: int, stop: int
) -> DexterousReference:
    """Keep a declared continuous local-test segment without changing its rate."""
    if not 0 <= start < stop <= reference.frame_count or stop - start < 2:
        raise ValueError("A Reference crop needs at least two valid frames.")
    time_fields = (
        "qpos",
        "qvel",
        "left_wrist_pose_w",
        "right_wrist_pose_w",
        "left_wrist_twist_w",
        "right_wrist_twist_w",
        "object_poses_w",
        "object_twists_w",
        "left_hand_frame_poses_w",
        "right_hand_frame_poses_w",
    )
    changes = {
        name: getattr(reference, name)[start:stop].copy() for name in time_fields
    }
    contacts = reference.contacts
    if contacts is not None:
        changes["contacts"] = replace(
            contacts,
            **{
                name: getattr(contacts, name)[start:stop].copy()
                for name in (
                    "link_positions_w",
                    "link_normals_w",
                    "object_positions_w",
                    "object_normals_w",
                    "object_indices",
                    "active",
                )
            },
        )
    mounted = dict(reference.metadata["mounted_retarget"])
    mounted["kinematically_feasible_frames"] = [
        i - start for i in mounted["kinematically_feasible_frames"] if start <= i < stop
    ]
    mounted.pop("reset_frames", None)
    changes["metadata"] = {
        **reference.metadata,
        "mounted_retarget": mounted,
        "local_test_crop": {
            "source_start_frame": start,
            "source_stop_frame_exclusive": stop,
            "source_frame_count": reference.frame_count,
            "fps": reference.fps,
        },
    }
    changes["source_path"] = None
    changes["training_qualification"] = None
    return replace(reference, **changes)


def retime_mounted_reference(
    reference: DexterousReference, model_path: str | Path, factor: int
) -> DexterousReference:
    """Slow the motion with interpolated samples, keeping the control rate.

    Contacts and collision qualification are invalidated. Interpolated poses
    must be projected/audited before this Reference can enter training.
    """
    if int(factor) != factor or factor < 1:
        raise ValueError("The retiming factor must be a positive integer.")
    times = np.arange(reference.frame_count, dtype=float)
    target = np.linspace(
        0, reference.frame_count - 1, (reference.frame_count - 1) * factor + 1
    )
    qpos = np.stack(
        [
            np.interp(target, times, reference.qpos[:, i])
            for i in range(reference.qpos.shape[1])
        ],
        axis=-1,
    )

    def poses(values):
        flat = np.asarray(values).reshape(reference.frame_count, -1, 7)
        out = np.empty((len(target), flat.shape[1], 7))
        for item in range(flat.shape[1]):
            out[:, item, :3] = np.stack(
                [np.interp(target, times, flat[:, item, axis]) for axis in range(3)],
                axis=-1,
            )
            out[:, item, 3:] = Slerp(
                times, Rotation.from_quat(flat[:, item, 3:], scalar_first=True)
            )(target).as_quat(scalar_first=True)
        return out.reshape((len(target), *values.shape[1:]))

    metadata = dict(reference.metadata)
    metadata["mounted_time_stretch"] = {
        "factor": factor,
        "input_frames": reference.frame_count,
        "output_frames": len(target),
        "fps": reference.fps,
        "method": "linear scalar joints/positions and quaternion SLERP; collision audit required",
    }
    metadata["mounted_retarget"] = {
        **metadata["mounted_retarget"],
        "kinematically_feasible_frames": [],
    }
    metadata["mounted_retarget"].pop("reset_frames", None)
    result = replace(
        reference,
        qpos=qpos,
        qvel=None,
        contacts=None,
        metadata=metadata,
        object_poses_w=poses(reference.object_poses_w),
        object_twists_w=None,
        left_wrist_pose_w=poses(reference.left_wrist_pose_w),
        left_wrist_twist_w=None,
        right_wrist_pose_w=poses(reference.right_wrist_pose_w),
        right_wrist_twist_w=None,
        left_hand_frame_poses_w=poses(reference.left_hand_frame_poses_w),
        right_hand_frame_poses_w=poses(reference.right_hand_frame_poses_w),
        training_qualification=None,
        source_path=None,
    )
    return refresh_mounted_geometry(result, model_path, qpos)
