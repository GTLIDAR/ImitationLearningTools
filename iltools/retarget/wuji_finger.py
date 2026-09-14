"""Wuji Hand 2 finger retargeting: anatomical limits, tips-only IK, audit gates.

This module adopts the MANO-keypoint-to-Wuji recipe reported as successful in
the ``wuji_retargeting_notes`` reference implementation (pink/pinocchio) on
the ILTools MuJoCo stack.  The archive contains no source sequence or
machine-readable audit record, so its reported corpus result remains upstream
evidence rather than a result reproduced by this repository.  The recipe is:

1. Constrain task space only: the five fingertip positions (plus the wrist,
   which this repository solves in its own arm-IK stage).  Do not imitate
   per-joint angles.
2. Tighten the joint limits inside the model before the IK sees them.  The
   Wuji URDF/MJCF lets every interphalangeal joint hyperextend to -60 deg, so
   a tips-only IK can place a tip exactly right with a Z-folded (inverted)
   finger.  Position targets cannot disambiguate the flexion sign when the
   phalanx lengths differ from the human's.  A hard 0 deg lower bound on the
   PIP/DIP/thumb-MCP/thumb-IP joints removes the inverted half of the
   solution space.
3. Add a weak posture prior toward a slight-curl rest pose so the per-finger
   redundancy settles mid-range instead of at a joint limit.
4. Keep the fingertip-target scale at 1.0.  Scaling targets about the wrist
   pulls the fingertips off the object surface and silently corrupts the
   contact geometry.
5. Judge with numbers, not eyes: a kinematic replay shows exactly what the IK
   optimized, so it cannot fail visually.  Gate on the joint-space audit.
6. Measure the palm-frame alignment geometrically (Procrustes fit of the four
   source knuckles onto the robot knuckle origins), never from behaviour.

Pre-registered acceptance gates for one retargeted sequence:
hyperextension 0 % of frames, wrist residual mean <= 2.5 mm, fingertip
residual p95 <= 5 mm, and per-joint frame-to-frame jump <= 35 deg.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import mujoco as _mujoco
import numpy as np

mujoco: Any = _mujoco


# Anatomical finger-limit envelope in degrees, keyed by joint-name substring.
# The first matching key wins.  ``None`` keeps the model bound.  Values follow
# the reference implementation: the 0 deg lower bounds are the Wuji official
# ``soft_min`` taken to a hard bound; the upper caps are the anatomical limits
# CHORD bakes into its own Sharpa hand description.
WUJI_ANATOMICAL_LIMITS_DEG: dict[str, tuple[float | None, float | None]] = {
    "mcp_abd": (-40.0, 40.0),
    "_pip": (0.0, 100.0),
    "_dip": (0.0, 80.0),
    "thumb_mcp": (0.0, None),
    "thumb_ip": (0.0, 100.0),
}

# Slight-curl rest pose for the posture prior, keyed by joint-name substring.
WUJI_POSTURE_REST_DEG: dict[str, float] = {
    "_flex": 20.0,
    "_abd": 0.0,
    "_pip": 30.0,
    "_dip": 20.0,
    "thumb_cmc_flex": 20.0,
    "thumb_cmc_abd": 0.0,
    "thumb_mcp": 20.0,
    "thumb_ip": 20.0,
}

# Interphalangeal flexion joints that must never hyperextend.
HYPEREXTENSION_JOINT_KEYS = ("_pip", "_dip", "thumb_mcp", "thumb_ip")

# Pre-registered per-sequence acceptance gates.
WUJI_FINGER_GATES: dict[str, float] = {
    "hyperextension_pct": 0.0,
    "wrist_mean_mm": 2.5,
    "tip_p95_mm": 5.0,
    "jump_max_deg": 35.0,
}

# Fingertip site names in the Vega-Wuji MJCF, thumb-to-pinky order.
WUJI_FINGERTIP_SITE_NAMES: dict[str, tuple[str, ...]] = {
    side: (
        f"{prefix}_thumb_tip",
        f"{prefix}_index_finger_tip",
        f"{prefix}_middle_finger_tip",
        f"{prefix}_ring_finger_tip",
        f"{prefix}_pinky_tip",
    )
    for side, prefix in (("left", "l"), ("right", "r"))
}

# Knuckle (MCP flexion) joints whose origins define the palm plane.
WUJI_KNUCKLE_JOINT_NAMES: dict[str, tuple[str, ...]] = {
    side: (
        f"{prefix}_index_finger_mcp_flex",
        f"{prefix}_middle_finger_mcp_flex",
        f"{prefix}_ring_finger_mcp_flex",
        f"{prefix}_pinky_mcp_flex",
    )
    for side, prefix in (("left", "l"), ("right", "r"))
}

WUJI_PIP_JOINT_NAMES: dict[str, tuple[str, ...]] = {
    side: (
        f"{prefix}_index_finger_pip",
        f"{prefix}_middle_finger_pip",
        f"{prefix}_ring_finger_pip",
        f"{prefix}_pinky_pip",
    )
    for side, prefix in (("left", "l"), ("right", "r"))
}

# In-palm-plane angle axes (numerator, denominator of atan2) for which
# _axis_rotation(palm_normal_axis, theta) increases the angle by theta.
_PALM_PLANE_ANGLE_AXES: dict[int, tuple[int, int]] = {0: (2, 1), 1: (0, 2), 2: (1, 0)}


def _match_key(name: str, keys: Sequence[str]) -> str | None:
    for key in keys:
        if key in name:
            return key
    return None


def apply_anatomical_finger_limits(
    model: Any,
    *,
    limits_deg: Mapping[str, tuple[float | None, float | None]] | None = None,
) -> list[dict[str, Any]]:
    """Tighten hinge-joint ranges in the model in place; return log rows.

    Only joints whose name contains one of the ``limits_deg`` keys change.
    A bound only tightens: the new lower bound is the maximum of the model
    bound and the anatomical bound, and symmetrically for the upper bound.
    """

    table = dict(WUJI_ANATOMICAL_LIMITS_DEG if limits_deg is None else limits_deg)
    rows: list[dict[str, Any]] = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is None:
            continue
        key = _match_key(name, tuple(table))
        if key is None:
            continue
        lower_deg, upper_deg = table[key]
        old = (float(model.jnt_range[joint_id, 0]), float(model.jnt_range[joint_id, 1]))
        if lower_deg is not None:
            model.jnt_range[joint_id, 0] = max(old[0], float(np.radians(lower_deg)))
        if upper_deg is not None:
            model.jnt_range[joint_id, 1] = min(old[1], float(np.radians(upper_deg)))
        new = (float(model.jnt_range[joint_id, 0]), float(model.jnt_range[joint_id, 1]))
        rows.append(
            {
                "joint": name,
                "old_deg": [round(np.degrees(value), 1) for value in old],
                "new_deg": [round(np.degrees(value), 1) for value in new],
            }
        )
        if not np.allclose(model.jnt_range[joint_id], new):
            raise RuntimeError(f"MuJoCo joint-range write-back failed for {name}.")
    return rows


def posture_rest_qpos(
    joint_names: Sequence[str],
    *,
    rest_deg: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Return the slight-curl rest pose in radians for the named joints."""

    table = dict(WUJI_POSTURE_REST_DEG if rest_deg is None else rest_deg)
    # More specific thumb keys must win over the generic flex/abd keys.
    ordered = sorted(table, key=len, reverse=True)
    rest = np.zeros(len(joint_names), dtype=np.float64)
    for index, name in enumerate(joint_names):
        key = _match_key(name, ordered)
        if key is not None:
            rest[index] = np.radians(table[key])
    return rest


@dataclass(frozen=True)
class FingertipIkConfig:
    """Numerical settings for the tips-only finger IK.

    ``posture_cost`` is the weight of the rest-pose rows relative to a
    fingertip position weight of 1.0 per meter, matching the reference
    implementation's pink cost of 2e-3.
    """

    iterations: int = 100
    damping: float = 0.03
    max_step_rad: float = 0.15
    # Keep a small numerical margin below the pre-registered 35 deg gate.
    max_frame_change_rad: float | None = float(np.radians(34.99))
    tolerance_m: float = 1.0e-4
    line_search_steps: int = 8
    tip_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 0.5)
    posture_cost: float = 2.0e-3

    def __post_init__(self) -> None:
        if int(self.iterations) < 1:
            raise ValueError("iterations must be positive.")
        for name in ("damping", "max_step_rad", "tolerance_m"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.max_frame_change_rad is not None and (
            not np.isfinite(self.max_frame_change_rad)
            or self.max_frame_change_rad <= 0.0
        ):
            raise ValueError("max_frame_change_rad must be finite and positive.")
        if int(self.line_search_steps) < 1:
            raise ValueError("line_search_steps must be positive.")
        if len(self.tip_weights) == 0 or any(
            not np.isfinite(weight) or weight <= 0.0 for weight in self.tip_weights
        ):
            raise ValueError("tip_weights must be finite and positive.")
        if not np.isfinite(self.posture_cost) or self.posture_cost < 0.0:
            raise ValueError("posture_cost must be finite and non-negative.")


@dataclass
class FingertipIkReport:
    """Residual evidence for one tips-only finger IK trajectory solve."""

    frame_count: int = 0
    site_names: tuple[str, ...] = ()
    tip_errors_m: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    iterations: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        errors = np.asarray(self.tip_errors_m, dtype=np.float64)
        per_frame_mean = (
            errors.mean(axis=1) if errors.size else np.zeros(0, dtype=np.float64)
        )
        return {
            "frame_count": self.frame_count,
            "site_names": list(self.site_names),
            "tip_mean_mm": float(errors.mean() * 1e3) if errors.size else 0.0,
            "tip_p95_mm": (
                float(np.percentile(per_frame_mean, 95.0) * 1e3) if errors.size else 0.0
            ),
            "tip_max_mm": float(errors.max() * 1e3) if errors.size else 0.0,
            "per_site_mean_mm": (
                [float(value * 1e3) for value in errors.mean(axis=0)]
                if errors.size
                else []
            ),
            "mean_iterations": (
                float(np.mean(self.iterations)) if self.iterations else 0.0
            ),
        }


@dataclass(frozen=True)
class FingertipAvoidanceConfig:
    """Keep named robot geoms out of a scene geom while the tips are solved.

    A fingertip target that lies on an object surface is reachable in more
    than one way, and the cheapest way is usually to sink part of the hand
    into the object. Adding the clearance as a task rather than as a
    post-hoc repair lets the solver trade a little fingertip accuracy for
    staying outside, instead of a later stage undoing its work.

    This is a first-order barrier, so it stalls where the escape direction
    lies in the null space of the contacting point's Jacobian: a straight
    finger pressed face-on into a slab cannot push itself off that slab by
    bending, because bending sweeps the tip sideways to first order. It is a
    cost, not a hard constraint, and a fail-closed geometry audit still has
    to gate the result.
    """

    #: ``(robot_geom, other_geom)`` pairs to keep apart. Each entry is a geom
    #: name or a geom id, because MuJoCo models routinely leave collision
    #: geoms unnamed and those are exactly the ones an audit reports.
    geom_pairs: tuple[tuple[str | int, str | int], ...] = ()
    #: Required separation. Negative permits that much interpenetration.
    clearance_m: float = 0.0
    #: Cost amplitude relative to a fingertip position weight of 1.0 per metre.
    weight: float = 10.0
    #: Distance beyond which a pair is ignored, for the MuJoCo range query.
    query_cutoff_m: float = 0.05
    #: Mocap body carrying the scene geom, when the scene object is animated.
    mocap_body_name: str | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.clearance_m):
            raise ValueError("clearance_m must be finite.")
        for name in ("weight", "query_cutoff_m"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.query_cutoff_m <= self.clearance_m:
            raise ValueError("query_cutoff_m must exceed clearance_m.")


def solve_fingertip_trajectory(
    model: Any,
    *,
    trajectory_joint_names: Sequence[str],
    qpos: np.ndarray,
    variable_joint_names: Sequence[str],
    target_site_names: Sequence[str],
    target_positions: np.ndarray,
    config: FingertipIkConfig | None = None,
    avoidance: FingertipAvoidanceConfig | None = None,
    scene_object_poses: np.ndarray | None = None,
) -> tuple[np.ndarray, FingertipIkReport]:
    """Solve the named finger joints toward fingertip position targets.

    The solver is a damped least-squares IK over the ``variable_joint_names``
    only; every other trajectory joint stays exactly at its input value.  All
    target sites of one frame are solved jointly, with a weak posture prior
    toward the slight-curl rest pose and a warm start from the previous output
    frame.  Joint values clamp to ``model.jnt_range`` every step, so callers
    must apply :func:`apply_anatomical_finger_limits` first to exclude
    hyperextended solutions.

    With ``avoidance`` the solve also carries a clearance task against the
    named scene geom, and the line search scores penetration alongside
    fingertip error, so a step that reaches a target by burying the hand is
    rejected. ``scene_object_poses`` supplies the per-frame ``[T, 7]`` pose of
    ``avoidance.mocap_body_name`` when the scene object moves.
    """

    settings = FingertipIkConfig() if config is None else config
    values = np.asarray(qpos, dtype=np.float64)
    targets = np.asarray(target_positions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(trajectory_joint_names):
        raise ValueError("qpos must have shape [T, len(trajectory_joint_names)].")
    if targets.shape != (len(values), len(target_site_names), 3):
        raise ValueError("target_positions must have shape [T, n_sites, 3].")
    if len(settings.tip_weights) != len(target_site_names):
        raise ValueError("tip_weights length must match target_site_names.")
    if not np.isfinite(values).all() or not np.isfinite(targets).all():
        raise ValueError("qpos and target_positions must be finite.")
    variable = tuple(variable_joint_names)
    unknown = set(variable) - set(trajectory_joint_names)
    if not variable or unknown:
        raise ValueError(f"variable_joint_names invalid; unknown: {sorted(unknown)}.")

    data = mujoco.MjData(model)
    trajectory_qpos_addresses = np.asarray(
        [
            int(model.jnt_qposadr[model.joint(name).id])
            for name in trajectory_joint_names
        ],
        dtype=np.int32,
    )
    variable_indices = np.asarray(
        [trajectory_joint_names.index(name) for name in variable], dtype=np.int32
    )
    variable_dofs = np.asarray(
        [int(model.jnt_dofadr[model.joint(name).id]) for name in variable],
        dtype=np.int32,
    )
    lower = np.asarray(
        [float(model.jnt_range[model.joint(name).id, 0]) for name in variable],
        dtype=np.float64,
    )
    upper = np.asarray(
        [float(model.jnt_range[model.joint(name).id, 1]) for name in variable],
        dtype=np.float64,
    )
    site_ids = np.asarray(
        [int(model.site(name).id) for name in target_site_names], dtype=np.int32
    )
    rest = posture_rest_qpos(variable)
    weights = np.repeat(np.asarray(settings.tip_weights, dtype=np.float64), 3)
    weight_matrix = np.diag(weights**2)
    posture_weight = settings.posture_cost**2

    avoid_pairs: list[tuple[int, int, int]] = []
    avoid_mocap_id: int | None = None
    scene_poses: np.ndarray | None = None
    if avoidance is not None and avoidance.geom_pairs:

        def _geom_id(value: str | int) -> int:
            return (
                int(value)
                if isinstance(value, (int, np.integer))
                else int(model.geom(value).id)
            )

        for robot_ref, scene_ref in avoidance.geom_pairs:
            robot_geom = _geom_id(robot_ref)
            scene_geom = _geom_id(scene_ref)
            avoid_pairs.append(
                (
                    robot_geom,
                    scene_geom,
                    int(model.geom_bodyid[robot_geom]),
                    int(model.geom_bodyid[scene_geom]),
                )
            )
        if avoidance.mocap_body_name is not None:
            body_id = int(model.body(avoidance.mocap_body_name).id)
            avoid_mocap_id = int(model.body_mocapid[body_id])
            if avoid_mocap_id < 0:
                raise ValueError(f"{avoidance.mocap_body_name!r} is not a mocap body.")
            if scene_object_poses is None:
                raise ValueError(
                    "scene_object_poses is required with a mocap scene body."
                )
            scene_poses = np.asarray(scene_object_poses, dtype=np.float64)
            if scene_poses.shape != (len(values), 7):
                raise ValueError("scene_object_poses must have shape [T, 7].")
    avoid_weight = float(avoidance.weight) ** 2 if avoid_pairs else 0.0
    avoid_clearance = float(avoidance.clearance_m) if avoid_pairs else 0.0
    avoid_cutoff = float(avoidance.query_cutoff_m) if avoid_pairs else 0.0
    fromto = np.zeros(6, dtype=np.float64)
    point_jacobian = np.zeros((3, model.nv), dtype=np.float64)
    other_jacobian = np.zeros((3, model.nv), dtype=np.float64)

    def avoidance_rows(frame: int) -> tuple[np.ndarray, np.ndarray, float]:
        """Rows pushing violating robot geoms out, plus the worst violation."""

        rows: list[np.ndarray] = []
        errors_out: list[float] = []
        worst = 0.0
        for robot_geom, scene_geom, robot_body, scene_body in avoid_pairs:
            distance = float(
                mujoco.mj_geomDistance(
                    model, data, robot_geom, scene_geom, avoid_cutoff, fromto
                )
            )
            if distance >= avoid_cutoff - 1.0e-12 or distance >= avoid_clearance:
                continue
            direction = fromto[3:] - fromto[:3]
            norm = float(np.linalg.norm(direction))
            if norm <= 1.0e-9:
                continue
            unit = direction / norm
            # ``direction`` runs from the witness on the robot to the witness
            # on the scene geom, and its meaning flips with contact. While the
            # two are apart, moving the robot along it closes the gap, so the
            # escape is -unit. Once they overlap, the robot witness is the
            # deepest point and the scene witness is the surface it must come
            # back out through, so the escape is +unit. Using one sign for
            # both drives penetrating geoms deeper.
            escape = unit if distance < 0.0 else -unit
            # Both witnesses move when the pair is robot-on-robot and both
            # bodies hang off the joints being solved, so the separation
            # responds to the RELATIVE motion. Using only the first body's
            # Jacobian can push in a direction that actually closes the gap.
            # For a static or mocap scene body the second term is zero, so the
            # relative form is correct in every case.
            mujoco.mj_jac(
                model, data, point_jacobian, None, fromto[:3].copy(), robot_body
            )
            relative = point_jacobian[:, variable_dofs].copy()
            mujoco.mj_jac(
                model, data, other_jacobian, None, fromto[3:].copy(), scene_body
            )
            relative -= other_jacobian[:, variable_dofs]
            rows.append(escape @ relative)
            errors_out.append(avoid_clearance - distance)
            worst = max(worst, avoid_clearance - distance)
        if not rows:
            return (
                np.zeros((0, len(variable)), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                0.0,
            )
        return np.asarray(rows), np.asarray(errors_out), worst

    output = values.copy()
    report = FingertipIkReport(
        frame_count=len(values),
        site_names=tuple(target_site_names),
        tip_errors_m=np.zeros((len(values), len(site_ids)), dtype=np.float64),
    )
    jacobian_translation = np.zeros((3, model.nv), dtype=np.float64)
    jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)

    def frame_objective(
        solution: np.ndarray, residuals: np.ndarray, violation: float = 0.0
    ) -> float:
        # ``residuals`` is either per-site scalar norms or the raw per-site
        # error vectors; both weight to the same quantity.
        values_in = np.asarray(residuals, dtype=np.float64)
        weighted = (
            weights * values_in.reshape(-1)
            if values_in.ndim == 2
            else np.asarray(settings.tip_weights, dtype=np.float64) * values_in
        )
        return float(
            np.sum(weighted**2)
            + posture_weight * np.sum((solution - rest) ** 2)
            + avoid_weight * float(violation) ** 2
        )

    def solve_frame(
        frame: int,
        seed: np.ndarray,
        frame_lower: np.ndarray,
        frame_upper: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        solution = np.clip(seed, frame_lower, frame_upper)
        base = output[frame].copy()
        if avoid_mocap_id is not None and scene_poses is not None:
            data.mocap_pos[avoid_mocap_id] = scene_poses[frame, :3]
            data.mocap_quat[avoid_mocap_id] = scene_poses[frame, 3:7]
        iterations_used = int(settings.iterations)
        for iteration in range(int(settings.iterations)):
            base[variable_indices] = solution
            data.qpos[trajectory_qpos_addresses] = base
            mujoco.mj_forward(model, data)
            errors = targets[frame] - data.site_xpos[site_ids]
            # With a posture task, a zero tip residual is not the full-QP
            # optimum: redundant joints can still move toward the slight-curl
            # pose in the fingertip Jacobian null space. Pink also includes the
            # posture rows in every QP solve. Only use a tip-only early exit
            # when the posture cost is disabled.
            avoid_task, avoid_error, avoid_violation = (
                avoidance_rows(frame)
                if avoid_pairs
                else (
                    np.zeros((0, len(variable)), dtype=np.float64),
                    np.zeros(0, dtype=np.float64),
                    0.0,
                )
            )
            if (
                posture_weight == 0.0
                and avoid_violation == 0.0
                and float(np.max(np.linalg.norm(errors, axis=-1)))
                <= settings.tolerance_m
            ):
                iterations_used = iteration
                break
            task_rows = np.zeros((3 * len(site_ids), len(variable)), dtype=np.float64)
            for row, site_id in enumerate(site_ids):
                mujoco.mj_jacSite(
                    model, data, jacobian_translation, jacobian_rotation, int(site_id)
                )
                task_rows[3 * row : 3 * row + 3] = jacobian_translation[
                    :, variable_dofs
                ]
            # Box-constrained active-set refinement.  A joint that reaches a
            # model bound keeps the exact bound step while the remaining free
            # joints re-solve the residual.  Do not only mark the joint active:
            # that would discard its proposed move and leave it at the old
            # value.
            fixed = np.zeros(len(variable), dtype=bool)
            delta = np.zeros(len(variable), dtype=np.float64)
            delta_lower = np.maximum(frame_lower - solution, -settings.max_step_rad)
            delta_upper = np.minimum(frame_upper - solution, settings.max_step_rad)
            for _ in range(len(variable)):
                free = ~fixed
                if not free.any():
                    break
                rows_free = task_rows[:, free]
                normal = rows_free.T @ weight_matrix @ rows_free
                normal += (posture_weight + settings.damping**2) * np.eye(
                    int(free.sum())
                )
                task_residual = errors.reshape(-1)
                if fixed.any():
                    task_residual = task_residual - task_rows[:, fixed] @ delta[fixed]
                gradient = rows_free.T @ weight_matrix @ task_residual
                gradient += posture_weight * (rest - solution)[free]
                if avoid_task.shape[0]:
                    avoid_free = avoid_task[:, free]
                    avoid_residual = avoid_error
                    if fixed.any():
                        avoid_residual = (
                            avoid_residual - avoid_task[:, fixed] @ delta[fixed]
                        )
                    normal += avoid_weight * (avoid_free.T @ avoid_free)
                    gradient += avoid_weight * (avoid_free.T @ avoid_residual)
                delta[free] = np.linalg.solve(normal, gradient)
                below = free & (delta < delta_lower)
                above = free & (delta > delta_upper)
                newly_fixed = below | above
                if not newly_fixed.any():
                    break
                delta[below] = delta_lower[below]
                delta[above] = delta_upper[above]
                fixed |= newly_fixed
            delta = np.clip(delta, delta_lower, delta_upper)
            current_objective = frame_objective(solution, errors, avoid_violation)
            accepted = False
            for line_search in range(int(settings.line_search_steps)):
                candidate = np.clip(
                    solution + (0.5**line_search) * delta,
                    frame_lower,
                    frame_upper,
                )
                base[variable_indices] = candidate
                data.qpos[trajectory_qpos_addresses] = base
                mujoco.mj_forward(model, data)
                candidate_errors = targets[frame] - data.site_xpos[site_ids]
                candidate_violation = avoidance_rows(frame)[2] if avoid_pairs else 0.0
                candidate_objective = frame_objective(
                    candidate, candidate_errors, candidate_violation
                )
                if candidate_objective < current_objective:
                    solution = candidate
                    accepted = True
                    break
            if not accepted:
                iterations_used = iteration + 1
                break
        base[variable_indices] = solution
        data.qpos[trajectory_qpos_addresses] = base
        mujoco.mj_forward(model, data)
        residuals = np.linalg.norm(targets[frame] - data.site_xpos[site_ids], axis=-1)
        return solution, residuals, iterations_used

    for frame in range(len(output)):
        frame_lower = lower
        frame_upper = upper
        if frame > 0 and settings.max_frame_change_rad is not None:
            previous = output[frame - 1, variable_indices]
            frame_lower = np.maximum(
                lower, previous - float(settings.max_frame_change_rad)
            )
            frame_upper = np.minimum(
                upper, previous + float(settings.max_frame_change_rad)
            )
        seed = (
            output[frame - 1, variable_indices]
            if frame > 0  # Warm start: keep the previous frame's solution.
            else output[frame, variable_indices]
        )
        solution, residuals, iterations_used = solve_frame(
            frame, seed, frame_lower, frame_upper
        )
        if frame == 0 and float(residuals.max()) > settings.tolerance_m:
            # The first frame has no previous solution.  Compare its supplied
            # seed with the slight-curl posture once, then use only sequential
            # warm starts for every later frame.
            retry_solution, retry_residuals, retry_iterations = solve_frame(
                frame, rest, frame_lower, frame_upper
            )
            if frame_objective(retry_solution, retry_residuals) < frame_objective(
                solution, residuals
            ):
                solution, residuals, iterations_used = (
                    retry_solution,
                    retry_residuals,
                    retry_iterations,
                )
        output[frame, variable_indices] = solution
        report.tip_errors_m[frame] = residuals
        report.iterations.append(iterations_used)
    return output, report


def audit_finger_joints(
    qpos: np.ndarray,
    joint_names: Sequence[str],
    limits: Mapping[str, tuple[float, float]],
) -> dict[str, Any]:
    """Joint-space audit of one side's finger trajectory (radians in).

    Same definitions as the reference ``finger_joint_audit`` tool: per-joint
    min/max against limits, per-joint fraction of frames within 2 deg of a
    limit, hyperextension beyond 5 deg on the interphalangeal joints, maximum
    frame-to-frame jump, DIP-vs-PIP coupling slope, and the worst frames by
    summed hyperextension.
    """

    values = np.asarray(qpos, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(joint_names):
        raise ValueError("qpos must have shape [T, len(joint_names)].")
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("qpos must contain at least one finite frame.")
    frame_count = len(values)
    rows: list[dict[str, Any]] = []
    hyper_sum = np.zeros(frame_count, dtype=np.float64)
    for column, name in enumerate(joint_names):
        series = values[:, column]
        lower, upper = limits.get(name, (np.nan, np.nan))
        jump = float(np.max(np.abs(np.diff(series)))) if frame_count > 1 else 0.0
        row: dict[str, Any] = {
            "joint": name,
            "min_deg": round(float(np.degrees(series.min())), 1),
            "max_deg": round(float(np.degrees(series.max())), 1),
            "limit_deg": [
                round(float(np.degrees(lower)), 1),
                round(float(np.degrees(upper)), 1),
            ],
            "pct_near_lower": (
                round(100.0 * float(np.mean(series <= lower + np.radians(2.0))), 1)
                if np.isfinite(lower)
                else None
            ),
            "pct_near_upper": (
                round(100.0 * float(np.mean(series >= upper - np.radians(2.0))), 1)
                if np.isfinite(upper)
                else None
            ),
            "max_jump_deg": round(float(np.degrees(jump)), 1),
            "max_jump_deg_raw": float(np.degrees(jump)),
        }
        if _match_key(name, HYPEREXTENSION_JOINT_KEYS) is not None:
            row["pct_hyperext_gt5deg"] = round(
                100.0 * float(np.mean(series < -np.radians(5.0))), 1
            )
            row["min_hyperext_deg"] = round(float(np.degrees(series.min())), 1)
            hyper_sum += np.clip(-series, 0.0, None)
        rows.append(row)

    coupling: dict[str, dict[str, float]] = {}
    index_of = {name: column for column, name in enumerate(joint_names)}
    pairs = [
        (name, name.replace("_pip", "_dip"))
        for name in joint_names
        if name.endswith("_pip")
    ] + [
        (name, name.replace("thumb_mcp", "thumb_ip"))
        for name in joint_names
        if name.endswith("thumb_mcp")
    ]
    for pip_name, dip_name in pairs:
        if pip_name not in index_of or dip_name not in index_of:
            continue
        pip_values = values[:, index_of[pip_name]]
        dip_values = values[:, index_of[dip_name]]
        mask = pip_values > np.radians(10.0)
        # The slope is undefined when a finger never flexes past 10 deg.
        # Report None rather than NaN: this record is serialized into
        # Reference metadata, which must stay JSON-compatible and finite.
        slope = (
            round(
                float(
                    np.sum(pip_values[mask] * dip_values[mask])
                    / max(float(np.sum(pip_values[mask] ** 2)), 1.0e-9)
                ),
                2,
            )
            if mask.any()
            else None
        )
        within = float(
            np.mean(np.abs(dip_values - 0.7 * pip_values) < np.radians(15.0))
        )
        coupling[dip_name] = {
            "slope_dip_vs_pip": slope,
            "flexed_frame_count": int(np.count_nonzero(mask)),
            "pct_within_15deg_of_0.7pip": round(100.0 * within, 1),
        }

    worst = np.argsort(-hyper_sum)[:10]
    hyperextension_pct_raw = 100.0 * float(np.mean(hyper_sum > np.radians(5.0)))
    maximum_jump_deg_raw = max(
        (float(row["max_jump_deg_raw"]) for row in rows), default=0.0
    )
    return {
        "num_frames": frame_count,
        "joints": rows,
        "coupling_dip_vs_pip": coupling,
        "worst_frames_by_hyperextension": [
            {
                "frame": int(frame),
                "sum_hyperext_deg": round(float(np.degrees(hyper_sum[frame])), 1),
            }
            for frame in worst
            if np.degrees(hyper_sum[frame]) > 0.05
        ],
        "pct_frames_any_hyperext_gt5deg": round(hyperextension_pct_raw, 1),
        "pct_frames_any_hyperext_gt5deg_raw": hyperextension_pct_raw,
        "max_jump_deg": round(maximum_jump_deg_raw, 1),
        "max_jump_deg_raw": maximum_jump_deg_raw,
    }


def evaluate_finger_gates(
    *,
    audit: Mapping[str, Any],
    tip_p95_mm: float | None = None,
    wrist_mean_mm: float | None = None,
    gates: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Apply the pre-registered gates to one side's audit and residuals.

    ``tip_p95_mm`` and ``wrist_mean_mm`` are optional because an offline audit
    of a stored trajectory has no IK targets; a gate without its measurement
    reports as ``not_measured`` instead of passing silently.
    """

    limits = dict(WUJI_FINGER_GATES if gates is None else gates)
    values: dict[str, Any] = {
        "hyperextension_pct": float(
            audit.get(
                "pct_frames_any_hyperext_gt5deg_raw",
                audit["pct_frames_any_hyperext_gt5deg"],
            )
        ),
        "jump_max_deg": float(audit.get("max_jump_deg_raw", audit["max_jump_deg"])),
        "tip_p95_mm": None if tip_p95_mm is None else float(tip_p95_mm),
        "wrist_mean_mm": None if wrist_mean_mm is None else float(wrist_mean_mm),
    }
    failures: list[str] = []
    not_measured: list[str] = []
    for name, limit in limits.items():
        if name not in values:
            raise KeyError(f"Unknown Wuji finger gate: {name!r}.")
        if not np.isfinite(limit) or limit < 0.0:
            raise ValueError(f"Gate {name!r} must be finite and non-negative.")
        value = values.get(name)
        if value is None:
            not_measured.append(name)
        elif not np.isfinite(value):
            failures.append(name)
        elif value > limit:
            failures.append(name)
    return {
        "pass": not failures and not not_measured,
        "failures": failures,
        "not_measured": not_measured,
        "values": values,
        "gates": limits,
    }


def wuji_knuckle_positions_palm_frame(model: Any, side: str) -> np.ndarray:
    """Return the four MCP-flex joint anchors in the palm-site frame at q=0."""

    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'.")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    palm_id = int(model.site(f"{side}_palm").id)
    palm_position = data.site_xpos[palm_id]
    palm_rotation = data.site_xmat[palm_id].reshape(3, 3)
    anchors = np.asarray(
        [
            data.xanchor[int(model.joint(name).id)]
            for name in WUJI_KNUCKLE_JOINT_NAMES[side]
        ],
        dtype=np.float64,
    )
    return (anchors - palm_position) @ palm_rotation


def wuji_finger_directions_palm_frame(model: Any, side: str) -> np.ndarray:
    """Return the four unit MCP->PIP directions in the palm frame at q=0.

    This is the robot's zero-pose finger fan. Aligning the mapped source
    finger fan with it is the notes' "definition B" of palm alignment, and it
    is what determines how much MCP abduction the retarget must spend.
    """

    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'.")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    palm_rotation = data.site_xmat[int(model.site(f"{side}_palm").id)].reshape(3, 3)
    directions = []
    for mcp_name, pip_name in zip(
        WUJI_KNUCKLE_JOINT_NAMES[side], WUJI_PIP_JOINT_NAMES[side]
    ):
        segment = (
            data.xanchor[int(model.joint(pip_name).id)]
            - data.xanchor[int(model.joint(mcp_name).id)]
        )
        norm = float(np.linalg.norm(segment))
        if norm <= 1.0e-9:
            raise ValueError(f"{side} {mcp_name} to {pip_name} segment is degenerate.")
        directions.append(segment / norm)
    return np.asarray(directions, dtype=np.float64) @ palm_rotation


def fit_palm_yaw_from_finger_directions(
    source_directions: np.ndarray,
    robot_directions: np.ndarray,
    *,
    palm_normal_axis: int = 1,
    minimum_in_plane_fraction: float = 0.4,
) -> dict[str, Any]:
    """Palm yaw that centres the MCP abduction demand (notes' definition B).

    Aligning knuckle *positions* (definition A) fixes where the fingers start;
    it does not fix which way they point. When every finger of both hands sits
    at the same-signed abduction limit, the residual is a palm-normal rotation,
    not finger spreading: real spreading pushes index and pinky in opposite
    directions.

    ``source_directions`` is ``[T, 4, 3]`` unit MCP->PIP directions already
    expressed in the knuckle-calibrated palm frame; ``robot_directions`` is the
    robot's ``[4, 3]`` zero-pose fan. Frames whose projection into the palm
    plane is shorter than ``minimum_in_plane_fraction`` of the segment are
    dropped: a finger curled through the palm normal has no meaningful in-plane
    angle. The returned yaw is the mean over fingers of the per-finger median
    demand, so one finger's posture cannot dominate.
    """

    source = np.asarray(source_directions, dtype=np.float64)
    robot = np.asarray(robot_directions, dtype=np.float64)
    if source.ndim != 3 or source.shape[1:] != (4, 3) or len(source) == 0:
        raise ValueError("source_directions must have shape [T, 4, 3] with T > 0.")
    if robot.shape != (4, 3):
        raise ValueError("robot_directions must have shape [4, 3].")
    if not np.isfinite(source).all() or not np.isfinite(robot).all():
        raise ValueError("Finger directions must be finite.")
    if palm_normal_axis not in _PALM_PLANE_ANGLE_AXES:
        raise ValueError("palm_normal_axis must be 0, 1, or 2.")
    if not 0.0 < minimum_in_plane_fraction < 1.0:
        raise ValueError("minimum_in_plane_fraction must be inside (0, 1).")

    first, second = _PALM_PLANE_ANGLE_AXES[int(palm_normal_axis)]

    def angle(vectors: np.ndarray) -> np.ndarray:
        return np.degrees(np.arctan2(vectors[..., first], vectors[..., second]))

    robot_angle = angle(robot)
    lengths = np.linalg.norm(source, axis=-1)
    in_plane = np.hypot(source[..., first], source[..., second])
    usable = in_plane > minimum_in_plane_fraction * np.maximum(lengths, 1.0e-12)
    demand = angle(source) - robot_angle[None, :]
    demand = (demand + 180.0) % 360.0 - 180.0

    per_finger: list[float] = []
    counts: list[int] = []
    for finger in range(4):
        values = demand[usable[:, finger], finger]
        counts.append(int(values.size))
        if values.size == 0:
            raise ValueError(
                "No frame has a usable in-plane finger direction; the palm "
                "yaw demand cannot be measured."
            )
        per_finger.append(float(np.median(values)))
    yaw_deg = float(np.mean(per_finger))
    return {
        "yaw_deg": yaw_deg,
        "per_finger_median_demand_deg": [round(value, 2) for value in per_finger],
        "per_finger_residual_after_yaw_deg": [
            round(value - yaw_deg, 2) for value in per_finger
        ],
        "per_finger_usable_frame_count": counts,
        "robot_zero_pose_finger_angles_deg": [
            round(float(value), 2) for value in robot_angle
        ],
        "definition": "finger_direction_abduction_demand",
    }


def fit_palm_frame_from_knuckles(
    source_knuckles: np.ndarray,
    robot_knuckles: np.ndarray,
    *,
    palm_normal_axis: int = 1,
    yaw_grid_deg: tuple[float, float, float] = (-40.0, 40.0, 0.1),
) -> dict[str, float]:
    """Measure the palm-frame yaw and normal offset from geometry only.

    Both point sets are the four index-to-pinky knuckles expressed in the same
    nominal wrist frame (the source points already through the candidate axis
    permutation).  The fit is a planar Procrustes in the palm plane with the
    origin pinned at the wrist and a free uniform scale for hand size: the
    returned yaw is the rotation about the palm normal that best overlays the
    robot knuckles on the source knuckles.  The returned palm-normal offset is
    the mean out-of-plane separation the wrist target must shift to overlay
    the knuckle planes.  Never estimate these numbers from IK behaviour: a
    behavioural estimate measures finger spreading, not alignment.
    """

    source = np.asarray(source_knuckles, dtype=np.float64)
    robot = np.asarray(robot_knuckles, dtype=np.float64)
    if source.shape != (4, 3) or robot.shape != (4, 3):
        raise ValueError("Knuckle point sets must have shape [4, 3].")
    if not np.isfinite(source).all() or not np.isfinite(robot).all():
        raise ValueError("Knuckle point sets must be finite.")
    if palm_normal_axis not in (0, 1, 2):
        raise ValueError("palm_normal_axis must be 0, 1, or 2.")
    plane = [axis for axis in range(3) if axis != palm_normal_axis]
    start, stop, step = yaw_grid_deg
    best: dict[str, float] | None = None
    for yaw_deg in np.arange(start, stop + 0.5 * step, step):
        rotation = _axis_rotation(palm_normal_axis, float(np.radians(yaw_deg)))
        rotated = robot @ rotation.T
        numerator = float((rotated[:, plane] * source[:, plane]).sum())
        denominator = float((rotated[:, plane] ** 2).sum())
        if denominator <= 0.0:
            raise ValueError("Robot knuckles are degenerate in the palm plane.")
        scale = numerator / denominator
        if scale <= 0.0:
            continue
        residual = float(
            np.sqrt(
                ((scale * rotated[:, plane] - source[:, plane]) ** 2).sum(axis=1).mean()
            )
        )
        if best is None or residual < best["knuckle_rms_m"]:
            best = {
                "yaw_deg": float(yaw_deg),
                "knuckle_rms_m": residual,
                "scale": scale,
            }
    if best is None:
        raise ValueError("No positive-scale palm-frame fit exists.")
    normal_offset_m = float(
        (source[:, palm_normal_axis] - robot[:, palm_normal_axis]).mean()
    )
    best["palm_normal_offset_m"] = normal_offset_m
    best["knuckle_rms_mm"] = best["knuckle_rms_m"] * 1e3
    best["palm_normal_offset_mm"] = normal_offset_m * 1e3
    return best


def fit_source_palm_frame_from_knuckles(
    source_knuckles: np.ndarray,
    robot_knuckles: np.ndarray,
    *,
    palm_normal_axis: int = 1,
    yaw_grid_deg: tuple[float, float, float] = (-40.0, 40.0, 0.1),
) -> dict[str, Any]:
    """Fit a source-wrist to Wuji-palm correction from rigid knuckles.

    ``source_knuckles`` has shape ``[T, 4, 3]`` in the source wrist frame and
    is ordered index, middle, ring, pinky.  The source axes are inferred from
    geometry: index-to-pinky is the robot ``+x`` direction and the wrist-to-
    knuckle direction is robot ``-z``.  The remaining proper axis is the palm
    normal.  A planar yaw fit then aligns the four points with the Wuji MCP
    origins.  The fitted scale is diagnostic only and is never part of the
    returned source-to-robot correction.

    Row-vector application is explicit: ``v_robot = v_source @ correction``.
    Subtract ``palm_normal_offset_m`` from the mapped palm-normal component
    when the Wuji wrist target and the human anatomical wrist denote different
    points.
    """

    source = np.asarray(source_knuckles, dtype=np.float64)
    robot = np.asarray(robot_knuckles, dtype=np.float64)
    if source.ndim != 3 or source.shape[1:] != (4, 3) or len(source) == 0:
        raise ValueError("source_knuckles must have shape [T, 4, 3] with T > 0.")
    if robot.shape != (4, 3):
        raise ValueError("robot_knuckles must have shape [4, 3].")
    if not np.isfinite(source).all() or not np.isfinite(robot).all():
        raise ValueError("Knuckle point sets must be finite.")

    mean_source = source.mean(axis=0)
    thumbward = mean_source[0] - mean_source[-1]
    thumbward_norm = float(np.linalg.norm(thumbward))
    if thumbward_norm <= 1.0e-8:
        raise ValueError("Source index-to-pinky knuckle span is degenerate.")
    source_x = thumbward / thumbward_norm
    source_z = -mean_source.mean(axis=0)
    source_z = source_z - source_x * float(source_z @ source_x)
    source_z_norm = float(np.linalg.norm(source_z))
    if source_z_norm <= 1.0e-8:
        raise ValueError("Source wrist-to-knuckle direction is degenerate.")
    source_z /= source_z_norm
    source_y = np.cross(source_z, source_x)
    source_y /= np.linalg.norm(source_y)
    nominal_correction = np.column_stack((source_x, source_y, source_z))
    if float(np.linalg.det(nominal_correction)) < 1.0 - 1.0e-8:
        raise RuntimeError("Source palm basis is not a proper rotation.")

    nominal_source = mean_source @ nominal_correction
    fit = fit_palm_frame_from_knuckles(
        nominal_source,
        robot,
        palm_normal_axis=palm_normal_axis,
        yaw_grid_deg=yaw_grid_deg,
    )
    yaw_rotation = _axis_rotation(palm_normal_axis, float(np.radians(fit["yaw_deg"])))
    correction = nominal_correction @ yaw_rotation
    mapped_source = mean_source @ correction
    normal_offset_m = float(
        (mapped_source[:, palm_normal_axis] - robot[:, palm_normal_axis]).mean()
    )
    return {
        **fit,
        "source_to_robot_rotation": correction.tolist(),
        "nominal_source_to_robot_rotation": nominal_correction.tolist(),
        "palm_normal_axis": int(palm_normal_axis),
        "palm_normal_offset_m": normal_offset_m,
        "palm_normal_offset_mm": normal_offset_m * 1e3,
        "source_knuckle_rigidity_std_mm": float(source.std(axis=0).max() * 1e3),
        "source_knuckle_mean_m": mean_source.tolist(),
        "robot_knuckles_m": robot.tolist(),
        "scale_applied_to_targets": 1.0,
    }


def measure_fingertip_residuals(
    model: Any,
    *,
    trajectory_joint_names: Sequence[str],
    qpos: np.ndarray,
    target_site_names: Sequence[str],
    target_positions: np.ndarray,
) -> FingertipIkReport:
    """Measure direct world-position residuals without changing the trajectory."""

    values = np.asarray(qpos, dtype=np.float64)
    targets = np.asarray(target_positions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(trajectory_joint_names):
        raise ValueError("qpos must have shape [T, len(trajectory_joint_names)].")
    if targets.shape != (len(values), len(target_site_names), 3):
        raise ValueError("target_positions must have shape [T, n_sites, 3].")
    if (
        len(values) == 0
        or not np.isfinite(values).all()
        or not np.isfinite(targets).all()
    ):
        raise ValueError("qpos and target_positions must contain finite frames.")

    addresses = np.asarray(
        [
            int(model.jnt_qposadr[model.joint(name).id])
            for name in trajectory_joint_names
        ],
        dtype=np.int32,
    )
    site_ids = np.asarray(
        [int(model.site(name).id) for name in target_site_names], dtype=np.int32
    )
    report = FingertipIkReport(
        frame_count=len(values),
        site_names=tuple(target_site_names),
        tip_errors_m=np.zeros((len(values), len(site_ids)), dtype=np.float64),
    )
    data = mujoco.MjData(model)
    for frame, row in enumerate(values):
        mujoco.mj_resetData(model, data)
        data.qpos[addresses] = row
        mujoco.mj_forward(model, data)
        report.tip_errors_m[frame] = np.linalg.norm(
            targets[frame] - data.site_xpos[site_ids], axis=-1
        )
    return report


def _axis_rotation(axis: int, angle_rad: float) -> np.ndarray:
    cos, sin = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    if axis == 0:
        return np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]])
    if axis == 1:
        return np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
