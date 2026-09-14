"""Tests for the Wuji finger retargeting recipe (limits, IK, audit, gates).

The synthetic models reproduce the two failure modes the recipe exists to
remove: a Z-folded (hyperextended) finger that reaches its fingertip target
exactly, and a redundant finger that settles at a joint limit instead of
mid-range.  The real-asset test runs only when the Vega-Wuji MJCF and its
meshes are available.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import mujoco

from iltools.retarget.wuji_finger import (
    FingertipAvoidanceConfig,
    FingertipIkConfig,
    WUJI_FINGER_GATES,
    apply_anatomical_finger_limits,
    audit_finger_joints,
    evaluate_finger_gates,
    fit_palm_frame_from_knuckles,
    fit_palm_yaw_from_finger_directions,
    fit_source_palm_frame_from_knuckles,
    measure_fingertip_residuals,
    posture_rest_qpos,
    solve_fingertip_trajectory,
    wuji_finger_directions_palm_frame,
    wuji_knuckle_positions_palm_frame,
)
from iltools.retarget.wuji_finger import _axis_rotation


VEGA_WUJI_MODEL = (
    Path(__file__).resolve().parents[3]
    / "source/isaaclab_imitation/isaaclab_imitation/assets/vega_wuji"
    / "vega_u_wuji_v2_beta1_with_mount.xml"
)

TWO_LINK_FINGER_XML = """
<mujoco>
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="proximal" pos="0 0 0">
      <joint name="r_test_finger_pip" type="hinge" axis="1 0 0"
             range="-1.047 2.094" limited="true"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.04" size="0.004"/>
      <body name="distal" pos="0 0 -0.04">
        <joint name="r_test_finger_dip" type="hinge" axis="1 0 0"
               range="-1.047 1.57" limited="true"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.03" size="0.004"/>
        <site name="r_test_finger_tip" pos="0 0 -0.03"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

THREE_LINK_FINGER_XML = """
<mujoco>
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="proximal" pos="0 0 0">
      <joint name="r_test_finger_mcp_flex" type="hinge" axis="1 0 0"
             range="-1.047 1.57" limited="true"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.045" size="0.004"/>
      <body name="middle" pos="0 0 -0.045">
        <joint name="r_test_finger_pip" type="hinge" axis="1 0 0"
               range="-1.047 2.094" limited="true"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.03" size="0.004"/>
        <body name="distal" pos="0 0 -0.03">
          <joint name="r_test_finger_dip" type="hinge" axis="1 0 0"
                 range="-1.047 1.57" limited="true"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.025" size="0.004"/>
          <site name="r_test_finger_tip" pos="0 0 -0.025"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _fingertip_at(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(model)
    data.qpos[: len(qpos)] = qpos
    mujoco.mj_forward(model, data)
    return data.site_xpos[model.site("r_test_finger_tip").id].copy()


def test_anatomical_limits_tighten_in_place_with_rows() -> None:
    model = mujoco.MjModel.from_xml_string(TWO_LINK_FINGER_XML)
    rows = apply_anatomical_finger_limits(model)
    assert [row["joint"] for row in rows] == [
        "r_test_finger_pip",
        "r_test_finger_dip",
    ]
    pip_range = model.jnt_range[model.joint("r_test_finger_pip").id]
    dip_range = model.jnt_range[model.joint("r_test_finger_dip").id]
    assert pip_range[0] == pytest.approx(0.0)
    assert np.degrees(pip_range[1]) == pytest.approx(100.0, abs=0.1)
    assert dip_range[0] == pytest.approx(0.0)
    assert np.degrees(dip_range[1]) == pytest.approx(80.0, abs=0.1)


def test_anatomical_limits_only_tighten() -> None:
    model = mujoco.MjModel.from_xml_string(TWO_LINK_FINGER_XML)
    rows = apply_anatomical_finger_limits(
        model, limits_deg={"_pip": (-90.0, 500.0), "_dip": (None, None)}
    )
    pip_range = model.jnt_range[model.joint("r_test_finger_pip").id]
    assert np.degrees(pip_range[0]) == pytest.approx(-60.0, abs=0.1)
    assert np.degrees(pip_range[1]) == pytest.approx(120.0, abs=0.1)
    assert [row["joint"] for row in rows] == [
        "r_test_finger_pip",
        "r_test_finger_dip",
    ]


def test_tips_only_ik_z_fold_removed_by_limit_clamp() -> None:
    joint_names = ("r_test_finger_pip", "r_test_finger_dip")
    natural = np.array([0.7, 0.5])
    config = FingertipIkConfig(
        iterations=200, tip_weights=(1.0,), posture_cost=0.0, tolerance_m=1.0e-5
    )
    # A seed on the inverted branch keeps an unclamped solver hyperextended
    # even though the fingertip lands exactly on target.
    hyperextended_seed = np.array([[1.35, -0.62]])

    unclamped = mujoco.MjModel.from_xml_string(TWO_LINK_FINGER_XML)
    target = _fingertip_at(unclamped, natural)[None, None, :]
    solved_unclamped, report_unclamped = solve_fingertip_trajectory(
        unclamped,
        trajectory_joint_names=joint_names,
        qpos=hyperextended_seed,
        variable_joint_names=joint_names,
        target_site_names=("r_test_finger_tip",),
        target_positions=target,
        config=config,
    )
    assert float(report_unclamped.tip_errors_m.max()) < 2.0e-3
    assert np.degrees(solved_unclamped[0, 1]) < -5.0  # Z-fold: DIP hyperextended.

    clamped = mujoco.MjModel.from_xml_string(TWO_LINK_FINGER_XML)
    apply_anatomical_finger_limits(clamped)
    solved_clamped, report_clamped = solve_fingertip_trajectory(
        clamped,
        trajectory_joint_names=joint_names,
        qpos=hyperextended_seed,
        variable_joint_names=joint_names,
        target_site_names=("r_test_finger_tip",),
        target_positions=target,
        config=config,
    )
    assert float(report_clamped.tip_errors_m.max()) < 2.0e-3
    assert np.all(solved_clamped[0] >= -1.0e-9)
    assert np.allclose(solved_clamped[0], natural, atol=5.0e-2)


def test_posture_prior_settles_redundancy_mid_range() -> None:
    joint_names = (
        "r_test_finger_mcp_flex",
        "r_test_finger_pip",
        "r_test_finger_dip",
    )
    model = mujoco.MjModel.from_xml_string(THREE_LINK_FINGER_XML)
    apply_anatomical_finger_limits(model)
    rest = posture_rest_qpos(joint_names)
    target = _fingertip_at(model, rest)[None, None, :]
    seed = np.zeros((1, 3))

    with_prior, report_prior = solve_fingertip_trajectory(
        model,
        trajectory_joint_names=joint_names,
        qpos=seed,
        variable_joint_names=joint_names,
        target_site_names=("r_test_finger_tip",),
        target_positions=target,
        config=FingertipIkConfig(iterations=300, tip_weights=(1.0,)),
    )
    without_prior, report_free = solve_fingertip_trajectory(
        model,
        trajectory_joint_names=joint_names,
        qpos=seed,
        variable_joint_names=joint_names,
        target_site_names=("r_test_finger_tip",),
        target_positions=target,
        config=FingertipIkConfig(iterations=300, tip_weights=(1.0,), posture_cost=0.0),
    )
    assert float(report_prior.tip_errors_m.max()) < 1.0e-3
    assert float(report_free.tip_errors_m.max()) < 1.0e-3
    distance_with = float(np.linalg.norm(with_prior[0] - rest))
    distance_without = float(np.linalg.norm(without_prior[0] - rest))
    assert distance_with < distance_without


def test_posture_prior_still_acts_from_an_exact_tip_seed() -> None:
    joint_names = (
        "r_test_finger_mcp_flex",
        "r_test_finger_pip",
        "r_test_finger_dip",
    )
    model = mujoco.MjModel.from_xml_string(THREE_LINK_FINGER_XML)
    apply_anatomical_finger_limits(model)
    seed = np.asarray(((0.8, 0.15, 0.15),))
    target = _fingertip_at(model, seed[0])[None, None, :]
    rest = posture_rest_qpos(joint_names)

    solved, report = solve_fingertip_trajectory(
        model,
        trajectory_joint_names=joint_names,
        qpos=seed,
        variable_joint_names=joint_names,
        target_site_names=("r_test_finger_tip",),
        target_positions=target,
        config=FingertipIkConfig(iterations=300, tip_weights=(1.0,)),
    )

    assert report.iterations[0] > 0
    assert float(report.tip_errors_m.max()) < 1.0e-3
    assert np.linalg.norm(solved[0] - rest) < np.linalg.norm(seed[0] - rest)


def test_warm_start_keeps_frames_continuous() -> None:
    joint_names = ("r_test_finger_pip", "r_test_finger_dip")
    model = mujoco.MjModel.from_xml_string(TWO_LINK_FINGER_XML)
    apply_anatomical_finger_limits(model)
    poses = np.stack(
        [np.array([0.3 + 0.02 * step, 0.2 + 0.015 * step]) for step in range(20)]
    )
    targets = np.stack([_fingertip_at(model, pose) for pose in poses])[:, None, :]
    solved, report = solve_fingertip_trajectory(
        model,
        trajectory_joint_names=joint_names,
        qpos=np.zeros((len(poses), 2)),
        variable_joint_names=joint_names,
        target_site_names=("r_test_finger_tip",),
        target_positions=targets,
        config=FingertipIkConfig(iterations=200, tip_weights=(1.0,)),
    )
    assert float(report.tip_errors_m.max()) < 2.0e-3
    jumps = np.degrees(np.abs(np.diff(solved, axis=0)))
    assert float(jumps.max()) < 10.0


def test_trajectory_velocity_limit_bounds_an_abrupt_target() -> None:
    joint_names = ("r_test_finger_pip", "r_test_finger_dip")
    model = mujoco.MjModel.from_xml_string(TWO_LINK_FINGER_XML)
    apply_anatomical_finger_limits(model)
    targets = np.stack(
        (
            _fingertip_at(model, np.zeros(2)),
            _fingertip_at(model, np.array([1.5, 1.2])),
        )
    )[:, None, :]
    solved, _ = solve_fingertip_trajectory(
        model,
        trajectory_joint_names=joint_names,
        qpos=np.zeros((2, 2)),
        variable_joint_names=joint_names,
        target_site_names=("r_test_finger_tip",),
        target_positions=targets,
        config=FingertipIkConfig(
            iterations=200,
            max_frame_change_rad=0.1,
            tip_weights=(1.0,),
            posture_cost=0.0,
        ),
    )
    assert float(np.max(np.abs(np.diff(solved, axis=0)))) <= 0.1 + 1.0e-12


def test_bound_active_set_commits_the_bound_step() -> None:
    joint_names = ("r_test_finger_pip", "r_test_finger_dip")
    model = mujoco.MjModel.from_xml_string(TWO_LINK_FINGER_XML)
    apply_anatomical_finger_limits(model)
    upper_pip = float(model.jnt_range[model.joint(joint_names[0]).id, 1])
    target = _fingertip_at(model, np.asarray((upper_pip, 0.0)))[None, None, :]
    solved, report = solve_fingertip_trajectory(
        model,
        trajectory_joint_names=joint_names,
        qpos=np.zeros((1, 2)),
        variable_joint_names=(joint_names[0],),
        target_site_names=("r_test_finger_tip",),
        target_positions=target,
        config=FingertipIkConfig(
            iterations=200,
            max_step_rad=0.15,
            tip_weights=(1.0,),
            posture_cost=0.0,
            tolerance_m=1.0e-6,
        ),
    )
    assert solved[0, 0] == pytest.approx(upper_pip, abs=2.0e-3)
    assert float(report.tip_errors_m.max()) < 1.0e-4


def test_audit_reports_hyperextension_jump_and_coupling() -> None:
    joint_names = ("r_index_finger_pip", "r_index_finger_dip")
    limits = {name: (-1.047, 2.094) for name in joint_names}
    qpos = np.zeros((10, 2))
    qpos[:, 0] = 0.5
    qpos[:, 1] = 0.35
    qpos[4, 1] = -0.5  # one hyperextended frame with a large jump
    audit = audit_finger_joints(qpos, joint_names, limits)
    assert audit["num_frames"] == 10
    assert audit["pct_frames_any_hyperext_gt5deg"] == pytest.approx(10.0)
    assert audit["max_jump_deg"] > 35.0
    dip_row = next(row for row in audit["joints"] if row["joint"].endswith("_dip"))
    assert dip_row["pct_hyperext_gt5deg"] == pytest.approx(10.0)
    assert audit["worst_frames_by_hyperextension"][0]["frame"] == 4
    assert "r_index_finger_dip" in audit["coupling_dip_vs_pip"]


def test_audit_stays_json_serializable_for_an_unflexed_finger() -> None:
    """An extended finger has no coupling slope; it must not emit NaN.

    The audit is embedded in Reference metadata, which rejects non-finite
    values, so an all-extended hand used to abort the whole conversion.
    """

    joint_names = ("r_index_finger_pip", "r_index_finger_dip")
    limits = {name: (0.0, 2.094) for name in joint_names}
    qpos = np.full((6, 2), np.radians(2.0))  # never flexes past 10 deg
    audit = audit_finger_joints(qpos, joint_names, limits)
    coupling = audit["coupling_dip_vs_pip"]["r_index_finger_dip"]
    assert coupling["slope_dip_vs_pip"] is None
    assert coupling["flexed_frame_count"] == 0
    encoded = json.dumps(audit, allow_nan=False)
    assert "NaN" not in encoded


def test_gates_use_raw_values_instead_of_rounded_presentation() -> None:
    joint_names = ("r_index_finger_pip",)
    qpos = np.radians(np.asarray(((0.0,), (35.04,))))
    audit = audit_finger_joints(
        qpos,
        joint_names,
        {joint_names[0]: (0.0, np.radians(100.0))},
    )
    assert audit["max_jump_deg"] == 35.0
    assert audit["max_jump_deg_raw"] == pytest.approx(35.04)
    gates = evaluate_finger_gates(
        audit=audit,
        tip_p95_mm=0.0,
        wrist_mean_mm=0.0,
    )
    assert gates["pass"] is False
    assert gates["failures"] == ["jump_max_deg"]


def test_gates_fail_and_not_measured_semantics() -> None:
    clean = {"pct_frames_any_hyperext_gt5deg": 0.0, "max_jump_deg": 12.0}
    result = evaluate_finger_gates(audit=clean, tip_p95_mm=3.2, wrist_mean_mm=1.1)
    assert result["pass"] is True
    assert result["failures"] == []

    result = evaluate_finger_gates(audit=clean, tip_p95_mm=3.2)
    assert result["pass"] is False
    assert result["not_measured"] == ["wrist_mean_mm"]

    dirty = {"pct_frames_any_hyperext_gt5deg": 4.0, "max_jump_deg": 80.0}
    result = evaluate_finger_gates(audit=dirty, tip_p95_mm=9.0, wrist_mean_mm=4.0)
    assert result["pass"] is False
    assert set(result["failures"]) == {
        "hyperextension_pct",
        "jump_max_deg",
        "tip_p95_mm",
        "wrist_mean_mm",
    }
    assert set(result["gates"]) == set(WUJI_FINGER_GATES)


def test_palm_frame_fit_recovers_injected_yaw_scale_and_offset() -> None:
    robot = np.array(
        [
            [0.028, 0.004, -0.085],
            [0.009, 0.005, -0.092],
            [-0.010, 0.004, -0.089],
            [-0.028, 0.003, -0.080],
        ]
    )
    yaw = np.radians(7.0)
    cos, sin = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])
    source = 1.1 * (robot @ rotation.T)
    source[:, 1] += 0.0096
    fit = fit_palm_frame_from_knuckles(source, robot, palm_normal_axis=1)
    assert fit["yaw_deg"] == pytest.approx(7.0, abs=0.2)
    assert fit["scale"] == pytest.approx(1.1, abs=0.01)
    assert fit["knuckle_rms_mm"] < 0.5
    assert fit["palm_normal_offset_mm"] == pytest.approx(
        float((source[:, 1] - robot[:, 1]).mean() * 1e3), abs=0.01
    )


def test_source_palm_fit_infers_axes_and_never_applies_scale() -> None:
    robot = np.array(
        [
            [0.028, 0.004, -0.085],
            [0.009, 0.005, -0.092],
            [-0.010, 0.004, -0.089],
            [-0.028, 0.003, -0.080],
        ]
    )
    source_rotation = Rotation.from_euler("xyz", (35.0, -20.0, 70.0), degrees=True)
    source = source_rotation.inv().apply(robot)
    source = np.repeat(source[None, :, :], 8, axis=0)
    fit = fit_source_palm_frame_from_knuckles(source, robot)
    correction = np.asarray(fit["source_to_robot_rotation"])
    np.testing.assert_allclose(correction.T @ correction, np.eye(3), atol=1.0e-10)
    assert np.linalg.det(correction) == pytest.approx(1.0)
    mapped = source[0] @ correction
    mapped[:, int(fit["palm_normal_axis"])] -= fit["palm_normal_offset_m"]
    np.testing.assert_allclose(mapped, robot, atol=1.0e-3)
    assert fit["knuckle_rms_mm"] < 1.0
    assert fit["source_knuckle_rigidity_std_mm"] < 1.0e-9
    assert fit["scale_applied_to_targets"] == 1.0


def test_measure_fingertip_residuals_does_not_change_qpos() -> None:
    joint_names = ("r_test_finger_pip", "r_test_finger_dip")
    model = mujoco.MjModel.from_xml_string(TWO_LINK_FINGER_XML)
    qpos = np.asarray(((0.2, 0.3), (0.4, 0.5)))
    targets = np.stack([_fingertip_at(model, row) for row in qpos])[:, None, :]
    before = qpos.copy()
    report = measure_fingertip_residuals(
        model,
        trajectory_joint_names=joint_names,
        qpos=qpos,
        target_site_names=("r_test_finger_tip",),
        target_positions=targets,
    )
    np.testing.assert_array_equal(qpos, before)
    np.testing.assert_allclose(report.tip_errors_m, 0.0, atol=1.0e-12)


def _load_vega_wuji_model() -> mujoco.MjModel | None:
    if not VEGA_WUJI_MODEL.is_file():
        return None
    try:
        return mujoco.MjModel.from_xml_path(str(VEGA_WUJI_MODEL))
    except Exception:  # noqa: BLE001 - missing LFS meshes must skip, not fail.
        return None


def test_real_asset_limits_knuckles_and_rest_pose_ik() -> None:
    model = _load_vega_wuji_model()
    if model is None:
        pytest.skip("Vega-Wuji MJCF or its meshes are not available.")
    rows = apply_anatomical_finger_limits(model)
    # Per side: 4 MCP abductions, 4 PIP, 4 DIP, thumb MCP, thumb IP.
    assert len(rows) == 28
    pip_range = model.jnt_range[model.joint("r_index_finger_pip").id]
    assert pip_range[0] == pytest.approx(0.0)
    assert np.degrees(pip_range[1]) == pytest.approx(100.0, abs=0.1)

    knuckles = wuji_knuckle_positions_palm_frame(model, "right")
    assert knuckles.shape == (4, 3)
    span = float(np.linalg.norm(knuckles[0] - knuckles[3]))
    assert 0.03 < span < 0.15

    joint_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
        and str(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        ).startswith("r_")
    )
    assert len(joint_names) == 20
    rest = posture_rest_qpos(joint_names)
    data = mujoco.MjData(model)
    for name, value in zip(joint_names, rest):
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    mujoco.mj_forward(model, data)
    site_names = (
        "r_thumb_tip",
        "r_index_finger_tip",
        "r_middle_finger_tip",
        "r_ring_finger_tip",
        "r_pinky_tip",
    )
    targets = np.asarray([data.site_xpos[model.site(name).id] for name in site_names])[
        None, :, :
    ]
    solved, report = solve_fingertip_trajectory(
        model,
        trajectory_joint_names=joint_names,
        qpos=np.zeros((1, len(joint_names))),
        variable_joint_names=joint_names,
        target_site_names=site_names,
        target_positions=targets,
        config=FingertipIkConfig(iterations=200),
    )
    assert float(report.tip_errors_m.max()) < 5.0e-3
    limits = {
        name: tuple(model.jnt_range[model.joint(name).id]) for name in joint_names
    }
    audit = audit_finger_joints(solved, joint_names, limits)
    assert audit["pct_frames_any_hyperext_gt5deg"] == 0.0


def _fan(angles_deg: tuple[float, ...], palm_normal_axis: int = 1) -> np.ndarray:
    """Unit in-plane directions at the given palm-plane angles."""

    first, second = {0: (2, 1), 1: (0, 2), 2: (1, 0)}[palm_normal_axis]
    directions = np.zeros((len(angles_deg), 3), dtype=np.float64)
    for row, angle in enumerate(angles_deg):
        directions[row, first] = np.sin(np.radians(angle))
        directions[row, second] = np.cos(np.radians(angle))
    return directions


def test_finger_direction_yaw_recovers_a_uniform_rotation() -> None:
    """A same-signed demand on every finger is a palm yaw, and is recovered."""

    robot = _fan((-6.0, -2.0, 2.0, 6.0))
    source = _fan((-6.0 + 35.0, -2.0 + 35.0, 2.0 + 35.0, 6.0 + 35.0))[None, ...]
    fit = fit_palm_yaw_from_finger_directions(source, robot)
    assert fit["yaw_deg"] == pytest.approx(35.0, abs=0.05)
    assert fit["per_finger_residual_after_yaw_deg"] == [0.0, 0.0, 0.0, 0.0]


def test_finger_direction_yaw_sign_composes_onto_the_correction() -> None:
    """Applying the fitted yaw must remove the demand, not double it.

    ``correction @ _axis_rotation(axis, yaw)`` is how the converter composes
    the extra yaw, so the sign convention is pinned by construction here.
    """

    axis = 1
    robot = _fan((-6.0, -2.0, 2.0, 6.0), axis)
    yaw_truth = 22.0
    rotation = _axis_rotation(axis, np.radians(yaw_truth))
    # Row-vector convention: v_palm = v_source @ correction.
    source = (robot @ rotation.T)[None, ...]
    fit = fit_palm_yaw_from_finger_directions(source, robot, palm_normal_axis=axis)
    assert fit["yaw_deg"] == pytest.approx(yaw_truth, abs=0.05)

    corrected = source[0] @ _axis_rotation(axis, np.radians(fit["yaw_deg"]))
    residual = fit_palm_yaw_from_finger_directions(
        corrected[None, ...], robot, palm_normal_axis=axis
    )
    assert abs(residual["yaw_deg"]) < 0.05


def test_finger_direction_yaw_ignores_genuine_spread() -> None:
    """Real spreading is opposite-signed and must not read as a yaw."""

    robot = _fan((-6.0, -2.0, 2.0, 6.0))
    source = _fan((-16.0, -6.0, 6.0, 16.0))[None, ...]
    fit = fit_palm_yaw_from_finger_directions(source, robot)
    assert abs(fit["yaw_deg"]) < 0.05
    assert fit["per_finger_residual_after_yaw_deg"][0] < -5.0
    assert fit["per_finger_residual_after_yaw_deg"][-1] > 5.0


def test_finger_direction_yaw_drops_curled_frames() -> None:
    """A finger curled through the palm normal has no in-plane angle."""

    robot = _fan((-6.0, -2.0, 2.0, 6.0))
    good = _fan((4.0, 8.0, 12.0, 16.0))
    curled = np.tile(np.asarray((0.0, 1.0, 0.0)), (4, 1))  # along the palm normal
    source = np.stack([good, curled, good])
    fit = fit_palm_yaw_from_finger_directions(source, robot)
    assert fit["per_finger_usable_frame_count"] == [2, 2, 2, 2]
    assert fit["yaw_deg"] == pytest.approx(10.0, abs=0.05)


def test_real_asset_zero_pose_finger_fan_is_ordered_and_narrow() -> None:
    model = _load_vega_wuji_model()
    if model is None:
        pytest.skip("Vega-Wuji MJCF or its meshes are not available.")
    for side in ("left", "right"):
        directions = wuji_finger_directions_palm_frame(model, side)
        assert directions.shape == (4, 3)
        assert np.allclose(np.linalg.norm(directions, axis=-1), 1.0)
        # A zero-pose fan spans a modest angle. The fingers point near the
        # -z palm axis, so the raw angles straddle +/-180 and must be
        # unwrapped against the first finger before measuring the span.
        angles = np.degrees(np.arctan2(directions[:, 0], directions[:, 2]))
        relative = (angles - angles[0] + 180.0) % 360.0 - 180.0
        assert float(np.ptp(relative)) < 60.0
        # Fitting the fan against itself must report no yaw demand.
        fit = fit_palm_yaw_from_finger_directions(directions[None, ...], directions)
        assert abs(fit["yaw_deg"]) < 1.0e-6


AVOIDANCE_XML = """
<mujoco>
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="wall" pos="0 0.03 -0.05" mocap="true">
      <geom name="wall_geom" type="box" size="0.2 0.01 0.2"/>
    </body>
    <body name="proximal" pos="0 0 0">
      <joint name="r_test_finger_pip" type="hinge" axis="1 0 0"
             range="-1.047 2.094" limited="true"/>
      <geom name="prox_geom" type="capsule" fromto="0 0 0 0 0 -0.04" size="0.004"/>
      <body name="distal" pos="0 0 -0.04">
        <joint name="r_test_finger_dip" type="hinge" axis="1 0 0"
               range="-1.047 1.57" limited="true"/>
        <geom name="dist_geom" type="capsule" fromto="0 0 0 0 0 -0.03" size="0.004"/>
        <site name="r_test_finger_tip" pos="0 0 -0.03"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _wall_penetration(model, qpos, joint_names) -> float:
    """Worst signed distance between the finger geoms and the wall."""

    data = mujoco.MjData(model)
    fromto = np.zeros(6)
    worst = np.inf
    for frame in range(len(qpos)):
        for index, name in enumerate(joint_names):
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = qpos[frame, index]
        mujoco.mj_forward(model, data)
        for geom in ("prox_geom", "dist_geom"):
            worst = min(
                worst,
                float(
                    mujoco.mj_geomDistance(
                        model,
                        data,
                        model.geom(geom).id,
                        model.geom("wall_geom").id,
                        0.5,
                        fromto,
                    )
                ),
            )
    return worst


def test_avoidance_trades_tip_accuracy_for_clearance() -> None:
    """A target inside a wall must not be reached by burying the finger."""

    joint_names = ("r_test_finger_pip", "r_test_finger_dip")
    model = mujoco.MjModel.from_xml_string(AVOIDANCE_XML)
    apply_anatomical_finger_limits(model)
    seed = np.zeros((1, 2))
    # The wall is a slab spanning y in [0.02, 0.04]. Bending the finger sweeps
    # its tip through +y, so a target past the far face is reachable only by
    # driving through the wall - and the escape direction is one the joints can
    # actually produce, which a barrier needs.
    target = np.asarray([[[0.0, 0.05, -0.05]]])
    poses = np.asarray([[0.0, 0.03, -0.05, 1.0, 0.0, 0.0, 0.0]])

    free, free_report = solve_fingertip_trajectory(
        model,
        trajectory_joint_names=joint_names,
        qpos=seed,
        variable_joint_names=joint_names,
        target_site_names=("r_test_finger_tip",),
        target_positions=target,
        config=FingertipIkConfig(iterations=200, tip_weights=(1.0,)),
    )
    guarded, guarded_report = solve_fingertip_trajectory(
        model,
        trajectory_joint_names=joint_names,
        qpos=seed,
        variable_joint_names=joint_names,
        target_site_names=("r_test_finger_tip",),
        target_positions=target,
        config=FingertipIkConfig(iterations=200, tip_weights=(1.0,)),
        avoidance=FingertipAvoidanceConfig(
            geom_pairs=(("dist_geom", "wall_geom"), ("prox_geom", "wall_geom")),
            clearance_m=0.0,
            weight=50.0,
            query_cutoff_m=0.1,
            mocap_body_name="wall",
        ),
        scene_object_poses=poses,
    )
    free_gap = _wall_penetration(model, free, joint_names)
    guarded_gap = _wall_penetration(model, guarded, joint_names)
    # The barrier is a cost, not a hard constraint, so it settles at a trade
    # point: measurably further out of the wall...
    assert guarded_gap > free_gap + 2.0e-3
    # ...paid for with fingertip accuracy, not the reverse.
    assert float(guarded_report.tip_errors_m.max()) >= float(
        free_report.tip_errors_m.max()
    )


def test_avoidance_is_inert_when_nothing_is_close() -> None:
    """A finger far from the wall must solve exactly as it did before."""

    joint_names = ("r_test_finger_pip", "r_test_finger_dip")
    model = mujoco.MjModel.from_xml_string(AVOIDANCE_XML)
    apply_anatomical_finger_limits(model)
    seed = np.zeros((1, 2))
    target = np.asarray([[[0.0, 0.02, -0.05]]])
    poses = np.asarray([[0.0, 0.0, -5.0, 1.0, 0.0, 0.0, 0.0]])  # wall far below

    plain, plain_report = solve_fingertip_trajectory(
        model,
        trajectory_joint_names=joint_names,
        qpos=seed,
        variable_joint_names=joint_names,
        target_site_names=("r_test_finger_tip",),
        target_positions=target,
        config=FingertipIkConfig(iterations=200, tip_weights=(1.0,)),
    )
    guarded, guarded_report = solve_fingertip_trajectory(
        model,
        trajectory_joint_names=joint_names,
        qpos=seed,
        variable_joint_names=joint_names,
        target_site_names=("r_test_finger_tip",),
        target_positions=target,
        config=FingertipIkConfig(iterations=200, tip_weights=(1.0,)),
        avoidance=FingertipAvoidanceConfig(
            geom_pairs=(("dist_geom", "wall_geom"),),
            clearance_m=0.0005,
            weight=50.0,
            query_cutoff_m=0.05,
            mocap_body_name="wall",
        ),
        scene_object_poses=poses,
    )
    assert np.allclose(plain, guarded, atol=1.0e-9)
    assert float(guarded_report.tip_errors_m.max()) == pytest.approx(
        float(plain_report.tip_errors_m.max())
    )


def test_avoidance_config_rejects_an_unreachable_cutoff() -> None:
    with pytest.raises(ValueError, match="must exceed clearance_m"):
        FingertipAvoidanceConfig(
            geom_pairs=(("a", "b"),), clearance_m=0.05, query_cutoff_m=0.05
        )


def test_avoidance_escapes_from_an_already_penetrating_seed() -> None:
    """The witness direction flips meaning once geoms overlap.

    While apart it points across the gap, so escaping means moving against
    it; once overlapping it points from the deepest robot point out through
    the surface, so escaping means moving along it. Using the separated-case
    sign for both drove penetrating geoms deeper, which on the real clip took
    hand-object penetration from -10.25 mm to -20.36 mm.
    """

    joint_names = ("r_test_finger_pip", "r_test_finger_dip")
    model = mujoco.MjModel.from_xml_string(AVOIDANCE_XML)
    apply_anatomical_finger_limits(model)
    poses = np.asarray([[0.0, 0.03, -0.05, 1.0, 0.0, 0.0, 0.0]])
    # Seed already bent into the slab, and a target that keeps it there.
    seed = np.asarray([[0.9, 0.2]])
    target = np.asarray([[[0.0, 0.05, -0.05]]])

    guarded, _ = solve_fingertip_trajectory(
        model,
        trajectory_joint_names=joint_names,
        qpos=seed,
        variable_joint_names=joint_names,
        target_site_names=("r_test_finger_tip",),
        target_positions=target,
        config=FingertipIkConfig(iterations=200, tip_weights=(1.0,)),
        avoidance=FingertipAvoidanceConfig(
            geom_pairs=(("dist_geom", "wall_geom"), ("prox_geom", "wall_geom")),
            clearance_m=0.0,
            weight=50.0,
            query_cutoff_m=0.1,
            mocap_body_name="wall",
        ),
        scene_object_poses=poses,
    )
    seed_gap = _wall_penetration(model, seed, joint_names)
    guarded_gap = _wall_penetration(model, guarded, joint_names)
    assert seed_gap < 0.0, "the seed must start inside the slab"
    # It must come out, not burrow further in.
    assert guarded_gap > seed_gap
