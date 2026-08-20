"""Soft replay must remove penetration without abandoning the retarget."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from iltools.retarget.contact_settling import (
    ContactSettlingConfig,
    settle_contact_trajectory,
)

# A slide joint pushes a sphere along +x into a fixed sphere. Both radii are
# 0.05, so the surfaces meet when the centres are 0.10 apart. A target that
# asks for less than that is a penetrating retarget.
_MODEL_XML = """
<mujoco model="contact_settling_fixture">
  <option timestep="0.002"/>
  <worldbody>
    <body name="hand" pos="0 0 0">
      <joint name="slide_x" type="slide" axis="1 0 0" range="-1 1"/>
      <geom name="tip" type="sphere" size="0.05" contype="1" conaffinity="1"/>
    </body>
    <body name="object_body" mocap="true" pos="0.30 0 0">
      <geom name="object_geom" type="sphere" size="0.05"
            contype="1" conaffinity="1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="slide_x_position" joint="slide_x" kp="200" kv="20"
              ctrlrange="-1 1" forcerange="-50 50"/>
  </actuator>
</mujoco>
"""

OBJECT_X = 0.30
TOUCH_SEPARATION = 0.10


def _settle(centre_gaps, *, config=None):
    model = mujoco.MjModel.from_xml_string(_MODEL_XML)
    data = mujoco.MjData(model)
    frames = len(centre_gaps)
    qpos = np.array([[OBJECT_X - gap] for gap in centre_gaps], dtype=np.float64)
    poses = np.tile(
        np.array([[OBJECT_X, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]), (frames, 1)
    ).reshape(frames, 1, 7)
    achieved, report = settle_contact_trajectory(
        model,
        data,
        qpos=qpos,
        joint_names=["slide_x"],
        object_geom_names=["object_geom"],
        robot_geom_names=["tip"],
        object_mocap_poses=poses,
        object_mocap_body_names=["object_body"],
        config=config,
    )
    # Report the achieved centre gap, which is easier to reason about.
    return OBJECT_X - achieved[:, 0], report


def test_a_penetrating_target_is_pushed_out_to_the_surface() -> None:
    # Ask for a 3 cm overlap.
    gaps, report = _settle([TOUCH_SEPARATION - 0.03])

    assert report.worst_penetration_before_m < -0.02
    # The tip ends resting on the surface, not inside it and not far away.
    assert gaps[0] == pytest.approx(TOUCH_SEPARATION, abs=2.0e-3)
    assert report.frames_penetrating_before == 1
    assert report.worst_penetration_after_m > -2.0e-3


def test_a_clear_target_is_left_alone() -> None:
    """Soft replay must not move a pose that never touched the object."""

    requested = TOUCH_SEPARATION + 0.05
    gaps, report = _settle([requested])

    assert gaps[0] == pytest.approx(requested, abs=1.0e-3)
    assert report.frames_penetrating_before == 0
    assert report.frames_penetrating_after == 0
    assert report.max_joint_shift_rad < 1.0e-3


def test_the_settled_pose_stays_close_to_the_retarget() -> None:
    """The correction is the smallest one that removes the overlap.

    A 3 cm overlap must be answered by roughly a 3 cm push, not by retreating
    to some larger standoff. This is the property that separates soft replay
    from the projection it replaces.
    """

    overlap = 0.03
    gaps, report = _settle([TOUCH_SEPARATION - overlap])

    assert report.max_joint_shift_rad == pytest.approx(overlap, abs=3.0e-3)
    assert gaps[0] < TOUCH_SEPARATION + 5.0e-3


def test_a_mixed_trajectory_settles_each_frame_on_its_own() -> None:
    gaps, report = _settle(
        [
            TOUCH_SEPARATION - 0.02,  # penetrating
            TOUCH_SEPARATION + 0.10,  # clear
            TOUCH_SEPARATION - 0.01,  # penetrating
        ]
    )

    assert report.frame_count == 3
    assert report.frames_penetrating_before == 2
    assert gaps[0] == pytest.approx(TOUCH_SEPARATION, abs=3.0e-3)
    assert gaps[1] == pytest.approx(TOUCH_SEPARATION + 0.10, abs=2.0e-3)
    assert gaps[2] == pytest.approx(TOUCH_SEPARATION, abs=3.0e-3)


def test_settling_restores_the_model_it_borrowed() -> None:
    """Gravity and actuator gains must survive the pass unchanged."""

    model = mujoco.MjModel.from_xml_string(_MODEL_XML)
    data = mujoco.MjData(model)
    gravity = np.array(model.opt.gravity)
    gain = np.array(model.actuator_gainprm)
    bias = np.array(model.actuator_biasprm)

    settle_contact_trajectory(
        model,
        data,
        qpos=np.array([[OBJECT_X - TOUCH_SEPARATION + 0.02]]),
        joint_names=["slide_x"],
        object_geom_names=["object_geom"],
        robot_geom_names=["tip"],
        object_mocap_poses=np.array([[[OBJECT_X, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]]),
        object_mocap_body_names=["object_body"],
        config=ContactSettlingConfig(stiffness_scale=2.0, damping_scale=3.0),
    )

    assert np.allclose(model.opt.gravity, gravity)
    assert np.allclose(model.actuator_gainprm, gain)
    assert np.allclose(model.actuator_biasprm, bias)


def test_config_rejects_a_degenerate_settle() -> None:
    with pytest.raises(ValueError, match="settle_steps"):
        ContactSettlingConfig(settle_steps=0)
    with pytest.raises(ValueError, match="stiffness_scale"):
        ContactSettlingConfig(stiffness_scale=0.0)


def test_a_joint_without_an_actuator_is_refused() -> None:
    xml = _MODEL_XML.replace(
        '<position name="slide_x_position" joint="slide_x" kp="200" kv="20"\n'
        '              ctrlrange="-1 1" forcerange="-50 50"/>',
        "",
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    with pytest.raises(ValueError, match="no position actuator"):
        settle_contact_trajectory(
            model,
            data,
            qpos=np.array([[0.2]]),
            joint_names=["slide_x"],
            object_geom_names=["object_geom"],
            robot_geom_names=["tip"],
        )
