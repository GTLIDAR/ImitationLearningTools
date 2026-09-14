"""``settle_contact_trajectory`` drives extra mocap bodies per frame."""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from iltools.retarget.contact_settling import (  # noqa: E402
    ContactSettlingConfig,
    settle_contact_trajectory,
)

_MODEL = """
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="wall" mocap="true" pos="0.3 0 0">
      <geom name="wall_geom" type="box" size="0.02 0.2 0.2"/>
    </body>
    <body name="base" pos="0 0 0">
      <joint name="slide" type="slide" axis="1 0 0" range="-1 1"/>
      <geom name="tip" type="sphere" size="0.05"/>
    </body>
  </worldbody>
  <actuator>
    <position name="slide_ctrl" joint="slide" kp="50"/>
  </actuator>
</mujoco>
"""


def _model():
    model = mujoco.MjModel.from_xml_string(_MODEL)
    return model, mujoco.MjData(model)


def test_mocap_poses_move_the_named_body_each_frame() -> None:
    model, data = _model()
    # The settle starts every frame *at* its target and only resolves overlap.
    # Frame 0: the wall is far away, nothing overlaps. Frame 1: the wall sits
    # on the target itself, so the sphere (radius 0.05) starts inside it.
    wall = np.array(
        [[5.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.4, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
    )
    target = np.array([[0.4], [0.4]])
    achieved, report = settle_contact_trajectory(
        model,
        data,
        qpos=target,
        joint_names=["slide"],
        object_geom_names=["wall_geom"],
        robot_geom_names=["tip"],
        mocap_poses={"wall": wall},
        config=ContactSettlingConfig(settle_steps=400),
    )
    assert report.frame_count == 2
    # Frame 0: nothing to resolve, the pose is kept.
    assert achieved[0, 0] == pytest.approx(0.4, abs=0.02)
    # Frame 1: contact pushes the sphere out of the wall, against its actuator.
    assert report.frames_penetrating_before == 1
    assert achieved[1, 0] < 0.38
    assert abs(achieved[1, 0] - 0.4) > 0.02


def test_mocap_poses_shape_is_validated() -> None:
    model, data = _model()
    with pytest.raises(ValueError, match="mocap_poses"):
        settle_contact_trajectory(
            model,
            data,
            qpos=np.zeros((3, 1)),
            joint_names=["slide"],
            object_geom_names=["wall_geom"],
            robot_geom_names=["tip"],
            mocap_poses={"wall": np.zeros((2, 7))},
        )


def test_mocap_poses_rejects_a_non_mocap_body() -> None:
    model, data = _model()
    with pytest.raises(ValueError, match="not a mocap body"):
        settle_contact_trajectory(
            model,
            data,
            qpos=np.zeros((1, 1)),
            joint_names=["slide"],
            object_geom_names=["wall_geom"],
            robot_geom_names=["tip"],
            mocap_poses={"base": np.zeros((1, 7))},
        )
