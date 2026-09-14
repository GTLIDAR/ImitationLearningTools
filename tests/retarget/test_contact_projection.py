from __future__ import annotations

import mujoco
import numpy as np

from iltools.retarget import (
    ContactConstraintProjectionConfig,
    MujocoContactConstraintProjector,
)


MODEL = """
<mujoco>
  <worldbody>
    <body name="left_slider" pos="-0.2 0 0">
      <joint name="left_x" type="slide" axis="1 0 0" range="-0.5 0.5"/>
      <geom name="left_tip" type="sphere" size="0.02"/>
    </body>
    <body name="right_slider" pos="0.2 0 0">
      <joint name="right_x" type="slide" axis="1 0 0" range="-0.5 0.5"/>
      <geom name="right_tip" type="sphere" size="0.02"/>
    </body>
    <body name="object" mocap="true">
      <geom name="object_geom" type="sphere" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_contact_constraint_projector_recovers_both_sides() -> None:
    model = mujoco.MjModel.from_xml_string(MODEL)
    projector = MujocoContactConstraintProjector(
        model,
        trajectory_joint_names=("left_x", "right_x"),
        variable_joint_names={"left": ("left_x",), "right": ("right_x",)},
        contact_geom_names={"left": ("left_tip",), "right": ("right_tip",)},
        object_geom_name="object_geom",
        object_mocap_body_name="object",
        config=ContactConstraintProjectionConfig(iterations=40),
    )
    qpos = np.zeros((2, 2), dtype=np.float64)
    poses = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    active = np.asarray([[[True], [False]], [[False], [True]]])

    result, report = projector.project(qpos, object_poses_wxyz=poses, active=active)

    assert report.qualified
    assert report.recovery_rate == 1.0
    assert report.per_side_contact_frames == {"left": 1, "right": 1}
    assert result[0, 0] > 0.1
    assert result[1, 1] < -0.1
    np.testing.assert_allclose(result[0, 1], 0.0)
    np.testing.assert_allclose(result[1, 0], 0.0)
