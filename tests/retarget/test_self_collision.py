from __future__ import annotations

import mujoco
import numpy as np

from iltools.retarget import MujocoSelfCollisionClosureProjector


MODEL = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="fixed" pos="0 0 0">
      <geom name="fixed_finger" type="sphere" size="0.05"/>
    </body>
    <body name="closing" pos="0.2 0 0">
      <joint name="closure" type="slide" axis="-1 0 0" range="0 0.2"/>
      <geom name="moving_finger" type="sphere" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_self_collision_projection_keeps_maximal_safe_closure() -> None:
    model = mujoco.MjModel.from_xml_string(MODEL)
    projector = MujocoSelfCollisionClosureProjector(
        model,
        trajectory_joint_names=("closure",),
        closure_joint_names=("closure",),
        robot_geom_names=("fixed_finger", "moving_finger"),
    )

    result, scales, report = projector.project(np.asarray([[0.04], [0.15]]))

    np.testing.assert_allclose(result[0], [0.04])
    assert 0.09 <= result[1, 0] <= 0.102
    assert scales[0] == 1.0
    assert 0.5 < scales[1] < 0.75
    assert report.violating_frames_before == 1
    assert report.violating_frames_after == 0
    assert report.qualified
