from __future__ import annotations

import mujoco
import numpy as np

from iltools.retarget.contact_target_ik import (
    ContactTargetIKConfig,
    MujocoContactTargetRetargeter,
)


MODEL = """
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="arm" pos="0 0 0">
      <joint name="shoulder" type="hinge" axis="0 0 1" range="-170 170"/>
      <geom name="arm_geom" type="capsule" fromto="0 0 0 1 0 0" size="0.04"/>
      <body name="finger" pos="1 0 0">
        <joint name="finger_joint" type="hinge" axis="0 0 1" range="-170 170"/>
        <geom name="finger_geom" type="capsule" fromto="0 0 0 0.8 0 0" size="0.04"/>
      </body>
    </body>
    <body name="object" mocap="true" pos="1.4 0.4 0">
      <geom name="object_geom" type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_contact_target_ik_reduces_link_target_error() -> None:
    model = mujoco.MjModel.from_xml_string(MODEL)
    retargeter = MujocoContactTargetRetargeter(
        model,
        trajectory_joint_names=("shoulder", "finger_joint"),
        variable_joint_names={"left": ("shoulder", "finger_joint")},
        contact_body_names={"left": ("finger",)},
        contact_geom_names={"left": ("finger_geom",)},
        object_geom_name="object_geom",
        object_mocap_body_name="object",
        config=ContactTargetIKConfig(
            iterations=100,
            damping=0.01,
            regularization=1.0e-4,
            max_step_rad=0.1,
            tolerance_m=0.005,
        ),
    )
    qpos = np.zeros((1, 2), dtype=np.float64)
    object_pose = np.asarray(((1.4, 0.4, 0.0, 1.0, 0.0, 0.0, 0.0),))
    target = np.asarray(((((1.4, 0.3, 0.0),),),), dtype=np.float64)
    active = np.ones((1, 1, 1), dtype=bool)

    result, report = retargeter.retarget(
        qpos,
        object_poses_wxyz=object_pose,
        target_positions=target,
        active=active,
    )

    assert report.solved_targets == 1
    assert report.mean_final_error_m < report.mean_initial_error_m
    assert report.mean_final_error_m < 0.04
    assert not np.allclose(result, qpos)
