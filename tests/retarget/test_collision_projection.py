from __future__ import annotations

import mujoco
import numpy as np

from iltools.retarget import (
    CollisionProjectionConfig,
    MujocoCollisionProjector,
    MujocoRadialEscapeProjector,
    MujocoSceneCollisionClosureProjector,
    RadialEscapeProjectionConfig,
    SceneCollisionClosureProjectionConfig,
)


MODEL = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="slider">
      <joint name="height" type="slide" axis="0 0 1" range="-0.2 0.2"/>
      <geom name="robot_sphere" type="sphere" size="0.05"/>
      <site name="task_site"/>
    </body>
    <geom name="support" type="plane" size="1 1 0.01"/>
  </worldbody>
</mujoco>
"""

MOCAP_MODEL = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="slider">
      <joint name="height" type="slide" axis="0 0 1" range="-0.2 0.2"/>
      <geom name="robot_sphere" type="sphere" size="0.05"/>
      <site name="task_site"/>
    </body>
    <body name="moving_obstacle" mocap="true" pos="0 0 -1">
      <geom name="obstacle" type="sphere" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_collision_projector_removes_forbidden_penetration() -> None:
    model = mujoco.MjModel.from_xml_string(MODEL)
    projector = MujocoCollisionProjector(
        model,
        trajectory_joint_names=("height",),
        variable_joint_names=("height",),
        robot_geom_names=("robot_sphere",),
        scene_geom_names=("support",),
        config=CollisionProjectionConfig(
            clearance_m=0.005,
            iterations=20,
            damping=0.001,
            regularization=1.0e-5,
            collision_weight=10.0,
            preserve_site_weight=0.1,
            max_step=0.05,
        ),
    )

    result, report = projector.project(np.asarray([[0.02], [0.08]]))

    assert report.violating_frames_before == 1
    assert report.violating_frames_after == 0
    assert report.corrected_frames == 1
    assert report.worst_clearance_after_m >= 0.0048
    assert result[0, 0] >= 0.0548
    np.testing.assert_allclose(result[1], [0.08], atol=1.0e-8)


def test_collision_projector_reports_unresolvable_fixed_pose() -> None:
    model = mujoco.MjModel.from_xml_string(MODEL)
    projector = MujocoCollisionProjector(
        model,
        trajectory_joint_names=("height",),
        variable_joint_names=("height",),
        robot_geom_names=("robot_sphere",),
        scene_geom_names=("support",),
        preserve_site_names=("task_site",),
        config=CollisionProjectionConfig(
            clearance_m=0.005,
            iterations=2,
            regularization=10.0,
            collision_weight=0.01,
            preserve_site_weight=10.0,
            max_step=0.001,
        ),
    )

    _, report = projector.project(np.asarray([[0.02]]))

    assert not report.qualified
    assert report.residual_frames == [0]


def test_collision_projector_accepts_an_exact_movable_pair() -> None:
    model = mujoco.MjModel.from_xml_string(MODEL)
    projector = MujocoCollisionProjector(
        model,
        trajectory_joint_names=("height",),
        variable_joint_names=("height",),
        forbidden_geom_pairs=(("robot_sphere", "support"),),
        config=CollisionProjectionConfig(clearance_m=0.005),
    )

    _, report = projector.project(np.asarray([[0.02]]))

    assert report.qualified


def test_collision_projector_replays_moving_scene_geometry() -> None:
    model = mujoco.MjModel.from_xml_string(MOCAP_MODEL)
    projector = MujocoCollisionProjector(
        model,
        trajectory_joint_names=("height",),
        variable_joint_names=("height",),
        robot_geom_names=("robot_sphere",),
        scene_geom_names=("obstacle",),
        scene_mocap_body_names=("moving_obstacle",),
        config=CollisionProjectionConfig(
            clearance_m=0.005,
            iterations=20,
            damping=0.001,
            regularization=1.0e-5,
            collision_weight=10.0,
            max_step=0.05,
        ),
    )
    poses = np.asarray(
        [
            [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0]],
        ]
    )

    result, report = projector.project(
        np.asarray([[0.02], [0.02]]), scene_mocap_poses=poses
    )

    assert report.violating_frames_before == 1
    assert report.violating_frames_after == 0
    assert result[0, 0] >= 0.1048
    np.testing.assert_allclose(result[1], [0.02], atol=1.0e-8)


def test_collision_projector_supports_tolerated_contact_penetration() -> None:
    model = mujoco.MjModel.from_xml_string(MODEL)
    projector = MujocoCollisionProjector(
        model,
        trajectory_joint_names=("height",),
        variable_joint_names=("height",),
        robot_geom_names=("robot_sphere",),
        scene_geom_names=("support",),
        config=CollisionProjectionConfig(
            clearance_m=-0.001,
            iterations=20,
            damping=0.001,
            regularization=1.0e-5,
            collision_weight=10.0,
            max_step=0.05,
        ),
    )

    result, report = projector.project(np.asarray([[0.02]]))

    assert report.qualified
    assert report.worst_clearance_after_m >= -0.0012
    assert 0.0488 <= result[0, 0] <= 0.0501


def test_radial_escape_projector_leaves_a_penetration_basin() -> None:
    model = mujoco.MjModel.from_xml_string(MOCAP_MODEL)
    projector = MujocoRadialEscapeProjector(
        model,
        trajectory_joint_names=("height",),
        variable_joint_names=("height",),
        robot_geom_names=("robot_sphere",),
        scene_geom_name="obstacle",
        target_site_name="task_site",
        scene_mocap_body_name="moving_obstacle",
        config=RadialEscapeProjectionConfig(
            clearance_m=0.005,
            retreat_step_m=0.01,
            maximum_retreat_m=0.12,
            damping=0.001,
        ),
    )
    poses = np.asarray([[[0.0, 0.0, -0.02, 1.0, 0.0, 0.0, 0.0]]])

    result, report = projector.project(np.asarray([[0.0]]), scene_mocap_poses=poses)

    assert report.violating_frames_before == 1
    assert report.qualified
    assert report.corrected_frames == 1
    assert report.worst_clearance_after_m >= 0.0049
    assert result[0, 0] >= 0.0849


def test_scene_collision_closure_retains_maximal_safe_fraction() -> None:
    model = mujoco.MjModel.from_xml_string(MOCAP_MODEL)
    projector = MujocoSceneCollisionClosureProjector(
        model,
        trajectory_joint_names=("height",),
        closure_joint_names=("height",),
        robot_geom_names=("robot_sphere",),
        scene_geom_name="obstacle",
        scene_mocap_body_name="moving_obstacle",
        config=SceneCollisionClosureProjectionConfig(
            clearance_m=-0.001,
            search_iterations=24,
        ),
    )
    poses = np.asarray([[[0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]]])

    result, scales, report = projector.project(
        np.asarray([[0.1]]), scene_mocap_poses=poses
    )

    assert report.violating_frames_before == 1
    assert report.qualified
    assert report.corrected_frames == 1
    assert 0.0 < scales[0] < 0.02
    assert 0.0 < result[0, 0] < 0.002
