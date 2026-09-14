import mujoco
import numpy as np

from iltools.core import DexterousReference
from iltools.retarget.mounted_scene import (
    contact_geometry_and_clearance,
    project_contact_constraints,
)
from iltools.retarget.mounted_hands import retime_mounted_reference


def scene():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><compiler autolimits="true"/><worldbody>
      <body name="left" pos="-0.08 0 0">
        <joint name="left_slide" type="slide" axis="-1 0 0" range="0 .1"/>
        <geom type="sphere" size=".05"/>
      </body>
      <body name="right" pos="0.08 0 0">
        <joint name="right_slide" type="slide" axis="1 0 0" range="0 .1"/>
        <geom type="sphere" size=".05"/>
      </body>
      <body name="object" mocap="true"><geom type="box" size=".05 .05 .05"/></body>
      <body name="stand" pos="0 0 -1"><geom type="box" size=".1 .1 .1"/></body>
    </worldbody></mujoco>""")
    reference = DexterousReference(
        sequence_id="contact",
        robot_name="fixture",
        fps=20,
        joint_names=("left_slide", "right_slide"),
        qpos=np.array([[0.0, 0.0], [0.1, 0.1]]),
        fixed_root_pose_w=np.array([0, 0, 0, 1, 0, 0, 0]),
        left_wrist_pose_w=np.tile([-0.08, 0, 0, 1, 0, 0, 0], (2, 1)),
        right_wrist_pose_w=np.tile([0.08, 0, 0, 1, 0, 0, 0], (2, 1)),
        left_wrist_frame_name="left",
        right_wrist_frame_name="right",
        object_names=("object",),
        object_poses_w=np.tile([0, 0, 0, 1, 0, 0, 0], (2, 1, 1)),
    )
    return model, reference


def test_contact_witnesses_distinguish_penetration_from_separation():
    model, reference = scene()
    contacts, report = contact_geometry_and_clearance(
        model, reference, {"left": ["left"], "right": ["right"]}
    )
    assert contacts.active[0].all()
    assert not contacts.active[1].any()
    np.testing.assert_allclose(
        report["robot_object_penetration_m"], [0.02, 0], atol=1e-6
    )
    np.testing.assert_allclose(contacts.link_normals_w[0, 0, 0], [1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(contacts.link_normals_w[0, 1, 0], [-1, 0, 0], atol=1e-6)


def test_bounded_projection_can_leave_a_joint_limit_and_preserves_clear_frames():
    model, reference = scene()
    qpos, report = project_contact_constraints(model, reference)
    assert report["maximum_penetration_after_m"] < 0.0006
    assert np.all(qpos[0] > 0.0194)
    np.testing.assert_allclose(qpos[1], reference.qpos[1])
    assert np.all(qpos >= 0) and np.all(qpos <= 0.1)


def test_projection_keeps_frozen_joints_and_reports_unresolved_contacts():
    model, reference = scene()
    qpos, report = project_contact_constraints(
        model, reference, variable_joint_names=("right_slide",)
    )
    np.testing.assert_allclose(qpos[:, 0], reference.qpos[:, 0])
    assert qpos[0, 1] > 0.0194
    assert report["maximum_penetration_after_m"] > 0.019


def test_neighbor_restarts_keep_original_joint_correction_bounds():
    model, reference = scene()
    qpos, report = project_contact_constraints(
        model, reference, joint_change_limits={"left_slide": 0.005}
    )
    assert qpos[0, 0] <= 0.005 + 1e-8
    assert report["maximum_penetration_after_m"] >= 0.015 - 1e-6


def test_retiming_preserves_endpoints_and_recomputes_twists(tmp_path):
    model, reference = scene()
    reference.metadata["mounted_retarget"] = {"kinematically_feasible_frames": [0, 1]}
    reference.object_poses_w[1, 0] = [1, 0, 0, 2**-0.5, 0, 0, 2**-0.5]
    path = tmp_path / "model.xml"
    mujoco.mj_saveLastXML(str(path), model)
    result = retime_mounted_reference(reference, path, 3)
    assert result.frame_count == 4 and result.fps == 20
    np.testing.assert_allclose(result.qpos[[0, -1]], reference.qpos)
    np.testing.assert_allclose(result.object_poses_w[[0, -1]], reference.object_poses_w)
    np.testing.assert_allclose(result.qvel, 2 / 3, atol=1e-6)
    np.testing.assert_allclose(result.object_twists_w[..., 0], 20 / 3, atol=1e-5)
    assert result.contacts is None and result.metadata["contact_geometry_pending"]
