import mujoco
import numpy as np

from iltools.core import CollisionAssetDependency, Trajectory
from iltools.retarget import (
    MujocoDualHandRetargeter,
    dexterous_reference_from_trajectory,
)


def _model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <compiler angle="radian"/>
          <worldbody>
            <body name="left_arm_body" pos="0 0.5 0">
              <geom type="sphere" size="0.01" mass="0.1"/>
              <joint name="left_arm" type="hinge" axis="0 0 1" range="-1 1"/>
              <site name="left_wrist" pos="0.3 0 0"/>
            </body>
            <body name="right_arm_body" pos="0 -0.5 0">
              <geom type="sphere" size="0.01" mass="0.1"/>
              <joint name="right_arm" type="hinge" axis="0 0 1" range="-1 1"/>
              <site name="right_wrist" pos="0.3 0 0"/>
            </body>
            <body name="left_finger_body">
              <geom type="sphere" size="0.01" mass="0.1"/>
              <joint name="left_finger" type="hinge" range="0 0.5"/>
            </body>
            <body name="right_finger_body">
              <geom type="sphere" size="0.01" mass="0.1"/>
              <joint name="right_finger" type="hinge" range="0 0.5"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )


def _wrist_poses(
    model: mujoco.MjModel,
    left_values: np.ndarray,
    right_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    result = []
    for joint_name, site_name, values in (
        ("left_arm", "left_wrist", left_values),
        ("right_arm", "right_wrist", right_values),
    ):
        poses = []
        for value in values:
            mujoco.mj_resetData(model, data)
            data.qpos[model.jnt_qposadr[model.joint(joint_name).id]] = value
            mujoco.mj_forward(model, data)
            quaternion = np.empty(4, dtype=np.float64)
            site_id = model.site(site_name).id
            mujoco.mju_mat2Quat(quaternion, data.site_xmat[site_id])
            poses.append(np.concatenate((data.site_xpos[site_id], quaternion)))
        result.append(np.asarray(poses))
    return result[0], result[1]


def test_dual_hand_retargeter_solves_wrist_pose_and_clips_fingers() -> None:
    model = _model()
    left_values = np.asarray([0.2, 0.35])
    right_values = np.asarray([-0.15, -0.3])
    left_pose, right_pose = _wrist_poses(model, left_values, right_values)
    object_poses = np.zeros((2, 1, 7), dtype=np.float64)
    object_poses[..., 3] = 1.0
    object_twists = np.asarray(
        [[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9, 1.0, 1.1, 1.2]]]
    )
    source = Trajectory(
        observations={
            "left_wrist_pose_w": left_pose,
            "right_wrist_pose_w": right_pose,
            "left_finger_qpos": np.asarray([[0.1], [0.8]]),
            "right_finger_qpos": np.asarray([[0.2], [0.3]]),
            "object_poses_w": object_poses,
            "object_twists_w": object_twists,
        },
        infos={"coordinate_frame": "robot"},
        dt=0.02,
    )
    retargeter = MujocoDualHandRetargeter(
        model,
        target_joint_names=(
            "left_arm",
            "right_arm",
            "left_finger",
            "right_finger",
        ),
        arm_joint_names=("left_arm", "right_arm"),
        left_finger_joint_names=("left_finger",),
        right_finger_joint_names=("right_finger",),
        left_wrist_site="left_wrist",
        right_wrist_site="right_wrist",
        iterations=150,
        damping=0.005,
        tolerance=1.0e-7,
        previous_posture_weight=1.0e-6,
        neutral_posture_weight=1.0e-7,
    )

    result = retargeter.retarget(source)

    np.testing.assert_allclose(
        result.observations["qpos"][:, 0], left_values, atol=2e-3
    )
    np.testing.assert_allclose(
        result.observations["qpos"][:, 1], right_values, atol=2e-3
    )
    np.testing.assert_allclose(result.observations["qpos"][:, 2], [0.1, 0.5])
    np.testing.assert_allclose(result.observations["qpos"][:, 3], [0.2, 0.3])
    assert result.infos["retarget"]["method"] == "mujoco_dual_wrist_se3"
    assert result.infos["retarget"]["left_wrist_frame_name"] == "left_wrist"
    assert result.infos["retarget"]["right_wrist_frame_name"] == "right_wrist"
    assert result.infos["retarget"]["previous_posture_weight"] == 1.0e-6
    assert result.infos["retarget"]["neutral_posture_weight"] == 1.0e-7

    reference = dexterous_reference_from_trajectory(
        result,
        sequence_id="dual_hand",
        robot_name="vega_wuji",
        fps=50.0,
        fixed_root_pose_w=np.asarray([0.0, 0.0, 0.19, 1.0, 0.0, 0.0, 0.0]),
        object_names=("cube",),
        object_asset_paths=("cube.usda",),
        collision_asset_dependencies=(
            CollisionAssetDependency(
                asset_role="object",
                asset_index=0,
                uri="collision.obj",
                sha256="0" * 64,
            ),
        ),
    )
    assert reference.joint_names == (
        "left_arm",
        "right_arm",
        "left_finger",
        "right_finger",
    )
    assert reference.left_wrist_frame_name == "left_wrist"
    assert reference.right_wrist_frame_name == "right_wrist"
    expected_object_poses = object_poses.copy()
    expected_object_poses[..., 2] += 0.19
    np.testing.assert_allclose(reference.object_poses_w, expected_object_poses)
    np.testing.assert_allclose(reference.object_twists_w, object_twists)
    assert reference.metadata["object_twist_frame"] == "world"
    assert reference.metadata["object_twist_source"] == "trajectory:object_twists_w"
    np.testing.assert_allclose(
        reference.left_wrist_pose_w[:, 2], left_pose[:, 2] + 0.19
    )
    assert reference.metadata["source_coordinate_frame"] == "robot"
    assert reference.collision_asset_dependencies[0].uri == "collision.obj"


def test_position_priority_does_not_trade_wrist_position_for_orientation() -> None:
    model = _model()
    left_position_pose, right_position_pose = _wrist_poses(
        model, np.zeros(2), np.zeros(2)
    )
    left_orientation_pose, right_orientation_pose = _wrist_poses(
        model, np.full(2, 0.8), np.full(2, -0.8)
    )
    left_target = left_position_pose.copy()
    right_target = right_position_pose.copy()
    left_target[:, 3:7] = left_orientation_pose[:, 3:7]
    right_target[:, 3:7] = right_orientation_pose[:, 3:7]
    source = Trajectory(
        observations={
            "left_wrist_pose_w": left_target,
            "right_wrist_pose_w": right_target,
            "left_finger_qpos": np.zeros((2, 1)),
            "right_finger_qpos": np.zeros((2, 1)),
        },
        infos={"coordinate_frame": "robot"},
    )
    retargeter = MujocoDualHandRetargeter(
        model,
        target_joint_names=(
            "left_arm",
            "right_arm",
            "left_finger",
            "right_finger",
        ),
        arm_joint_names=("left_arm", "right_arm"),
        left_finger_joint_names=("left_finger",),
        right_finger_joint_names=("right_finger",),
        left_wrist_site="left_wrist",
        right_wrist_site="right_wrist",
        iterations=20,
        damping=0.005,
        orientation_weight=10.0,
        position_priority=True,
    )

    result = retargeter.retarget(source)

    np.testing.assert_allclose(result.observations["qpos"][:, :2], 0.0, atol=1e-10)
    assert result.infos["retarget"]["position_priority"] is True


def test_reference_boundary_rotates_robot_frame_object_twists_to_world() -> None:
    poses = np.zeros((2, 7), dtype=np.float64)
    poses[:, 3] = 1.0
    object_poses = poses[:, None, :].copy()
    object_twists = np.zeros((2, 1, 6), dtype=np.float64)
    object_twists[..., 0] = 1.0
    object_twists[..., 3] = 2.0
    trajectory = Trajectory(
        observations={
            "qpos": np.zeros((2, 1), dtype=np.float64),
            "left_wrist_pose_w": poses,
            "right_wrist_pose_w": poses,
            "object_poses_w": object_poses,
            "object_twists_w": object_twists,
        },
        infos={
            "coordinate_frame": "robot",
            "retarget": {
                "target_joint_names": ["joint"],
                "left_wrist_frame_name": "left_wrist",
                "right_wrist_frame_name": "right_wrist",
            },
        },
        dt=0.02,
    )
    half_angle = np.pi / 4.0
    root_pose_w = np.asarray(
        [0.0, 0.0, 0.0, np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)]
    )

    reference = dexterous_reference_from_trajectory(
        trajectory,
        sequence_id="rotated_twist",
        robot_name="vega_wuji",
        fps=50.0,
        fixed_root_pose_w=root_pose_w,
        object_names=("cube",),
    )

    np.testing.assert_allclose(
        reference.object_twists_w[..., :3],
        [[[0.0, 1.0, 0.0]], [[0.0, 1.0, 0.0]]],
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        reference.object_twists_w[..., 3:],
        [[[0.0, 2.0, 0.0]], [[0.0, 2.0, 0.0]]],
        atol=1.0e-7,
    )
