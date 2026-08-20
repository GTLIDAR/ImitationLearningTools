import json

import numpy as np
import pytest

from iltools.core.trajectory import Trajectory
from iltools.retarget import (
    JointMapRetargeter,
    JointMapSpec,
    KeypointJointSpec,
    KeypointRetargeter,
    load_ego_pose_bundle,
    load_ego_pose_trajectory,
    merge_joint_trajectories,
    MujocoKeypointRetargeter,
    MujocoPositionTaskSpec,
    PinocchioKeypointRetargeter,
    PinocchioPositionTaskSpec,
    save_joint_reference_npz,
    transform_keypoint_trajectory,
)


def test_keypoint_retargeter_maps_hinge_angle_and_velocity() -> None:
    trajectory = Trajectory(
        observations={
            "keypoints": np.asarray(
                [
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
                ]
            )
        },
        dt=0.1,
    )
    retargeter = KeypointRetargeter(
        ("parent", "joint", "child"),
        (
            KeypointJointSpec(
                target_name="wuji_index_joint",
                parent="parent",
                joint="joint",
                child="child",
                lower=0.0,
                upper=np.pi,
            ),
        ),
    )

    result = retargeter.retarget(trajectory)

    np.testing.assert_allclose(result.observations["qpos"][:, 0], np.pi / 2.0)
    np.testing.assert_allclose(result.observations["qvel"], 0.0)
    assert result.infos["retarget"]["method"] == "keypoint_angle"


def test_joint_map_retargeter_clips_and_keeps_target_order() -> None:
    trajectory = Trajectory(
        observations={"joint_position": np.asarray([[0.25, 2.0]])},
        dt=0.02,
    )
    retargeter = JointMapRetargeter(
        ("human_index", "human_thumb"),
        ("wuji_thumb", "wuji_index"),
        (
            JointMapSpec("wuji_index", "human_index", -1.0, 1.0, scale=2.0),
            JointMapSpec("wuji_thumb", "human_thumb", -0.5, 0.5),
        ),
    )

    result = retargeter.retarget(trajectory)

    np.testing.assert_allclose(result.observations["qpos"], [[0.5, 0.5]])
    assert result.infos["retarget"]["target_joint_names"] == [
        "wuji_thumb",
        "wuji_index",
    ]


def test_ego_exo4d_loader_reads_world_keypoints(tmp_path) -> None:
    path = tmp_path / "take.json"
    path.write_text(
        json.dumps(
            {
                "10": [
                    {
                        "annotation3D": {
                            "left_wrist": {"x": 1, "y": 2, "z": 3},
                            "right_wrist": {"x": 4, "y": 5, "z": 6},
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    trajectory = load_ego_pose_trajectory(
        path,
        keypoint_names=("left_wrist", "right_wrist"),
        fps=30.0,
    )

    np.testing.assert_allclose(
        trajectory.observations["keypoints"], [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
    )
    np.testing.assert_array_equal(trajectory.observations["frame_number"], [10])
    assert trajectory.infos["coordinate_frame"] == "world"


def test_ego_exo4d_bundle_merges_body_and_hand_files(tmp_path) -> None:
    body_path = tmp_path / "body.json"
    hand_path = tmp_path / "hand.json"
    body_path.write_text(
        json.dumps(
            {
                "10": [{"annotation3D": {"right-wrist": {"x": 1, "y": 2, "z": 3}}}],
                "11": [{"annotation3D": {"right-wrist": {"x": 2, "y": 2, "z": 3}}}],
            }
        ),
        encoding="utf-8",
    )
    hand_path.write_text(
        json.dumps(
            {
                "10": [{"annotation3D": {"right_index_1": {"x": 4, "y": 5, "z": 6}}}],
                "11": [{"annotation3D": {"right_index_1": {"x": 5, "y": 5, "z": 6}}}],
            }
        ),
        encoding="utf-8",
    )

    trajectory = load_ego_pose_bundle(
        (body_path, hand_path),
        keypoint_names_by_file=(("right-wrist",), ("right_index_1",)),
        fps=50.0,
    )

    assert trajectory.infos["keypoint_names"] == ["right-wrist", "right_index_1"]
    np.testing.assert_allclose(
        trajectory.observations["keypoints"],
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[2.0, 2.0, 3.0], [5.0, 5.0, 6.0]],
        ],
    )


def test_ego_exo4d_loader_rejects_non_contiguous_frames(tmp_path) -> None:
    path = tmp_path / "gapped.json"
    path.write_text(
        json.dumps(
            {
                "10": [{"annotation3D": {"right-wrist": {"x": 1, "y": 2, "z": 3}}}],
                "12": [{"annotation3D": {"right-wrist": {"x": 2, "y": 2, "z": 3}}}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not contiguous"):
        load_ego_pose_trajectory(
            path,
            keypoint_names=("right-wrist",),
            fps=50.0,
        )


def test_ego_exo4d_loader_can_explicitly_resample_frame_gaps(tmp_path) -> None:
    path = tmp_path / "gapped.json"
    path.write_text(
        json.dumps(
            {
                "10": [{"annotation3D": {"right-wrist": {"x": 1, "y": 2, "z": 3}}}],
                "12": [{"annotation3D": {"right-wrist": {"x": 3, "y": 2, "z": 3}}}],
            }
        ),
        encoding="utf-8",
    )

    trajectory = load_ego_pose_trajectory(
        path,
        keypoint_names=("right-wrist",),
        fps=50.0,
        resample_gaps=True,
    )

    np.testing.assert_array_equal(trajectory.observations["frame_number"], [10, 11, 12])
    np.testing.assert_allclose(
        trajectory.observations["keypoints"],
        [[[1.0, 2.0, 3.0]], [[2.0, 2.0, 3.0]], [[3.0, 2.0, 3.0]]],
    )
    assert trajectory.infos["resampled_frame_gaps"] is True


def test_save_joint_reference_npz_round_trip(tmp_path) -> None:
    trajectory = Trajectory(
        observations={"qpos": np.asarray([[0.0], [0.5]])},
    )

    output = save_joint_reference_npz(
        trajectory,
        tmp_path / "reference.npz",
        joint_names=("wuji_index_joint",),
        fps=50.0,
    )

    with np.load(output) as data:
        np.testing.assert_allclose(data["qpos"], [[0.0], [0.5]])
        np.testing.assert_allclose(data["qvel"], [[25.0], [25.0]])
        assert data["joint_names"].tolist() == ["wuji_index_joint"]
        assert float(data["fps"]) == 50.0


def test_merge_joint_trajectories_preserves_requested_order() -> None:
    arm = Trajectory(
        observations={"qpos": np.asarray([[0.1, 0.2], [0.2, 0.3]])},
        infos={
            "retarget": {
                "method": "mujoco_dls_keypoint",
                "target_joint_names": ["arm_a", "arm_b"],
            }
        },
        dt=0.02,
    )
    hand = Trajectory(
        observations={"qpos": np.asarray([[0.4], [0.6]])},
        infos={
            "retarget": {
                "method": "keypoint_angle",
                "target_joint_names": ["finger"],
            }
        },
        dt=0.02,
    )

    result = merge_joint_trajectories(
        (arm, hand),
        joint_names=("finger", "arm_a", "arm_b"),
    )

    np.testing.assert_allclose(
        result.observations["qpos"], [[0.4, 0.1, 0.2], [0.6, 0.2, 0.3]]
    )
    np.testing.assert_allclose(
        result.observations["qvel"], [[10.0, 5.0, 5.0], [10.0, 5.0, 5.0]]
    )
    assert result.infos["retarget"]["method"] == "joint_merge"


def test_mujoco_keypoint_retargeter_solves_position_task() -> None:
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <compiler angle="radian"/>
          <worldbody>
            <body name="base">
              <joint name="joint_a" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
              <geom type="sphere" size="0.1" mass="1"/>
              <body pos="1 0 0">
                <joint name="joint_b" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                <geom type="sphere" size="0.1" mass="1"/>
                <site name="tip" pos="1 0 0"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    trajectory = Trajectory(
        observations={"keypoints": np.asarray([[[1.4, 0.8, 0.0]]])},
        infos={"keypoint_names": ["right_wrist"], "coordinate_frame": "robot"},
        dt=0.02,
    )
    retargeter = MujocoKeypointRetargeter(
        model,
        ("joint_a", "joint_b"),
        (MujocoPositionTaskSpec("right_wrist", "tip"),),
        iterations=200,
        tolerance=1.0e-6,
    )

    world_frame = Trajectory(
        observations={"keypoints": np.asarray([[[1.4, 0.8, 0.0]]])},
        infos={"keypoint_names": ["right_wrist"], "coordinate_frame": "world"},
        dt=0.02,
    )
    with pytest.raises(ValueError, match="robot model frame"):
        retargeter.retarget(world_frame)

    result = retargeter.retarget(trajectory)

    data = mujoco.MjData(model)
    joint_ids = [model.joint(name).id for name in ("joint_a", "joint_b")]
    data.qpos[model.jnt_qposadr[joint_ids]] = result.observations["qpos"][0]
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(
        data.site_xpos[model.site("tip").id], [1.4, 0.8, 0.0], atol=1.0e-3
    )
    assert result.infos["retarget"]["method"] == "mujoco_dls_keypoint"


def test_pinocchio_keypoint_retargeter_solves_position_task() -> None:
    pin = pytest.importorskip("pinocchio")

    model = pin.Model()
    joint_id = model.addJoint(0, pin.JointModelRY(), pin.SE3.Identity(), "joint_a")
    model.addFrame(
        pin.Frame(
            "tip",
            joint_id,
            0,
            pin.SE3(np.eye(3), np.asarray([1.0, 0.0, 0.0])),
            pin.FrameType.OP_FRAME,
        )
    )
    trajectory = Trajectory(
        observations={"keypoints": np.asarray([[[0.0, 0.0, -1.0]]])},
        infos={"keypoint_names": ["right_wrist"], "coordinate_frame": "robot"},
        dt=0.02,
    )
    retargeter = PinocchioKeypointRetargeter(
        model,
        ("joint_a",),
        (PinocchioPositionTaskSpec("right_wrist", "tip"),),
        iterations=100,
        tolerance=1.0e-6,
    )

    result = retargeter.retarget(trajectory)

    np.testing.assert_allclose(
        result.observations["qpos"], [[np.pi / 2.0]], atol=1.0e-3
    )
    assert result.infos["retarget"]["method"] == "pinocchio_dls_keypoint"


def test_transform_keypoint_trajectory_records_robot_frame() -> None:
    trajectory = Trajectory(
        observations={"keypoints": np.asarray([[[1.0, 2.0, 3.0]]])},
        infos={"coordinate_frame": "world"},
    )

    result = transform_keypoint_trajectory(
        trajectory,
        rotation=np.eye(3),
        translation=np.asarray([-1.0, 0.0, 1.0]),
        scale=2.0,
    )

    np.testing.assert_allclose(result.observations["keypoints"], [[[1.0, 4.0, 7.0]]])
    assert result.infos["coordinate_frame"] == "robot"
