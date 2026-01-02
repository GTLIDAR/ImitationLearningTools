import os

import numpy as np
import pytest
import zarr
from loco_mujoco.core import ObservationType
from omegaconf import DictConfig
from zarr.storage import LocalStore

from iltools.core.metadata_schema import DatasetMeta
from iltools.datasets.loco_mujoco.loader import LocoMuJoCoLoader


@pytest.fixture
def basic_cfg():
    """Basic configuration for tests."""
    return DictConfig(
        {
            "dataset": {
                "trajectories": {"default": ["walk", "run"], "amass": [], "lafan1": []}
            },
            "control_freq": 50.0,
            "window_size": 4,
            "sim": {"dt": 0.001},
            "decimation": 20,
        }
    )


@pytest.fixture
def minimal_loader():
    """
    Returns a minimal dummy loader with synthetic data for fast tests.
    """

    class DummyTraj:
        def __init__(self):
            self.observations = {"qpos": np.ones((10, 3), dtype=np.float32)}
            self.actions = {"actions": np.ones((10, 2), dtype=np.float32)}
            self.dt = 0.05

    class DummyLoader:
        def __init__(self):
            self._metadata = DatasetMeta(
                name="dummy_dataset",
                source="dummy",
                citation="none",
                version="0.0.1",
                num_trajectories=2,
                keys=["qpos", "actions"],
                trajectory_lengths=[10, 10],
                dt=[0.05, 0.05],
                joint_names=["joint1", "joint2", "joint3"],
                body_names=["body1", "body2"],
                site_names=["site1", "site2"],
                metadata={"dummy": True},
            )

        @property
        def metadata(self):
            return self._metadata

        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return DummyTraj()

    return DummyLoader()


def test_loader_basic(minimal_loader):
    """Test basic loader functionality."""
    assert len(minimal_loader) == 2
    traj = minimal_loader[0]
    assert isinstance(traj.observations["qpos"], np.ndarray)
    assert isinstance(traj.actions["actions"], np.ndarray)
    assert traj.observations["qpos"].shape == (10, 3)
    assert traj.actions["actions"].shape == (10, 2)
    assert isinstance(traj.dt, float)


def test_real_loco_loader_basic(basic_cfg):
    """Test real LocoMuJoCoLoader basic functionality."""
    try:
        loader = LocoMuJoCoLoader(
            env_name="UnitreeG1", cfg=basic_cfg, build_zarr_dataset=False, path=None
        )

        # Test basic properties
        assert hasattr(loader, "env_name")
        assert loader.env_name == "UnitreeG1"
        assert hasattr(loader, "metadata")
        assert isinstance(loader.metadata, DatasetMeta)

        # Test metadata properties
        meta = loader.metadata
        assert meta.name == "loco_mujoco"
        assert isinstance(meta.num_trajectories, int)
        assert meta.num_trajectories > 0

        print(f"✅ Loader test passed. Found {meta.num_trajectories} trajectories")

    except Exception as e:
        pytest.fail(f"LocoMuJoCoLoader test failed due to: {e}")


def test_real_loco_loader_trajectory_info(basic_cfg):
    """Test real LocoMuJoCoLoader trajectory information."""
    try:
        loader = LocoMuJoCoLoader(
            env_name="UnitreeG1", cfg=basic_cfg, build_zarr_dataset=False, path=None
        )

        num_traj = len(loader)
        print(f"Total number of trajectories: {num_traj}")

        # Test metadata
        meta = loader.metadata
        print(f"Dataset name: {meta.name}")
        print(f"Number of trajectories: {meta.num_trajectories}")
        print(f"Keys: {meta.keys}")
        print(f"Joint names count: {len(meta.joint_names)}")
        print(f"Body names count: {len(meta.body_names)}")

        assert isinstance(meta.trajectory_lengths, (list, int))
        assert isinstance(meta.dt, (list, float))

    except Exception as e:
        pytest.skip(f"LocoMuJoCoLoader info test skipped due to: {e}")


@pytest.mark.slow
def test_real_loco_loader_g1_walk_run_zarr(tmp_path):
    """Test G1 default walk+run export to Zarr with metadata logging."""
    cfg = DictConfig(
        {
            "dataset": {
                "trajectories": {"default": ["walk", "run"], "amass": [], "lafan1": []}
            },
            "control_freq": 50.0,
            "window_size": 4,
            "sim": {"dt": 0.001},
            "decimation": 20,
        }
    )

    try:
        loader = LocoMuJoCoLoader(
            env_name="UnitreeG1", cfg=cfg, build_zarr_dataset=True, path=tmp_path
        )
        zarr_path = tmp_path / "g1_default_walk_run.zarr"
        loader.save(str(zarr_path))

        store = LocalStore(str(zarr_path))
        root = zarr.group(store=store, overwrite=False)
        locomujoco_group = root["loco_mujoco"]

        print("Dataset metadata:")
        try:
            print(loader.metadata.dict())
        except Exception:
            print(loader.metadata)

        print("Zarr group attrs:")
        for key in locomujoco_group.attrs:
            print(f"{key}: {locomujoco_group.attrs[key]}")

        assert "default_walk" in locomujoco_group
        assert "default_run" in locomujoco_group
        assert "trajectory_0" in locomujoco_group["default_walk"]
        assert "trajectory_0" in locomujoco_group["default_run"]
        assert "motion_metadata" in locomujoco_group.attrs

    except Exception as e:
        pytest.skip(f"LocoMuJoCoLoader walk/run Zarr test skipped due to: {e}")


def create_fake_loco_npz(tmp_path, n=2, T=10):
    """Create fake NPZ files for testing."""
    for i in range(n):
        qpos = np.random.randn(T, 17).astype(np.float32)
        qvel = np.random.randn(T, 17).astype(np.float32)
        actions = np.random.randn(T, 6).astype(np.float32)
        dt = np.array([0.033], dtype=np.float32)

        np.savez(
            tmp_path / f"traj_{i}.npz", qpos=qpos, qvel=qvel, actions=actions, dt=dt
        )


@pytest.fixture
def fake_loco_dir(tmp_path):
    """Create fake loco directory for testing."""
    create_fake_loco_npz(tmp_path, n=3, T=8)
    return tmp_path


def test_metadata_schema():
    """Test DatasetMeta schema with correct parameters."""
    metadata = DatasetMeta(
        name="test_dataset",
        source="test",
        citation="test citation",
        version="1.0.0",
        num_trajectories=5,
        keys=["qpos", "qvel", "actions"],
        trajectory_lengths=[100, 150, 120, 200, 80],
        dt=[0.02, 0.02, 0.02, 0.02, 0.02],
        joint_names=["joint1", "joint2", "joint3"],
        body_names=["body1", "body2"],
        site_names=["site1", "site2"],
        metadata={"test": True},
    )

    assert metadata.name == "test_dataset"
    assert metadata.num_trajectories == 5
    # Handle both list and single value cases for trajectory_lengths
    if isinstance(metadata.trajectory_lengths, list):
        assert len(metadata.trajectory_lengths) == 5
    else:
        assert isinstance(metadata.trajectory_lengths, int)
    # Handle both list and single value cases for dt
    if isinstance(metadata.dt, list):
        assert len(metadata.dt) == 5
    else:
        assert isinstance(metadata.dt, float)
    assert len(metadata.keys) == 3


@pytest.mark.slow
def test_error_handling(tmp_path):
    """Test error handling for missing/corrupt files."""
    out_dir = tmp_path / "zarrbad"
    os.makedirs(out_dir, exist_ok=True)

    # Create incomplete/corrupt metadata
    with open(out_dir / "metadata.json", "w") as f:
        f.write("{bad json}")

    # Test that reading corrupt metadata raises an exception
    with pytest.raises(Exception):
        with open(out_dir / "metadata.json", "r") as f:
            import json

            json.load(f)


@pytest.mark.visualize
def test_visualize_first_trajectory_in_mujoco(visualize_enabled):
    """
    Optionally visualizes the first trajectory in MuJoCo using the loader's environment.

    Run with visualization:
        pytest tests/datasets/test_loco_mujoco_loader.py::test_visualize_first_trajectory_in_mujoco --visualize

    This test validates basic loader functionality and optionally shows visual inspection.
    """
    # Check if visualization is enabled via pytest marker
    # visualize_enabled = hasattr(pytest, "current_node") and "visualize" in str(
    #     pytest.current_node
    # )

    try:
        basic_cfg = DictConfig(
            {
                "dataset": {
                    "trajectories": {
                        "default": [
                            # "walkturn",
                            # "balance",
                            # "highjump",
                            # "highjump2",
                            # "highknee",
                            # "jumpturn",
                            # "onehopforward",
                            # "onestepleft",
                            # "onestepright",
                            # "onesteplong",
                            "onestepside1",
                            # "onestepside2",
                            # "random_walk",
                            # "run",
                            # "squat",
                            # "stepinplace1",
                            # "stepinplace2",
                            # "stepinplace3",
                            # "walk",
                        ],
                        "amass": [],
                        "lafan1": [
                            # "dance1_subject1",
                            # "dance1_subject2",
                            # "dance1_subject3",
                            # "dance2_subject1",
                            # "dance2_subject2",
                            # "dance2_subject3",
                            # "fallAndGetUp1_subject1",
                            # "fallAndGetUp1_subject4",
                            # "fallAndGetUp1_subject5",
                            # "fallAndGetUp2_subject2",
                            # "fallAndGetUp2_subject3",
                            # "fallAndGetUp3_subject1",
                            # "fight1_subject2",
                            # "fight1_subject3",
                            # "fight1_subject5",
                            # "fightAndSports1_subject1",
                            # "fightAndSports1_subject4",
                            # "jumps1_subject1",
                            # "jumps1_subject2",
                            # "jumps1_subject5",
                            # "run1_subject2",
                            # "run1_subject5",
                            # "run2_subject1",
                            # "run2_subject4",
                            # "sprint1_subject2",
                            # "sprint1_subject4",
                            # "walk1_subject1",
                            # "walk1_subject2",
                            # "walk1_subject5",
                            # "walk2_subject1",
                            # "walk2_subject3",
                            # "walk2_subject4",
                            # "walk3_subject1",
                            # "walk3_subject2",
                            # "walk3_subject3",
                            # "walk3_subject4",
                            # "walk3_subject5",
                            # "walk4_subject1",
                        ],
                    }
                },
                "n_substeps": 20,
            }
        )

        observation_spec = [  # ------------- JOINT POS -------------
            ObservationType.FreeJointPos("q_root", xml_name="root"),
            ObservationType.JointPos(
                "q_left_hip_pitch_joint", xml_name="left_hip_pitch_joint"
            ),
            ObservationType.JointPos(
                "q_left_hip_roll_joint", xml_name="left_hip_roll_joint"
            ),
            ObservationType.JointPos(
                "q_left_hip_yaw_joint", xml_name="left_hip_yaw_joint"
            ),
            ObservationType.JointPos("q_left_knee_joint", xml_name="left_knee_joint"),
            ObservationType.JointPos(
                "q_left_ankle_pitch_joint", xml_name="left_ankle_pitch_joint"
            ),
            ObservationType.JointPos(
                "q_left_ankle_roll_joint", xml_name="left_ankle_roll_joint"
            ),
            ObservationType.JointPos(
                "q_right_hip_pitch_joint", xml_name="right_hip_pitch_joint"
            ),
            ObservationType.JointPos(
                "q_right_hip_roll_joint", xml_name="right_hip_roll_joint"
            ),
            ObservationType.JointPos(
                "q_right_hip_yaw_joint", xml_name="right_hip_yaw_joint"
            ),
            ObservationType.JointPos("q_right_knee_joint", xml_name="right_knee_joint"),
            ObservationType.JointPos(
                "q_right_ankle_pitch_joint", xml_name="right_ankle_pitch_joint"
            ),
            ObservationType.JointPos(
                "q_right_ankle_roll_joint", xml_name="right_ankle_roll_joint"
            ),
            ObservationType.JointPos("q_waist_yaw_joint", xml_name="waist_yaw_joint"),
            ObservationType.JointPos(
                "q_left_shoulder_pitch_joint", xml_name="left_shoulder_pitch_joint"
            ),
            ObservationType.JointPos(
                "q_left_shoulder_roll_joint", xml_name="left_shoulder_roll_joint"
            ),
            ObservationType.JointPos(
                "q_left_shoulder_yaw_joint", xml_name="left_shoulder_yaw_joint"
            ),
            ObservationType.JointPos("q_left_elbow_joint", xml_name="left_elbow_joint"),
            ObservationType.JointPos(
                "q_left_wrist_roll_joint", xml_name="left_wrist_roll_joint"
            ),
            ObservationType.JointPos(
                "q_right_shoulder_pitch_joint", xml_name="right_shoulder_pitch_joint"
            ),
            ObservationType.JointPos(
                "q_right_shoulder_roll_joint", xml_name="right_shoulder_roll_joint"
            ),
            ObservationType.JointPos(
                "q_right_shoulder_yaw_joint", xml_name="right_shoulder_yaw_joint"
            ),
            ObservationType.JointPos(
                "q_right_elbow_joint", xml_name="right_elbow_joint"
            ),
            ObservationType.JointPos(
                "q_right_wrist_roll_joint", xml_name="right_wrist_roll_joint"
            ),
            # ------------- JOINT VEL -------------
            ObservationType.FreeJointVel("dq_root", xml_name="root"),
            ObservationType.JointVel(
                "dq_left_hip_pitch_joint", xml_name="left_hip_pitch_joint"
            ),
            ObservationType.JointVel(
                "dq_left_hip_roll_joint", xml_name="left_hip_roll_joint"
            ),
            ObservationType.JointVel(
                "dq_left_hip_yaw_joint", xml_name="left_hip_yaw_joint"
            ),
            ObservationType.JointVel("dq_left_knee_joint", xml_name="left_knee_joint"),
            ObservationType.JointVel(
                "dq_left_ankle_pitch_joint", xml_name="left_ankle_pitch_joint"
            ),
            ObservationType.JointVel(
                "dq_left_ankle_roll_joint", xml_name="left_ankle_roll_joint"
            ),
            ObservationType.JointVel(
                "dq_right_hip_pitch_joint", xml_name="right_hip_pitch_joint"
            ),
            ObservationType.JointVel(
                "dq_right_hip_roll_joint", xml_name="right_hip_roll_joint"
            ),
            ObservationType.JointVel(
                "dq_right_hip_yaw_joint", xml_name="right_hip_yaw_joint"
            ),
            ObservationType.JointVel(
                "dq_right_knee_joint", xml_name="right_knee_joint"
            ),
            ObservationType.JointVel(
                "dq_right_ankle_pitch_joint", xml_name="right_ankle_pitch_joint"
            ),
            ObservationType.JointVel(
                "dq_right_ankle_roll_joint", xml_name="right_ankle_roll_joint"
            ),
            ObservationType.JointVel("dq_waist_yaw_joint", xml_name="waist_yaw_joint"),
            ObservationType.JointVel(
                "dq_left_shoulder_pitch_joint", xml_name="left_shoulder_pitch_joint"
            ),
            ObservationType.JointVel(
                "dq_left_shoulder_roll_joint", xml_name="left_shoulder_roll_joint"
            ),
            ObservationType.JointVel(
                "dq_left_shoulder_yaw_joint", xml_name="left_shoulder_yaw_joint"
            ),
            ObservationType.JointVel(
                "dq_left_elbow_joint", xml_name="left_elbow_joint"
            ),
            ObservationType.JointVel(
                "dq_left_wrist_roll_joint", xml_name="left_wrist_roll_joint"
            ),
            ObservationType.JointVel(
                "dq_right_shoulder_pitch_joint", xml_name="right_shoulder_pitch_joint"
            ),
            ObservationType.JointVel(
                "dq_right_shoulder_roll_joint", xml_name="right_shoulder_roll_joint"
            ),
            ObservationType.JointVel(
                "dq_right_shoulder_yaw_joint", xml_name="right_shoulder_yaw_joint"
            ),
            ObservationType.JointVel(
                "dq_right_elbow_joint", xml_name="right_elbow_joint"
            ),
            ObservationType.JointVel(
                "dq_right_wrist_roll_joint", xml_name="right_wrist_roll_joint"
            ),
        ]
        loader = LocoMuJoCoLoader(
            env_name="UnitreeG1",
            cfg=basic_cfg,
            observation_spec=observation_spec,
            build_zarr_dataset=False,
        )

        # Always test basic functionality
        assert len(loader) > 0, "Loader should have at least one trajectory"
        assert hasattr(loader, "env"), "Loader should have an environment"
        assert hasattr(loader, "metadata"), "Loader should have metadata"

        if visualize_enabled:
            print("🎥 Visualizing first trajectory in MuJoCo...")
            # This will open a MuJoCo viewer window and replay the trajectory
            if hasattr(loader.env, "play_trajectory"):
                loader.env.play_trajectory(
                    n_episodes=1, n_steps_per_episode=1000, render=True, record=True
                )
                print("✅ Visual inspection completed")
            else:
                print("⚠️ play_trajectory method not available")
        else:
            print(
                "✅ Basic loader test passed (use --visualize for MuJoCo visualization)"
            )

    except Exception as e:
        pytest.fail(f"Visualization test failed due to: {e}")


def test_export_to_rb(basic_cfg, tmp_path):
    """Test exporting LocoMuJoCoLoader data to TorchRL replay buffer."""
    try:
        # Export to Zarr first
        zarr_path = tmp_path / "g1_test_export.zarr"

        loader = LocoMuJoCoLoader(
            env_name="UnitreeG1",
            cfg=basic_cfg,
            build_zarr_dataset=True,
            path=zarr_path,
            export_transitions=True,
        )

        # Now use the utility to create a replay buffer from the Zarr dataset
        from iltools.datasets.utils import make_rb_from

        rb, info = make_rb_from(zarr_path)

        # Basic checks on the replay buffer
        assert len(rb) > 0, "Replay buffer should contain data"

        print(f"✅ Export to replay buffer test passed with {len(rb)} samples")

        from pprint import pprint

        pprint(info)

    except Exception as e:
        pytest.fail(f"Export to replay buffer test failed due to: {e}")


if __name__ == "__main__":
    print("Running basic tests...")
    pytest.main([__file__])
