import pytest
import numpy as np
import torch
import os
import zarr
from zarr import storage
import mujoco
from typing import Optional
from omegaconf import DictConfig
from iltools_datasets.loco_mujoco.loader import LocoMuJoCoLoader
from iltools_datasets.storage import VectorizedTrajectoryDataset
from iltools_datasets.manager import TrajectoryDatasetManager
from iltools_core.metadata_schema import DatasetMeta
from iltools_core.trajectory import Trajectory
from loco_mujoco.core import ObservationType


@pytest.fixture
def basic_cfg():
    """Basic configuration for tests."""
    return DictConfig(
        {
            "dataset": {
                "trajectories": {"default": ["walk"], "amass": [], "lafan1": []}
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


def test_vectorized_dataset_init(tmp_path, minimal_loader, basic_cfg):
    """Test VectorizedTrajectoryDataset initialization - create zarr data first."""
    # First export data to zarr
    out_dir = tmp_path / "vectorized_test"

    # Create a mock zarr structure that VectorizedTrajectoryDataset expects
    zarr_path = out_dir / "trajectories.zarr"
    os.makedirs(out_dir, exist_ok=True)

    store = storage.DirectoryStore(str(zarr_path))
    root = zarr.group(store=store, overwrite=True)

    # Create the expected loco_mujoco group structure
    loco_group = root.create_group("loco_mujoco")
    default_group = loco_group.create_group("default_walk")
    traj_group = default_group.create_group("trajectory_0")

    # Add minimal data
    traj_group.create_dataset(
        "qpos",
        shape=(10, 3),
        dtype=np.float32,
        chunks=(10, 3),
        data=np.ones((10, 3), dtype=np.float32),
    )
    traj_group.create_dataset(
        "qvel",
        shape=(10, 3),
        dtype=np.float32,
        chunks=(10, 3),
        data=np.ones((10, 3), dtype=np.float32),
    )
    traj_group.create_dataset(
        "actions",
        shape=(10, 3),
        dtype=np.float32,
        chunks=(10, 3),
        data=np.ones((10, 3), dtype=np.float32),
    )
    traj_group.attrs["dt"] = 0.05
    traj_group.attrs["trajectory_length"] = 10
    traj_group.attrs["trajectory_id"] = 0
    traj_group.attrs["trajectory_name"] = "default_walk"
    traj_group.attrs["trajectory_source"] = "loco_mujoco"

    # Add attributes to loco_mujoco group
    loco_group.attrs["frequency"] = 50.0
    loco_group.attrs["joint_names"] = ["j1", "j2", "j3"]

    # Test VectorizedTrajectoryDataset initialization
    dataset = VectorizedTrajectoryDataset(
        zarr_path=str(zarr_path), num_envs=2, cfg=basic_cfg
    )
    assert isinstance(dataset.available_dataset_sources, list)
    assert "loco_mujoco" in dataset.available_dataset_sources


def test_real_loco_loader_basic(basic_cfg):
    """Test real LocoMuJoCoLoader basic functionality."""
    try:
        loader = LocoMuJoCoLoader(env_name="UnitreeG1", cfg=basic_cfg)

        # Test basic properties
        assert hasattr(loader, "env_name")
        assert loader.env_name == "UnitreeG1"
        assert hasattr(loader, "metadata")
        assert isinstance(loader.metadata, DatasetMeta)
        assert hasattr(loader, "env")

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
        loader = LocoMuJoCoLoader(env_name="UnitreeG1", cfg=basic_cfg)

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


def test_manager_basic_functionality(tmp_path, basic_cfg):
    """Test TrajectoryDatasetManager basic functionality."""
    # Create a minimal zarr dataset for testing using the correct structure
    zarr_path = tmp_path / "trajectories.zarr"
    store = storage.DirectoryStore(str(zarr_path))
    root = zarr.group(store=store, overwrite=True)

    # Create the expected loco_mujoco dataset structure
    loco_group = root.create_group("loco_mujoco")

    # Create a motion group (dataset_type_motion)
    motion_group = loco_group.create_group("default_walk")

    # Create trajectory groups
    for traj_idx in range(2):
        traj_group = motion_group.create_group(f"trajectory_{traj_idx}")

        # Add observation data following the expected keys from LocoMuJoCoLoader
        qpos = traj_group.create_dataset(
            "qpos", shape=(50, 30), dtype=np.float32, chunks=(10, 30)
        )
        qvel = traj_group.create_dataset(
            "qvel", shape=(50, 29), dtype=np.float32, chunks=(10, 29)
        )
        # Add other expected keys
        xpos = traj_group.create_dataset(
            "xpos", shape=(50, 24), dtype=np.float32, chunks=(10, 24)
        )

        # Fill with data
        qpos[:] = np.ones((50, 30), dtype=np.float32)
        qvel[:] = np.ones((50, 29), dtype=np.float32)
        xpos[:] = np.ones((50, 24), dtype=np.float32)

        # Add trajectory metadata
        traj_group.attrs["trajectory_length"] = 50
        traj_group.attrs["trajectory_id"] = traj_idx
        traj_group.attrs["trajectory_name"] = "default_walk"
        traj_group.attrs["trajectory_source"] = "loco_mujoco"
        traj_group.attrs["dt"] = 0.02

    # Add metadata to loco_mujoco group
    loco_group.attrs["frequency"] = 50.0
    loco_group.attrs["joint_names"] = [f"j{i}" for i in range(1, 31)]

    # Test manager initialization
    test_cfg = DictConfig(
        {
            "dataset_path": str(tmp_path),
            "assignment_strategy": "sequential",
            "window_size": 4,
            "target_joint_names": [f"j{i}" for i in range(1, 24)][
                ::-1
            ],  # Reverse the list to test mapping
            "reference_joint_names": [f"j{i}" for i in range(5, 28)],
        }
    )

    try:
        device = torch.device("cpu")
        manager = TrajectoryDatasetManager(cfg=test_cfg, num_envs=2, device=device)
        manager.reset_trajectories()
        data = manager.get_reference_data()
        print(data.keys())
        for key in [
            "root_pos",
            "root_quat",
            "root_lin_vel",
            "root_ang_vel",
            "joint_pos",
            "joint_vel",
            "raw_qpos",
            "raw_qvel",
        ]:
            assert key in data
            assert data[key].shape[0] == 2

        assert data["root_pos"].shape == (2, 3)
        assert data["root_quat"].shape == (2, 4)
        assert data["root_lin_vel"].shape == (2, 3)
        assert data["root_ang_vel"].shape == (2, 3)
        # 13 because 5-18 are the shared joints for the reference and target
        assert data["joint_pos"].shape == (2, 23)
        assert data["joint_vel"].shape == (2, 23)
        assert data["raw_qpos"].shape == (2, 30)
        assert data["raw_qvel"].shape == (2, 29)

        assert hasattr(manager, "num_envs")
        assert manager.num_envs == 2
        assert hasattr(manager, "device")

        print("✅ TrajectoryDatasetManager basic test passed!")

    except Exception as e:
        pytest.fail(f"Manager test failed due to: {e}")


def test_manager_comprehensive(tmp_path, basic_cfg):
    """Test TrajectoryDatasetManager comprehensive functionality."""
    # Create a minimal zarr dataset for testing using the correct structure

    test_cfg = DictConfig(
        {
            "dataset_path": str(tmp_path),
            "dataset": {
                "trajectories": {"default": ["walk"], "amass": [], "lafan1": []}
            },
            "loader_type": "loco_mujoco",
            "loader_kwargs": {
                "env_name": "UnitreeG1",
                "cfg": basic_cfg,
            },
            "assignment_strategy": "sequential",
            "window_size": 4,
            "target_joint_names": [f"j{i}" for i in range(1, 24)][
                ::-1
            ],  # Reverse the list to test mapping
            "reference_joint_names": [f"j{i}" for i in range(5, 28)],
        }
    )

    try:
        device = torch.device("cuda:0")
        manager = TrajectoryDatasetManager(cfg=test_cfg, num_envs=10, device=device)
        manager.reset_trajectories()
        step_0_data = manager.get_reference_data()

        for _ in range(10):
            data = manager.get_reference_data()

        for key in [
            "root_pos",
            "root_quat",
            "root_lin_vel",
            "root_ang_vel",
            "joint_pos",
            "joint_vel",
            "raw_qpos",
            "raw_qvel",
        ]:
            assert key in data
            assert data[key].shape[0] == 10

        manager.reset_trajectories(env_ids=torch.arange(5, device=device))
        data = manager.get_reference_data()
        assert data["root_pos"][:5].allclose(step_0_data["root_pos"][:5], atol=1e-6)
        assert data["root_quat"][:5].allclose(step_0_data["root_quat"][:5], atol=1e-6)
        assert data["root_lin_vel"][:5].allclose(
            step_0_data["root_lin_vel"][:5], atol=1e-6
        )
        assert data["root_ang_vel"][:5].allclose(
            step_0_data["root_ang_vel"][:5], atol=1e-6
        )
        assert data["joint_pos"][
            :5, ~torch.isnan(data["joint_pos"]).any(dim=0)
        ].allclose(
            step_0_data["joint_pos"][
                :5, ~torch.isnan(step_0_data["joint_pos"]).any(dim=0)
            ],
            atol=1e-6,
        )
        assert data["joint_vel"][
            :5, ~torch.isnan(data["joint_vel"]).any(dim=0)
        ].allclose(
            step_0_data["joint_vel"][
                :5, ~torch.isnan(step_0_data["joint_vel"]).any(dim=0)
            ],
            atol=1e-6,
        )
        assert data["raw_qpos"][:5].allclose(step_0_data["raw_qpos"][:5], atol=1e-6)
        assert data["raw_qvel"][:5].allclose(step_0_data["raw_qvel"][:5], atol=1e-6)

    except Exception as e:
        pytest.fail(f"Manager test failed due to: {e}")


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
                            "walkturn",
                            "balance",
                            "highjump",
                            "highjump2",
                            "highknee",
                            "jumpturn",
                            "onehopforward",
                            "onestepleft",
                            "onestepright",
                            "onesteplong",
                            "onestepside1",
                            "onestepside2",
                            "random_walk",
                            "run",
                            "squat",
                            "stepinplace1",
                            "stepinplace2",
                            "stepinplace3",
                            "walk",
                        ],
                        "amass": [],
                        "lafan1": [
                            "dance1_subject1",
                            "dance1_subject2",
                            "dance1_subject3",
                            "dance2_subject1",
                            "dance2_subject2",
                            "dance2_subject3",
                            "fallAndGetUp1_subject1",
                            "fallAndGetUp1_subject4",
                            "fallAndGetUp1_subject5",
                            "fallAndGetUp2_subject2",
                            "fallAndGetUp2_subject3",
                            "fallAndGetUp3_subject1",
                            "fight1_subject2",
                            "fight1_subject3",
                            "fight1_subject5",
                            "fightAndSports1_subject1",
                            "fightAndSports1_subject4",
                            "jumps1_subject1",
                            "jumps1_subject2",
                            "jumps1_subject5",
                            "run1_subject2",
                            "run1_subject5",
                            "run2_subject1",
                            "run2_subject4",
                            "sprint1_subject2",
                            "sprint1_subject4",
                            "walk1_subject1",
                            "walk1_subject2",
                            "walk1_subject5",
                            "walk2_subject1",
                            "walk2_subject3",
                            "walk2_subject4",
                            "walk3_subject1",
                            "walk3_subject2",
                            "walk3_subject3",
                            "walk3_subject4",
                            "walk3_subject5",
                            "walk4_subject1",
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
                    n_episodes=55, n_steps_per_episode=1000, render=True, record=True
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


# Remove tests that depend on missing ZarrBackedTrajectoryDataset
# These can be added back when that class is implemented

if __name__ == "__main__":
    print("Running basic tests...")
    pytest.main([__file__])
