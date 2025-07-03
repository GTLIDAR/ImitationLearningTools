import pytest
import tempfile
import shutil
import os
import json
from unittest.mock import Mock, patch
from typing import Any, Dict

import numpy as np
import torch
from tensordict import TensorDict

from iltools_datasets.dataset_manager import TrajectoryDatasetManager
from iltools_datasets.windowed_dataset import WindowedTrajectoryDataset


class MockConfig:
    """Mock configuration class for testing."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestTrajectoryDatasetManager:
    """Test the TrajectoryDatasetManager with all assignment strategies and features."""

    @pytest.fixture
    def sample_zarr_dataset(self):
        """Create a sample Zarr dataset for testing."""
        temp_dir = tempfile.mkdtemp()
        zarr_path = os.path.join(temp_dir, "trajectories.zarr")

        # Create sample data with realistic shapes
        num_trajectories = 8
        max_traj_length = 200

        import zarr

        store = zarr.DirectoryStore(zarr_path)
        root = zarr.group(store=store, overwrite=True)

        # Generate varying trajectory lengths
        traj_lengths = [100, 150, 200, 80, 120, 180, 90, 160]

        # Create observation data (qpos: COM + joints)
        obs_group = root.create_group("observations")
        qpos_data = obs_group.zeros(
            "qpos", shape=(num_trajectories, max_traj_length, 30), dtype=np.float32
        )
        qvel_data = obs_group.zeros(
            "qvel", shape=(num_trajectories, max_traj_length, 29), dtype=np.float32
        )

        # Create action data
        act_group = root.create_group("actions")
        action_data = act_group.zeros(
            "actions", shape=(num_trajectories, max_traj_length, 12), dtype=np.float32
        )

        # Fill with realistic data patterns
        for traj_idx in range(num_trajectories):
            length = traj_lengths[traj_idx]

            for t in range(length):
                # COM position (first 3: x, y, z)
                qpos_data[traj_idx, t, :3] = [
                    t * 0.02,  # Walking forward
                    0.1 * np.sin(t * 0.1),  # Slight lateral sway
                    0.8 + 0.05 * np.sin(t * 0.2),  # COM height variation
                ]

                # COM orientation (next 4: qw, qx, qy, qz)
                angle = t * 0.05
                qpos_data[traj_idx, t, 3:7] = [
                    np.cos(angle / 2),
                    0,
                    0,
                    np.sin(angle / 2),  # Slight yaw rotation
                ]

                # Joint positions (remaining 23 elements)
                phase = t * 0.1 + traj_idx * 0.5  # Different phase per trajectory
                qpos_data[traj_idx, t, 7:] = np.sin(phase + np.arange(23) * 0.3) * 0.5

                # COM linear velocity (first 3: vx, vy, vz)
                qvel_data[traj_idx, t, :3] = [
                    0.02 + 0.01 * np.sin(t * 0.1),  # Forward velocity
                    0.01 * np.cos(t * 0.1),  # Lateral velocity
                    0.005 * np.sin(t * 0.2),  # Vertical velocity
                ]

                # COM angular velocity (next 3: wx, wy, wz)
                qvel_data[traj_idx, t, 3:6] = [
                    0.01 * np.sin(t * 0.1),
                    0.005 * np.cos(t * 0.15),
                    0.05 * np.cos(t * 0.05),
                ]

                # Joint velocities (remaining 23 elements)
                qvel_data[traj_idx, t, 6:] = np.cos(phase + np.arange(23) * 0.3) * 0.1

                # Actions (motor commands)
                action_data[traj_idx, t, :] = (
                    np.sin(t * 0.15 + np.arange(12) * 0.4) * 0.2
                )

        # Create metadata
        metadata = {
            "num_trajectories": num_trajectories,
            "trajectory_lengths": traj_lengths,
            "observation_keys": ["qpos", "qvel"],
            "action_keys": ["actions"],
            "window_size": 64,
            "export_control_freq": 50.0,
            "original_frequency": 100.0,
            "effective_frequency": 50.0,
            "dt": [0.02 for _ in range(num_trajectories)],
        }

        metadata_path = os.path.join(temp_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        yield temp_dir, traj_lengths, num_trajectories

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_initialization_with_existing_zarr(self, sample_zarr_dataset):
        """Test initialization when Zarr dataset already exists."""
        data_dir, traj_lengths, num_trajectories = sample_zarr_dataset

        cfg = MockConfig(
            dataset_path=data_dir, assignment_strategy="random", window_size=64
        )

        num_envs = 16
        device = torch.device("cpu")

        manager = TrajectoryDatasetManager(cfg, num_envs, device)

        assert manager.num_envs == num_envs
        assert manager.device == device
        assert manager.num_trajectories == num_trajectories
        assert len(manager.traj_lengths) == num_trajectories
        assert manager.assignment_strategy == "random"

        # Check that trajectory and step tracking are initialized
        assert manager.env2traj.shape == (num_envs,)
        assert manager.env2step.shape == (num_envs,)
        assert torch.all(manager.env2traj == 0)
        assert torch.all(manager.env2step == 0)

    def test_assignment_strategy_random(self, sample_zarr_dataset):
        """Test random trajectory assignment strategy."""
        data_dir, traj_lengths, num_trajectories = sample_zarr_dataset

        cfg = MockConfig(dataset_path=data_dir, assignment_strategy="random")

        manager = TrajectoryDatasetManager(cfg, 32, torch.device("cpu"))

        # Reset all environments
        manager.reset_trajectories()

        # Check that trajectories are assigned
        assert torch.all(manager.env2traj >= 0)
        assert torch.all(manager.env2traj < num_trajectories)
        assert torch.all(manager.env2step == 0)

        # Check randomness (should have some variation in a large sample)
        unique_trajs = torch.unique(manager.env2traj)
        assert len(unique_trajs) > 1  # Should have multiple different trajectories

    def test_assignment_strategy_sequential(self, sample_zarr_dataset):
        """Test sequential trajectory assignment strategy."""
        data_dir, traj_lengths, num_trajectories = sample_zarr_dataset

        cfg = MockConfig(dataset_path=data_dir, assignment_strategy="sequential")

        manager = TrajectoryDatasetManager(cfg, 16, torch.device("cpu"))
        manager.reset_trajectories()

        # Sequential assignment should follow env_id % num_trajectories
        expected_assignments = torch.arange(16) % num_trajectories
        assert torch.equal(manager.env2traj, expected_assignments)

    def test_assignment_strategy_round_robin(self, sample_zarr_dataset):
        """Test round-robin trajectory assignment strategy."""
        data_dir, traj_lengths, num_trajectories = sample_zarr_dataset

        cfg = MockConfig(dataset_path=data_dir, assignment_strategy="round_robin")

        manager = TrajectoryDatasetManager(cfg, 20, torch.device("cpu"))

        # First reset
        manager.reset_trajectories()
        first_assignments = manager.env2traj.clone()

        # Reset a subset
        subset_envs = torch.tensor([5, 6, 7])
        manager.reset_trajectories(subset_envs)

        # The subset should get the next trajectories in round-robin order
        # Counter should have advanced
        assert manager._round_robin_counter > 0

    def test_assignment_strategy_sequence(self, sample_zarr_dataset):
        """Test custom sequence trajectory assignment strategy."""
        data_dir, traj_lengths, num_trajectories = sample_zarr_dataset

        custom_sequence = [0, 2, 4, 1, 3, 0, 2, 4]  # Custom pattern
        cfg = MockConfig(
            dataset_path=data_dir,
            assignment_strategy="sequence",
            assignment_sequence=custom_sequence,
        )

        manager = TrajectoryDatasetManager(cfg, 16, torch.device("cpu"))
        manager.reset_trajectories()

        # Check that assignments follow the custom sequence
        for env_id in range(16):
            expected_traj = custom_sequence[env_id % len(custom_sequence)]
            assert manager.env2traj[env_id].item() == expected_traj

    def test_assignment_strategy_curriculum(self, sample_zarr_dataset):
        """Test curriculum-based trajectory assignment strategy."""
        data_dir, traj_lengths, num_trajectories = sample_zarr_dataset

        cfg = MockConfig(dataset_path=data_dir, assignment_strategy="curriculum")

        manager = TrajectoryDatasetManager(cfg, 10, torch.device("cpu"))
        manager.reset_trajectories()

        # Curriculum should assign shorter trajectories first
        # Get the trajectory lengths for assigned trajectories
        assigned_lengths = manager.traj_lengths[manager.env2traj]

        # Should generally be shorter trajectories (though with some randomness due to env_id % num_trajectories)
        # At least the first few should be from shorter trajectories
        sorted_lengths, sorted_indices = torch.sort(manager.traj_lengths)

        # Check that at least some environments got shorter trajectories
        short_traj_indices = sorted_indices[: num_trajectories // 2]
        num_short_assigned = torch.sum(torch.isin(manager.env2traj, short_traj_indices))
        assert num_short_assigned > 0

    def test_com_and_joint_extraction(self, sample_zarr_dataset):
        """Test COM and joint data extraction from reference data."""
        data_dir, traj_lengths, num_trajectories = sample_zarr_dataset

        cfg = MockConfig(dataset_path=data_dir, assignment_strategy="sequential")

        manager = TrajectoryDatasetManager(cfg, 4, torch.device("cpu"))
        manager.reset_trajectories()

        # Get reference data
        reference_data = manager.get_reference_data()

        assert isinstance(reference_data, TensorDict)
        assert reference_data.batch_size == (4,)

        # Check COM position extraction (first 3 elements)
        assert "com_pos" in reference_data
        com_pos = reference_data["com_pos"]
        assert com_pos.shape == (4, 3)

        # Verify COM position values are reasonable (walking forward)
        assert torch.all(com_pos[:, 0] >= 0)  # X position should be positive (forward)
        assert torch.all(torch.abs(com_pos[:, 1]) < 0.5)  # Y position should be small
        assert torch.all(com_pos[:, 2] > 0.5)  # Z position should be above ground

        # Check COM orientation extraction (next 4 elements)
        assert "com_quat" in reference_data
        com_quat = reference_data["com_quat"]
        assert com_quat.shape == (4, 4)

        # Quaternions should be normalized
        quat_norms = torch.norm(com_quat, dim=1)
        assert torch.allclose(quat_norms, torch.ones(4), atol=1e-5)

        # Check COM velocities
        assert "com_lin_vel" in reference_data
        assert "com_ang_vel" in reference_data
        com_lin_vel = reference_data["com_lin_vel"]
        com_ang_vel = reference_data["com_ang_vel"]
        assert com_lin_vel.shape == (4, 3)
        assert com_ang_vel.shape == (4, 3)

        # Check joint data extraction
        assert "joint_pos" in reference_data
        assert "joint_vel" in reference_data
        joint_pos = reference_data["joint_pos"]
        joint_vel = reference_data["joint_vel"]
        assert joint_pos.shape == (4, 23)  # 30 - 7 (COM) = 23 joints
        assert joint_vel.shape == (4, 23)  # 29 - 6 (COM vel) = 23 joints

        # Check raw data preservation
        assert "raw_qpos" in reference_data
        assert "raw_qvel" in reference_data
        raw_qpos = reference_data["raw_qpos"]
        raw_qvel = reference_data["raw_qvel"]
        assert raw_qpos.shape == (4, 30)
        assert raw_qvel.shape == (4, 29)

    def test_trajectory_progression_and_completion(self, sample_zarr_dataset):
        """Test trajectory step progression and completion handling."""
        data_dir, traj_lengths, num_trajectories = sample_zarr_dataset

        cfg = MockConfig(dataset_path=data_dir, assignment_strategy="sequential")

        manager = TrajectoryDatasetManager(cfg, 2, torch.device("cpu"))

        # Assign specific trajectories
        manager.env2traj[0] = 3  # Trajectory with length 80
        manager.env2traj[1] = 4  # Trajectory with length 120
        manager.env2step[:] = 0

        # Step through trajectory for env 0 until near completion
        for step in range(78):  # Almost at the end (length 80)
            ref_data = manager.get_reference_data()
            assert manager.env2step[0] == step + 1

            # Verify that we're getting different data as we progress
            com_pos = ref_data["com_pos"][0]
            expected_x = step * 0.02  # Based on our test data pattern
            assert abs(com_pos[0].item() - expected_x) < 0.01

        # Next step should trigger trajectory completion and reset for env 0
        initial_traj_0 = manager.env2traj[0].item()
        ref_data = manager.get_reference_data()

        # Env 0 should have been reset (new trajectory assignment)
        assert manager.env2step[0] == 0  # Reset to beginning
        # Trajectory might have changed due to reset

        # Env 1 should still be progressing normally
        assert manager.env2step[1] == 79  # Still progressing

    def test_partial_environment_reset(self, sample_zarr_dataset):
        """Test resetting only specific environments."""
        data_dir, traj_lengths, num_trajectories = sample_zarr_dataset

        cfg = MockConfig(dataset_path=data_dir, assignment_strategy="sequential")

        manager = TrajectoryDatasetManager(cfg, 8, torch.device("cpu"))
        manager.reset_trajectories()

        # Progress some environments
        for _ in range(10):
            manager.get_reference_data()

        # Reset only specific environments
        reset_envs = torch.tensor([1, 3, 5])
        old_trajs = manager.env2traj.clone()
        old_steps = manager.env2step.clone()

        manager.reset_trajectories(reset_envs)

        # Check that only specified environments were reset
        for env_id in range(8):
            if env_id in reset_envs:
                assert manager.env2step[env_id] == 0  # Reset
                # Trajectory may or may not change depending on strategy
            else:
                assert manager.env2step[env_id] == old_steps[env_id]  # Unchanged
                assert manager.env2traj[env_id] == old_trajs[env_id]  # Unchanged

    def test_data_consistency_across_calls(self, sample_zarr_dataset):
        """Test that data is consistent when accessing the same trajectory step."""
        data_dir, traj_lengths, num_trajectories = sample_zarr_dataset

        cfg = MockConfig(dataset_path=data_dir, assignment_strategy="sequential")

        manager = TrajectoryDatasetManager(cfg, 1, torch.device("cpu"))

        # Set specific trajectory and step
        manager.env2traj[0] = 0
        manager.env2step[0] = 25

        # Get data multiple times from the same step
        data1 = manager.get_reference_data()

        # Reset to same position
        manager.env2traj[0] = 0
        manager.env2step[0] = 25

        data2 = manager.get_reference_data()

        # Data should be identical
        assert torch.allclose(data1["com_pos"], data2["com_pos"], atol=1e-6)
        assert torch.allclose(data1["joint_pos"], data2["joint_pos"], atol=1e-6)
        assert torch.allclose(data1["raw_qpos"], data2["raw_qpos"], atol=1e-6)

    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test missing dataset_path
        cfg_missing_path = MockConfig(assignment_strategy="random")

        with pytest.raises(ValueError, match="dataset_path must be provided"):
            TrajectoryDatasetManager(cfg_missing_path, 4, torch.device("cpu"))

        # Test invalid assignment strategy
        cfg_invalid_strategy = MockConfig(
            dataset_path="/tmp/nonexistent", assignment_strategy="invalid_strategy"
        )

        with pytest.raises(ValueError, match="Unknown assignment strategy"):
            manager = TrajectoryDatasetManager(
                cfg_invalid_strategy, 4, torch.device("cpu")
            )
            manager.reset_trajectories()  # This will trigger the error

        # Test sequence strategy without assignment_sequence
        cfg_no_sequence = MockConfig(
            dataset_path="/tmp/nonexistent", assignment_strategy="sequence"
        )

        with pytest.raises(ValueError, match="assignment_sequence must be provided"):
            manager = TrajectoryDatasetManager(cfg_no_sequence, 4, torch.device("cpu"))
            manager.reset_trajectories()  # This will trigger the error

    def test_device_consistency(self, sample_zarr_dataset):
        """Test that all tensors are on the correct device."""
        data_dir, traj_lengths, num_trajectories = sample_zarr_dataset

        cfg = MockConfig(dataset_path=data_dir, assignment_strategy="random")

        device = torch.device("cpu")
        manager = TrajectoryDatasetManager(cfg, 4, device)

        # Check internal tensors
        assert manager.env2traj.device == device
        assert manager.env2step.device == device
        assert manager.traj_lengths.device == device

        # Check reference data
        ref_data = manager.get_reference_data()
        for key in ref_data.keys():
            assert ref_data[key].device == device

    @pytest.mark.parametrize("batch_size", [1, 4, 16, 64])
    def test_scalability_with_batch_size(self, sample_zarr_dataset, batch_size):
        """Test manager performance with different batch sizes."""
        data_dir, traj_lengths, num_trajectories = sample_zarr_dataset

        cfg = MockConfig(dataset_path=data_dir, assignment_strategy="random")

        manager = TrajectoryDatasetManager(cfg, batch_size, torch.device("cpu"))
        manager.reset_trajectories()

        # Measure time for reference data retrieval
        import time

        start_time = time.time()

        for _ in range(10):
            ref_data = manager.get_reference_data()

        elapsed_time = time.time() - start_time

        # Check that all data has correct batch size
        assert ref_data.batch_size == (batch_size,)
        assert ref_data["com_pos"].shape[0] == batch_size
        assert ref_data["joint_pos"].shape[0] == batch_size

        print(f"Batch size {batch_size}: {elapsed_time:.4f}s for 10 steps")

        # Larger batches should not be dramatically slower per environment
        time_per_env = elapsed_time / (10 * batch_size)
        assert time_per_env < 0.01  # Less than 10ms per environment per step
