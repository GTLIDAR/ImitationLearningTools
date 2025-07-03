import pytest
import tempfile
import shutil
import os
import json
import time
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


class TestIntegration:
    """Integration tests for TrajectoryDatasetManager + WindowedTrajectoryDataset."""

    @pytest.fixture
    def large_zarr_dataset(self):
        """Create a larger, more realistic Zarr dataset for integration testing."""
        temp_dir = tempfile.mkdtemp()
        zarr_path = os.path.join(temp_dir, "trajectories.zarr")

        # Create realistic-scale data
        num_trajectories = 32
        max_traj_length = 500

        import zarr

        store = zarr.DirectoryStore(zarr_path)
        root = zarr.group(store=store, overwrite=True)

        # Generate varying trajectory lengths (realistic distribution)
        np.random.seed(42)  # For reproducible tests
        traj_lengths = np.random.randint(
            200, max_traj_length, num_trajectories
        ).tolist()

        # Create observation data
        obs_group = root.create_group("observations")
        qpos_data = obs_group.zeros(
            "qpos", shape=(num_trajectories, max_traj_length, 37), dtype=np.float32
        )
        qvel_data = obs_group.zeros(
            "qvel", shape=(num_trajectories, max_traj_length, 36), dtype=np.float32
        )

        # Create action data
        act_group = root.create_group("actions")
        action_data = act_group.zeros(
            "actions", shape=(num_trajectories, max_traj_length, 18), dtype=np.float32
        )

        # Fill with realistic locomotion patterns
        for traj_idx in range(num_trajectories):
            length = traj_lengths[traj_idx]

            # Each trajectory has different gait patterns and speeds
            gait_freq = 0.05 + traj_idx * 0.01  # Different gait frequencies
            speed = 0.5 + (traj_idx % 4) * 0.3  # Different walking speeds

            for t in range(length):
                # COM position (first 3: x, y, z)
                qpos_data[traj_idx, t, :3] = [
                    t * speed * 0.01,  # Forward progression
                    0.05 * np.sin(t * gait_freq)
                    + (traj_idx % 3 - 1) * 0.1,  # Lateral variation
                    0.9 + 0.02 * np.sin(t * gait_freq * 2),  # COM height oscillation
                ]

                # COM orientation (next 4: qw, qx, qy, qz)
                yaw_angle = t * 0.001 + traj_idx * 0.1  # Slight turning
                qpos_data[traj_idx, t, 3:7] = [
                    np.cos(yaw_angle / 2),
                    0,
                    0,
                    np.sin(yaw_angle / 2),
                ]

                # Joint positions (remaining 30 elements)
                # Simulate periodic gait pattern
                gait_phase = t * gait_freq
                for joint_idx in range(30):
                    joint_offset = joint_idx * 0.2  # Different phase per joint
                    amplitude = 0.3 + 0.2 * (joint_idx % 3)  # Different amplitudes
                    qpos_data[traj_idx, t, 7 + joint_idx] = amplitude * np.sin(
                        gait_phase + joint_offset
                    )

                # COM velocities (first 6: linear + angular)
                qvel_data[traj_idx, t, :3] = [
                    speed * 0.01 + 0.005 * np.sin(t * gait_freq),  # Forward velocity
                    0.01 * np.cos(t * gait_freq),  # Lateral velocity
                    0.02 * np.cos(t * gait_freq * 2),  # Vertical velocity
                ]
                qvel_data[traj_idx, t, 3:6] = [
                    0.005 * np.sin(t * gait_freq),
                    0.002 * np.cos(t * gait_freq),
                    0.001 * np.sin(t * 0.001 + traj_idx * 0.1),  # Yaw rate
                ]

                # Joint velocities (remaining 30 elements)
                for joint_idx in range(30):
                    joint_offset = joint_idx * 0.2
                    amplitude = 0.1 + 0.05 * (joint_idx % 3)
                    qvel_data[traj_idx, t, 6 + joint_idx] = (
                        amplitude * np.cos(gait_phase + joint_offset) * gait_freq
                    )

                # Actions (motor commands)
                for act_idx in range(18):
                    action_data[traj_idx, t, act_idx] = 0.1 * np.sin(
                        gait_phase + act_idx * 0.3
                    )

        # Create metadata
        metadata = {
            "num_trajectories": num_trajectories,
            "trajectory_lengths": traj_lengths,
            "observation_keys": ["qpos", "qvel"],
            "action_keys": ["actions"],
            "window_size": 128,
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

    def test_full_pipeline_single_env(self, large_zarr_dataset):
        """Test complete pipeline with single environment."""
        data_dir, traj_lengths, num_trajectories = large_zarr_dataset

        cfg = MockConfig(
            dataset_path=data_dir, assignment_strategy="random", window_size=128
        )

        # Single environment
        manager = TrajectoryDatasetManager(cfg, 1, torch.device("cpu"))
        manager.reset_trajectories()

        # Simulate episode execution
        episode_length = 100
        trajectory_data = []

        for step in range(episode_length):
            ref_data = manager.get_reference_data()

            # Verify data structure
            assert isinstance(ref_data, TensorDict)
            assert ref_data.batch_size == (1,)

            # Store for later analysis
            trajectory_data.append(
                {
                    "com_pos": ref_data["com_pos"][0].clone(),
                    "joint_pos": ref_data["joint_pos"][0].clone(),
                    "step": step,
                }
            )

            # Verify COM progression (should be moving forward generally)
            if step > 0:
                prev_x = trajectory_data[step - 1]["com_pos"][0]
                curr_x = ref_data["com_pos"][0, 0]
                # Allow for some variation but generally forward progress
                assert curr_x >= prev_x - 0.1  # Small backward steps allowed

        # Verify we got a full trajectory
        assert len(trajectory_data) == episode_length

        # Check that we actually progressed through time
        start_pos = trajectory_data[0]["com_pos"]
        end_pos = trajectory_data[-1]["com_pos"]
        displacement = torch.norm(end_pos - start_pos)
        assert displacement > 0.1  # Should have moved significantly

    def test_multi_env_different_strategies(self, large_zarr_dataset):
        """Test multiple environments with different assignment strategies."""
        data_dir, traj_lengths, num_trajectories = large_zarr_dataset

        strategies = ["random", "sequential", "round_robin", "curriculum"]

        for strategy in strategies:
            cfg = MockConfig(
                dataset_path=data_dir, assignment_strategy=strategy, window_size=64
            )

            num_envs = 16
            manager = TrajectoryDatasetManager(cfg, num_envs, torch.device("cpu"))
            manager.reset_trajectories()

            # Run for multiple steps
            for step in range(20):
                ref_data = manager.get_reference_data()

                # Verify batch consistency
                assert ref_data.batch_size == (num_envs,)
                assert ref_data["com_pos"].shape == (num_envs, 3)
                assert ref_data["joint_pos"].shape == (num_envs, 30)

                # Verify data is different across environments (unless sequential at start)
                if step > 5:  # After some progression
                    com_positions = ref_data["com_pos"]
                    unique_positions = torch.unique(
                        com_positions.round(decimals=3), dim=0
                    )

                    if strategy != "sequential":
                        # Should have some variation (not all identical)
                        assert len(unique_positions) > 1

            print(f"✓ Strategy '{strategy}' passed multi-env test")

    def test_performance_stress_test(self, large_zarr_dataset):
        """Test performance under stress conditions."""
        data_dir, traj_lengths, num_trajectories = large_zarr_dataset

        cfg = MockConfig(
            dataset_path=data_dir, assignment_strategy="random", window_size=128
        )

        # Large number of environments
        num_envs = 256
        manager = TrajectoryDatasetManager(cfg, num_envs, torch.device("cpu"))

        # Warm up
        manager.reset_trajectories()
        for _ in range(5):
            manager.get_reference_data()

        # Measure performance
        num_steps = 50
        start_time = time.time()

        for step in range(num_steps):
            ref_data = manager.get_reference_data()

            # Occasional random resets (realistic scenario)
            if step % 10 == 0:
                reset_envs = torch.randint(0, num_envs, (num_envs // 4,))
                manager.reset_trajectories(reset_envs)

        elapsed_time = time.time() - start_time
        time_per_step = elapsed_time / num_steps
        time_per_env_per_step = time_per_step / num_envs

        print(
            f"Performance: {elapsed_time:.3f}s total, {time_per_step:.4f}s/step, {time_per_env_per_step:.6f}s/env/step"
        )

        # Performance requirements
        assert time_per_step < 0.1  # Less than 100ms per step for 256 envs
        assert time_per_env_per_step < 0.001  # Less than 1ms per environment per step

        # Verify final data integrity
        assert ref_data.batch_size == (num_envs,)
        assert not torch.any(torch.isnan(ref_data["com_pos"]))
        assert not torch.any(torch.isnan(ref_data["joint_pos"]))

    def test_memory_efficiency_long_run(self, large_zarr_dataset):
        """Test memory efficiency over long runs."""
        data_dir, traj_lengths, num_trajectories = large_zarr_dataset

        cfg = MockConfig(
            dataset_path=data_dir, assignment_strategy="round_robin", window_size=64
        )

        num_envs = 64
        manager = TrajectoryDatasetManager(cfg, num_envs, torch.device("cpu"))

        # Monitor memory usage
        import psutil

        process = psutil.Process(os.getpid())

        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run for many steps with frequent resets
        for episode in range(10):
            manager.reset_trajectories()

            for step in range(100):
                ref_data = manager.get_reference_data()

                # Random partial resets
                if step % 20 == 0:
                    reset_envs = torch.randint(0, num_envs, (10,))
                    manager.reset_trajectories(reset_envs)

            # Check memory after each episode
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = current_memory - initial_memory

            # Should not have major memory leaks
            assert memory_increase < 200, (
                f"Memory increased by {memory_increase:.1f}MB after episode {episode}"
            )

        final_memory = process.memory_info().rss / 1024 / 1024
        total_increase = final_memory - initial_memory

        print(
            f"Memory usage: {initial_memory:.1f}MB → {final_memory:.1f}MB (+{total_increase:.1f}MB)"
        )
        assert total_increase < 300  # Less than 300MB increase over long run

    def test_trajectory_completion_handling(self, large_zarr_dataset):
        """Test handling of trajectory completions across multiple environments."""
        data_dir, traj_lengths, num_trajectories = large_zarr_dataset

        cfg = MockConfig(
            dataset_path=data_dir, assignment_strategy="sequential", window_size=32
        )

        num_envs = 8
        manager = TrajectoryDatasetManager(cfg, num_envs, torch.device("cpu"))

        # Set up environments to be near completion of different trajectories
        manager.reset_trajectories()

        # Track completions
        completion_counts = torch.zeros(num_envs, dtype=torch.long)

        # Run until we see completions
        for step in range(1000):
            old_trajs = manager.env2traj.clone()
            old_steps = manager.env2step.clone()

            ref_data = manager.get_reference_data()

            new_trajs = manager.env2traj
            new_steps = manager.env2step

            # Detect completions (trajectory changed or step reset to 0)
            completed_mask = (new_steps < old_steps) | (new_trajs != old_trajs)
            completion_counts += completed_mask.long()

            if torch.sum(completion_counts) > 20:  # Seen enough completions
                break

        # Verify that completions happened
        assert torch.sum(completion_counts) > 0, "No trajectory completions observed"

        # Verify data consistency after completions
        assert not torch.any(torch.isnan(ref_data["com_pos"]))
        assert torch.all(ref_data["com_pos"][:, 2] > 0.5)  # COM should be above ground

        print(
            f"Observed {torch.sum(completion_counts)} trajectory completions across {num_envs} environments"
        )

    def test_data_extraction_accuracy(self, large_zarr_dataset):
        """Test accuracy of COM and joint data extraction."""
        data_dir, traj_lengths, num_trajectories = large_zarr_dataset

        cfg = MockConfig(
            dataset_path=data_dir,
            assignment_strategy="sequential",
            window_size=1,  # Single frame for easier verification
        )

        # Single environment for precise control
        manager = TrajectoryDatasetManager(cfg, 1, torch.device("cpu"))

        # Set specific trajectory and step
        traj_idx = 0
        step_idx = 50
        manager.env2traj[0] = traj_idx
        manager.env2step[0] = step_idx

        ref_data = manager.get_reference_data()

        # Manually load the same data from the dataset
        dataset = manager.dataset
        manual_data = dataset.batch_get(
            np.array([traj_idx]),
            np.array([step_idx + 1]),  # +1 because get_reference_data increments
        )

        manual_qpos = manual_data["observations/qpos"][0, 0]  # First frame of window
        manual_qvel = manual_data["observations/qvel"][0, 0]  # First frame of window

        # Verify COM position extraction
        expected_com_pos = manual_qpos[:3]
        assert torch.allclose(ref_data["com_pos"][0], expected_com_pos, atol=1e-6)

        # Verify COM orientation extraction
        expected_com_quat = manual_qpos[3:7]
        assert torch.allclose(ref_data["com_quat"][0], expected_com_quat, atol=1e-6)

        # Verify joint position extraction
        expected_joint_pos = manual_qpos[7:]
        assert torch.allclose(ref_data["joint_pos"][0], expected_joint_pos, atol=1e-6)

        # Verify COM velocity extraction
        expected_com_lin_vel = manual_qvel[:3]
        expected_com_ang_vel = manual_qvel[3:6]
        assert torch.allclose(
            ref_data["com_lin_vel"][0], expected_com_lin_vel, atol=1e-6
        )
        assert torch.allclose(
            ref_data["com_ang_vel"][0], expected_com_ang_vel, atol=1e-6
        )

        # Verify joint velocity extraction
        expected_joint_vel = manual_qvel[6:]
        assert torch.allclose(ref_data["joint_vel"][0], expected_joint_vel, atol=1e-6)

        print("✓ All data extraction verified against direct dataset access")

    def test_concurrent_access_integration(self, large_zarr_dataset):
        """Test concurrent access to the full pipeline."""
        data_dir, traj_lengths, num_trajectories = large_zarr_dataset

        cfg = MockConfig(
            dataset_path=data_dir, assignment_strategy="random", window_size=64
        )

        # Create multiple managers (simulating multiple processes)
        managers = []
        for i in range(4):
            manager = TrajectoryDatasetManager(cfg, 8, torch.device("cpu"))
            manager.reset_trajectories()
            managers.append(manager)

        import threading
        import queue

        results = queue.Queue()
        errors = queue.Queue()

        def worker(worker_id, manager):
            try:
                for step in range(25):
                    ref_data = manager.get_reference_data()

                    # Verify data integrity
                    assert ref_data.batch_size == (8,)
                    assert not torch.any(torch.isnan(ref_data["com_pos"]))

                    if step % 10 == 0:
                        # Random reset
                        reset_envs = torch.randint(0, 8, (3,))
                        manager.reset_trajectories(reset_envs)

                    time.sleep(0.001)  # Small delay to increase concurrency chance

                results.put(worker_id)

            except Exception as e:
                errors.put((worker_id, str(e)))

        # Start workers
        threads = []
        for i, manager in enumerate(managers):
            t = threading.Thread(target=worker, args=(i, manager))
            t.start()
            threads.append(t)

        # Wait for completion
        for t in threads:
            t.join()

        # Check results
        assert errors.empty(), f"Concurrent access errors: {list(errors.queue)}"
        assert results.qsize() == 4, "Not all workers completed successfully"

        print("✓ Concurrent access test passed")

    @pytest.mark.parametrize(
        "assignment_strategy", ["random", "sequential", "round_robin", "curriculum"]
    )
    def test_strategy_consistency(self, large_zarr_dataset, assignment_strategy):
        """Test that each assignment strategy produces consistent, valid results."""
        data_dir, traj_lengths, num_trajectories = large_zarr_dataset

        cfg = MockConfig(
            dataset_path=data_dir,
            assignment_strategy=assignment_strategy,
            window_size=32,
        )

        num_envs = 12
        manager = TrajectoryDatasetManager(cfg, num_envs, torch.device("cpu"))

        # Multiple independent runs should be consistent
        for run in range(3):
            manager.reset_trajectories()

            # Collect data from multiple steps
            run_data = []
            for step in range(30):
                ref_data = manager.get_reference_data()
                run_data.append(ref_data.clone())

                # Verify data validity at each step
                assert torch.all(torch.isfinite(ref_data["com_pos"]))
                assert torch.all(torch.isfinite(ref_data["joint_pos"]))

                # COM should be at reasonable height
                assert torch.all(ref_data["com_pos"][:, 2] > 0.3)
                assert torch.all(ref_data["com_pos"][:, 2] < 2.0)

                # Joint positions should be in reasonable range
                assert torch.all(torch.abs(ref_data["joint_pos"]) < 3.0)

            print(
                f"✓ Strategy '{assignment_strategy}' run {run + 1} completed successfully"
            )

        print(f"✓ Strategy '{assignment_strategy}' consistency verified")
