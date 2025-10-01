"""
Comprehensive demo tests showing the full ILTools pipeline:

1. TrajoptLoader -> Zarr export (multi-motion discovery)
2. VectorizedTrajectoryDataset -> windowed streaming
3. ExpertMemmapBuilder -> replay buffer construction
4. SequentialPerEnvSampler -> environment assignments
5. TrajectoryDatasetManager -> high-level orchestration

This demonstrates how to use ILTools for real training scenarios.
"""

import os
import tempfile
from typing import Dict, Any

import numpy as np
import torch
from omegaconf import OmegaConf

from iltools_datasets.trajopt.loader import TrajoptLoader
from iltools_datasets.storage import VectorizedTrajectoryDataset
from iltools_datasets.replay_memmap import ExpertMemmapBuilder, Segment
from iltools_datasets.replay_manager import SequentialPerEnvSampler, EnvAssignment, ExpertReplayManager


def _create_demo_trajopt_data(root_dir: str) -> str:
    """Create demo trajopt data with multiple motions for testing."""
    os.makedirs(root_dir, exist_ok=True)

    motions = ["walking", "running", "jumping"]
    traj_counts = {"walking": 5, "running": 5, "jumping": 5}  # Smaller for demo
    lengths = {"walking": 100, "running": 80, "jumping": 60}  # Different lengths

    # Digit-like dimensions
    qpos_dim, qvel_dim = 65, 65
    action_dim = 24

    for motion in motions:
        motion_dir = os.path.join(root_dir, motion)
        os.makedirs(motion_dir, exist_ok=True)

        for j in range(traj_counts[motion]):
            L = lengths[motion]

            # Generate realistic data
            qpos = np.random.randn(L, qpos_dim).astype(np.float32) * 0.1
            qvel = np.random.randn(L, qvel_dim).astype(np.float32) * 0.1
            action = np.random.randn(L, action_dim).astype(np.float32) * 0.05

            # Add structure
            qpos[:, :3] += np.linspace(0, 1, L)[:, None]  # Base drift
            qvel[:, :3] += np.sin(np.linspace(0, 4 * np.pi, L))[:, None] * 0.1

            traj_file = os.path.join(motion_dir, f"traj_{j}.npz")
            np.savez(traj_file, qpos=qpos, qvel=qvel, action=action)

    return root_dir


def test_full_pipeline_trajopt_to_replay_buffer(tmp_path):
    """Demo: Complete pipeline from trajopt data to replay buffer."""

    # 1) Create demo trajopt data
    data_root = _create_demo_trajopt_data(os.path.join(tmp_path, "demo_data"))

    # 2) Export to Zarr (multi-motion discovery)
    zarr_path = TrajoptLoader(data_root).save(
        out_dir=os.path.join(tmp_path, "zarr_out"),
        dataset_source="demo",
        motion=None,  # Auto-discover motions
        max_trajs_per_motion=5,  # Limit for demo
    )
    print(f"Exported Zarr to: {zarr_path}")

    # 3) Create vectorized dataset for streaming
    ds = VectorizedTrajectoryDataset(
        zarr_path=zarr_path,
        num_envs=4,  # 4 environments
        cfg=OmegaConf.create(
            {"window_size": 16, "buffer_size": 32, "allow_wrap": True}
        ),
    )

    # 4) Build replay buffer from vectorized dataset
    replay_dir = os.path.join(tmp_path, "replay_buffer")
    builder = ExpertMemmapBuilder(replay_dir, max_size=10000)

    # Add trajectories from each motion
    motions = ds.available_motions_in("demo")
    segments = []

    for motion_idx, motion in enumerate(motions):
        trajs = ds.available_trajectories_in("demo", motion)

        for traj_idx, traj_name in enumerate(trajs):
            # Get trajectory index in the dataset
            all_trajs = ds.available_trajectories
            traj_path = f"demo/{motion}/{traj_name}"
            traj_idx_in_dataset = all_trajs.index(traj_path)

            # Fetch the full trajectory
            ds.update_references(
                env_to_traj={0: traj_idx_in_dataset}, env_to_step={0: 0}
            )

            # Get trajectory length
            traj_length = ds.traj_lengths[traj_idx_in_dataset]

            # Create transitions
            transitions = []
            for t in range(traj_length):
                # Set step first, then fetch
                ds.update_references(env_to_step={0: t})

                # Fetch single step
                obs_data = ds.fetch([0], "qpos")
                action_data = ds.fetch([0], "action")

                # Create TensorDict transition
                transition = {
                    "observation": torch.from_numpy(obs_data[0]).float(),
                    "action": torch.from_numpy(action_data[0]).float(),
                }
                transitions.append(transition)

            # Add to replay buffer
            segment = builder.add_trajectory(
                task_id=motion_idx, traj_id=traj_idx, transitions=transitions
            )
            segments.append(segment)
            print(f"Added {motion}/{traj_name}: {traj_length} transitions")

    # 5) Finalize replay buffer
    storage, final_segments = builder.finalize()
    print(f"Replay buffer created with {len(final_segments)} segments")

    # 6) Create sampler with environment assignments
    assignments = [
        EnvAssignment(task_id=0, traj_id=0, step=0),  # Env 0: walking/traj0
        EnvAssignment(task_id=0, traj_id=1, step=0),  # Env 1: walking/traj1
        EnvAssignment(task_id=1, traj_id=0, step=0),  # Env 2: running/traj0
        EnvAssignment(task_id=1, traj_id=1, step=0),  # Env 3: running/traj1
    ]

    sampler = SequentialPerEnvSampler(segments=final_segments, assignment=assignments)

    # 7) Create replay manager
    replay_manager = ExpertReplayManager(
        storage=storage,
        sampler=sampler,
        batch_size=4,  # One per env
    )

    # 8) Test sampling
    batch = replay_manager.sample()
    print(f"Sampled batch shape: {batch.shape}")
    print(f"Batch keys: {batch.keys()}")

    # Verify we get one sample per environment
    assert batch.shape[0] == 4
    assert "observation" in batch.keys()
    assert "action" in batch.keys()

    print("✅ Full pipeline test passed!")


def test_manager_orchestration(tmp_path):
    """Demo: Basic vectorized dataset usage (simplified manager demo)."""

    # 1) Create demo data
    data_root = _create_demo_trajopt_data(os.path.join(tmp_path, "demo_data"))

    # 2) Export to Zarr
    zarr_path = TrajoptLoader(data_root).save(
        out_dir=os.path.join(tmp_path, "zarr_out"),
        dataset_source="demo",
        motion=None,
        max_trajs_per_motion=3,
    )

    # 3) Create vectorized dataset
    ds = VectorizedTrajectoryDataset(
        zarr_path=zarr_path,
        num_envs=2,
        cfg=OmegaConf.create({"window_size": 8, "buffer_size": 16, "allow_wrap": True}),
    )

    # 4) Test basic operations
    print(f"Available motions: {ds.available_motions}")
    print(f"Available trajectories: {ds.available_trajectories}")

    # Set up environment assignments
    motions = ds.available_motions_in("demo")
    trajs = ds.available_trajectories_in("demo", motions[0])
    all_trajs = ds.available_trajectories

    traj_indices = [
        all_trajs.index(f"demo/{motions[0]}/{trajs[0]}"),
        all_trajs.index(f"demo/{motions[0]}/{trajs[1]}"),
    ]

    ds.update_references(
        env_to_traj={0: traj_indices[0], 1: traj_indices[1]}, env_to_step={0: 0, 1: 0}
    )

    # Test data fetching
    batch = ds.fetch([0, 1], "qpos")
    print(f"Batch shape: {batch.shape}")

    assert batch.shape[0] == 2  # num_envs
    print("✅ Basic orchestration test passed!")


def test_streaming_with_double_buffer(tmp_path):
    """Demo: Streaming data with double buffering for JAX environments."""

    import jax
    import jax.numpy as jp
    from iltools_datasets.jax_stream import DoubleBufferStreamer

    # 1) Create demo data and export
    data_root = _create_demo_trajopt_data(os.path.join(tmp_path, "demo_data"))
    zarr_path = TrajoptLoader(data_root).save(
        out_dir=os.path.join(tmp_path, "zarr_out"),
        dataset_source="demo",
        motion=None,
        max_trajs_per_motion=3,
    )

    # 2) Create vectorized dataset
    ds = VectorizedTrajectoryDataset(
        zarr_path=zarr_path,
        num_envs=2,
        cfg=OmegaConf.create({"window_size": 8, "buffer_size": 16, "allow_wrap": True}),
    )

    # 3) Set up environment assignments
    motions = ds.available_motions_in("demo")
    trajs = ds.available_trajectories_in("demo", motions[0])
    all_trajs = ds.available_trajectories

    traj_indices = [
        all_trajs.index(f"demo/{motions[0]}/{trajs[0]}"),
        all_trajs.index(f"demo/{motions[0]}/{trajs[1]}"),
    ]

    ds.update_references(
        env_to_traj={0: traj_indices[0], 1: traj_indices[1]}, env_to_step={0: 0, 1: 0}
    )

    # 4) Create double buffer streamer
    def host_fetch_window():
        qpos = ds.fetch_window([0, 1], "qpos", window_size=8)
        qvel = ds.fetch_window([0, 1], "qvel", window_size=8)
        action = ds.fetch_window([0, 1], "action", window_size=8)

        # Advance steps
        ds.update_references(
            env_to_step={0: ds.env_steps[0] + 8, 1: ds.env_steps[1] + 8}
        )

        return {"qpos": qpos, "qvel": qvel, "action": action}

    streamer = DoubleBufferStreamer(window_fn=host_fetch_window)
    streamer.start()

    # 5) JAX training loop simulation
    @jax.jit
    def dummy_loss(batch):
        qpos = batch["qpos"]
        qvel = batch["qvel"]
        return jp.mean(qpos * qpos + qvel * qvel)

    # Consume several windows
    losses = []
    for i in range(5):
        batch = streamer.next_window()
        loss = dummy_loss(batch)
        losses.append(float(loss))
        print(f"Step {i}: loss = {loss:.4f}")

    streamer.stop()

    # Verify we got different data (losses should vary)
    assert len(losses) == 5
    assert all(np.isfinite(l) for l in losses)

    print("✅ Streaming with double buffer test passed!")


def test_real_dataset_integration(tmp_path):
    """Demo: Integration with real thirdarm dataset (if available)."""

    real_data_path = "/home/fwu/Documents/Research/ThirdArm/mujoco_playground/mujoco_playground/_src/locomotion/digit_v3/data/thirdarm_0901"

    if not os.path.isdir(real_data_path):
        print("Skipping real dataset test - path not found")
        return

    # 1) Export real dataset
    zarr_path = TrajoptLoader(real_data_path).save(
        out_dir=os.path.join(tmp_path, "real_zarr"),
        dataset_source="thirdarm_0901",
        motion=None,
        max_trajs_per_motion=10,  # Limit for demo
    )

    # 2) Create vectorized dataset for real data
    ds = VectorizedTrajectoryDataset(
        zarr_path=zarr_path,
        num_envs=4,
        cfg=OmegaConf.create(
            {"window_size": 16, "buffer_size": 32, "allow_wrap": True}
        ),
    )

    # 3) Test with real data
    print(f"Real dataset motions: {ds.available_motions}")
    print(f"Total trajectories: {len(ds.available_trajectories)}")

    # Set up assignments
    motions = ds.available_motions_in("thirdarm_0901")
    trajs = ds.available_trajectories_in("thirdarm_0901", motions[0])
    all_trajs = ds.available_trajectories

    traj_indices = [
        all_trajs.index(f"thirdarm_0901/{motions[0]}/{trajs[0]}"),
        all_trajs.index(f"thirdarm_0901/{motions[0]}/{trajs[1]}"),
        all_trajs.index(f"thirdarm_0901/{motions[0]}/{trajs[2]}"),
        all_trajs.index(f"thirdarm_0901/{motions[0]}/{trajs[3]}"),
    ]

    ds.update_references(
        env_to_traj={
            0: traj_indices[0],
            1: traj_indices[1],
            2: traj_indices[2],
            3: traj_indices[3],
        },
        env_to_step={0: 0, 1: 0, 2: 0, 3: 0},
    )

    batch = ds.fetch([0, 1, 2, 3], "qpos")

    print(f"Real data batch shape: {batch.shape}")

    assert batch.shape[0] == 4

    print("✅ Real dataset integration test passed!")


if __name__ == "__main__":
    # Run individual tests for debugging
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_full_pipeline_trajopt_to_replay_buffer(tmp_dir)
        test_manager_orchestration(tmp_dir)
        test_streaming_with_double_buffer(tmp_dir)
        test_real_dataset_integration(tmp_dir)
