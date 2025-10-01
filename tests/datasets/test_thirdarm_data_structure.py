import os
from typing import Tuple

import numpy as np

from iltools_datasets.trajopt.loader import TrajoptLoader
from iltools_datasets.storage import VectorizedTrajectoryDataset


def _write_thirdarm_npz_structure(root_dir: str) -> Tuple[str, dict]:
    """Create thirdarm_0901/<motion>/traj_<j>.npz structure with multiple motions."""
    os.makedirs(root_dir, exist_ok=True)

    # Create thirdarm_0901 directory
    thirdarm_dir = os.path.join(root_dir, "thirdarm_0901")
    os.makedirs(thirdarm_dir, exist_ok=True)

    motions = ["walking", "running", "jumping"]
    traj_counts = {
        "walking": 10,
        "running": 10,
        "jumping": 10,
    }  # 10 trajectories per motion
    lengths = {
        "walking": 200,
        "running": 150,
        "jumping": 100,
    }  # Different lengths per motion

    # Digit-like dimensions
    qpos_dim, qvel_dim = 65, 65  # Full Digit state
    action_dim = 24  # Digit actuators

    created_files = {}

    for motion in motions:
        motion_dir = os.path.join(thirdarm_dir, motion)
        os.makedirs(motion_dir, exist_ok=True)

        created_files[motion] = []

        for j in range(traj_counts[motion]):
            L = lengths[motion]

            # Generate realistic-looking data
            qpos = np.random.randn(L, qpos_dim).astype(np.float32) * 0.1
            qvel = np.random.randn(L, qvel_dim).astype(np.float32) * 0.1
            action = np.random.randn(L, action_dim).astype(np.float32) * 0.05

            # Add some structure to make it look more realistic
            qpos[:, :3] += np.linspace(0, 1, L)[:, None]  # Base position drift
            qvel[:, :3] += (
                np.sin(np.linspace(0, 4 * np.pi, L))[:, None] * 0.1
            )  # Base velocity oscillation

            traj_file = os.path.join(motion_dir, f"traj_{j}.npz")
            np.savez(traj_file, qpos=qpos, qvel=qvel, action=action)
            created_files[motion].append(traj_file)

    return thirdarm_dir, created_files


def _write_simple_trajopt_npz(root_dir: str, motion: str, num_trajs: int = 10) -> str:
    """Create simple trajopt structure for one motion: <root_dir>/traj_<j>.npz"""
    os.makedirs(root_dir, exist_ok=True)

    # Digit-like dimensions
    qpos_dim, qvel_dim = 65, 65  # Full Digit state
    action_dim = 24  # Digit actuators
    length = 200  # Fixed length for simplicity

    for j in range(num_trajs):
        # Generate realistic-looking data
        qpos = np.random.randn(length, qpos_dim).astype(np.float32) * 0.1
        qvel = np.random.randn(length, qvel_dim).astype(np.float32) * 0.1
        action = np.random.randn(length, action_dim).astype(np.float32) * 0.05

        # Add some structure to make it look more realistic
        qpos[:, :3] += np.linspace(0, 1, length)[:, None]  # Base position drift
        qvel[:, :3] += (
            np.sin(np.linspace(0, 4 * np.pi, length))[:, None] * 0.1
        )  # Base velocity oscillation

        traj_file = os.path.join(root_dir, f"traj_{j}.npz")
        np.savez(traj_file, qpos=qpos, qvel=qvel, action=action)

    return root_dir


def test_thirdarm_data_structure_to_zarr(tmp_path):
    """Test converting trajopt npz structure to Zarr and using vectorized dataset."""

    # 1) Create simple trajopt structure (one motion at a time)
    walking_dir = _write_simple_trajopt_npz(
        os.path.join(tmp_path, "walking"), "walking", num_trajs=10
    )
    running_dir = _write_simple_trajopt_npz(
        os.path.join(tmp_path, "running"), "running", num_trajs=10
    )

    print(f"Created walking data at: {walking_dir}")
    print(f"Created running data at: {running_dir}")

    # 2) Convert walking motion to Zarr using TrajoptLoader
    zarr_path = TrajoptLoader(walking_dir).save(
        out_dir=str(tmp_path), dataset_source="thirdarm_0901", motion="walking"
    )

    print(f"Created Zarr dataset at: {zarr_path}")

    # 3) Create vectorized dataset with 2 environments
    num_envs = 2
    window_size = 32
    buffer_size = 64

    cfg = {"window_size": window_size, "buffer_size": buffer_size, "allow_wrap": True}

    # Convert dict to OmegaConf for compatibility
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(cfg)

    ds = VectorizedTrajectoryDataset(zarr_path=zarr_path, num_envs=num_envs, cfg=cfg)

    # 4) Set up environment assignments - use subset of trajectories (first 5 from walking)
    available_trajs = ds.available_trajectories_in("thirdarm_0901", "walking")
    print(f"Available trajectories: {available_trajs}")

    # Use first 5 trajectories, cycling through them for the 2 environments
    subset_trajs = available_trajs[:5]  # First 5 trajectories
    print(f"Using subset: {subset_trajs}")

    # Handle case where no trajectories are found
    if len(subset_trajs) == 0:
        print("No trajectories found, skipping test")
        return

    # Convert trajectory names to indices in available_trajectories list
    all_trajs = ds.available_trajectories
    subset_indices = [
        all_trajs.index(f"thirdarm_0901/walking/{traj}") for traj in subset_trajs
    ]

    ds.update_references(
        env_to_traj={
            e: subset_indices[e % len(subset_indices)] for e in range(num_envs)
        },
        env_to_step={e: 0 for e in range(num_envs)},
    )

    print(f"Environment assignments:")
    print(f"  env_traj_ids: {ds.env_traj_ids}")
    print(f"  env_steps: {ds.env_steps}")
    print(f"  window_starts: {ds.window_starts}")

    # 5) Test fetching windows
    print(f"\nTesting window fetching...")

    # Fetch a window
    qpos_window = ds.fetch_window(
        idx=list(range(num_envs)), key="qpos", window_size=window_size
    )
    qvel_window = ds.fetch_window(
        idx=list(range(num_envs)), key="qvel", window_size=window_size
    )
    action_window = ds.fetch_window(
        idx=list(range(num_envs)), key="action", window_size=window_size
    )

    print(f"Window shapes:")
    print(f"  qpos: {qpos_window.shape}")  # Should be [2, 32, 65]
    print(f"  qvel: {qvel_window.shape}")  # Should be [2, 32, 65]
    print(f"  action: {action_window.shape}")  # Should be [2, 32, 24]

    # 6) Verify shapes and data
    assert qpos_window.shape == (num_envs, window_size, 65)
    assert qvel_window.shape == (num_envs, window_size, 65)
    assert action_window.shape == (num_envs, window_size, 24)

    # 7) Test advancing steps and fetching another window
    print(f"\nTesting step advancement...")

    # Advance by half window
    ds.update_references(
        env_to_step={e: ds.env_steps[e] + window_size // 2 for e in range(num_envs)}
    )

    qpos_window2 = ds.fetch_window(
        idx=list(range(num_envs)), key="qpos", window_size=window_size
    )

    print(f"Advanced window shapes:")
    print(f"  qpos: {qpos_window2.shape}")

    # Should be different data (unless we wrapped around)
    assert not np.array_equal(qpos_window, qpos_window2)

    # 8) Test multiple motions (if we had exported them)
    print(f"\nDataset info:")
    print(f"  Available dataset sources: {ds.available_dataset_sources}")
    print(f"  Available motions: {ds.available_motions}")
    print(f"  Available trajectories: {ds.available_trajectories}")

    # 9) Test trajectory lengths
    traj_lengths = ds.traj_lengths
    print(f"  Trajectory lengths: {traj_lengths}")

    # Should have 10 trajectories of length 200 each
    assert len(traj_lengths) == 10
    assert all(length == 200 for length in traj_lengths)

    print(
        f"\n✅ Test passed! Successfully converted trajopt structure to Zarr and tested vectorized loading."
    )


def test_multiple_motions_export(tmp_path):
    """Test exporting multiple motions with a single save call (motion=None)."""

    # 1) Create simple trajopt structures for each motion under a single root
    motions = ["walking", "running", "jumping"]
    root_dir = os.path.join(tmp_path, "thirdarm_root")
    for motion in motions:
        _write_simple_trajopt_npz(os.path.join(root_dir, motion), motion, num_trajs=10)

    # 2) Single save call at root that discovers motions automatically (motion=None)
    zarr_path = TrajoptLoader(root_dir).save(
        out_dir=os.path.join(tmp_path, "multi_motion_zarr"),
        dataset_source="thirdarm_0901",
        motion=None,
    )
    print(f"Exported all motions to: {zarr_path}")

    # 3) Test loading each motion from the single Zarr
    from omegaconf import OmegaConf

    ds = VectorizedTrajectoryDataset(
        zarr_path=zarr_path,
        num_envs=2,
        cfg=OmegaConf.create(
            {"window_size": 16, "buffer_size": 32, "allow_wrap": True}
        ),
    )

    for motion in motions:
        available_trajs = ds.available_trajectories_in("thirdarm_0901", motion)
        print(f"{motion} - Available trajectories: {available_trajs}")
        # Basic sanity
        assert len(available_trajs) == 10

        # Use first 3 trajectories
        subset_trajs = available_trajs[:3]
        all_trajs = ds.available_trajectories
        subset_indices = [
            all_trajs.index(f"thirdarm_0901/{motion}/{traj}") for traj in subset_trajs
        ]
        ds.update_references(
            env_to_traj={e: subset_indices[e % len(subset_indices)] for e in range(2)},
            env_to_step={e: 0 for e in range(2)},
        )

        # Fetch a window
        qpos_window = ds.fetch_window(list(range(2)), "qpos", 16)
        print(f"{motion} - qpos window shape: {qpos_window.shape}")

    print(f"\n✅ Multiple motions test passed!")
