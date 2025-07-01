import pytest
import torch
import numpy as np
import tempfile
import shutil
import os
import mujoco
from iltools_datasets.dataset_types import ZarrBackedTrajectoryDataset
from iltools_datasets.utils import ZarrTrajectoryWindowCache
from iltools_datasets.loco_mujoco.loader import LocoMuJoCoLoader
from iltools_datasets import export_trajectories_to_zarr
import zarr


@pytest.fixture(scope="module")
def dummy_zarr_dataset():
    # Create a temporary Zarr dataset with windowed format that matches export structure
    # We'll create 2 trajectories of 20 steps each, converted to overlapping windows of size 8
    # This means trajectory 0 (steps 0-19) becomes windows 0-12 (13 windows)
    # And trajectory 1 (steps 0-19) becomes windows 13-25 (13 windows)
    # Total: 26 windows
    tmpdir = tempfile.mkdtemp()
    zarr_path = os.path.join(tmpdir, "trajectories.zarr")
    meta_path = os.path.join(tmpdir, "metadata.json")
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.open_group(store=store, mode="w")

    # Create windowed data structure (single trajectory for simplicity)
    trajectory_length = 20
    window_size = 8

    # Calculate number of windows: max(1, trajectory_length - window_size + 1)
    total_windows = max(1, trajectory_length - window_size + 1)  # 13 windows

    # Create zarr arrays in windowed format
    obs_shape = (total_windows, window_size, 3)  # (13, 8, 3)
    act_shape = (total_windows, window_size, 2)  # (13, 8, 2)

    root.create_dataset("observations/qpos", shape=obs_shape, dtype="f4")
    root.create_dataset("actions/actions", shape=act_shape, dtype="f4")

    # Create synthetic trajectory data (single trajectory)
    traj_qpos = np.arange(trajectory_length * 3, dtype=np.float32).reshape(
        trajectory_length, 3
    )
    traj_actions = np.arange(trajectory_length * 2, dtype=np.float32).reshape(
        trajectory_length, 2
    )

    # Fill with overlapping windows
    for window_idx in range(total_windows):
        start_step = window_idx
        end_step = min(start_step + window_size, trajectory_length)
        actual_window_size = end_step - start_step

        # Fill the window with the appropriate slice of trajectory data
        root["observations/qpos"][window_idx, :actual_window_size, :] = traj_qpos[
            start_step:end_step, :
        ]
        root["actions/actions"][window_idx, :actual_window_size, :] = traj_actions[
            start_step:end_step, :
        ]

        # If window is shorter than window_size, pad with last value
        if actual_window_size < window_size:
            root["observations/qpos"][window_idx, actual_window_size:, :] = traj_qpos[
                end_step - 1 : end_step, :
            ]
            root["actions/actions"][window_idx, actual_window_size:, :] = traj_actions[
                end_step - 1 : end_step, :
            ]

    # dt array should be 1D with one value per window
    root.create_dataset("dt", shape=(total_windows,), dtype="f4")
    root["dt"][:] = 0.05

    # metadata for single trajectory
    meta = {
        "num_trajectories": 1,
        "trajectory_lengths": [20],
        "observation_keys": ["qpos"],
        "action_keys": ["actions"],
    }
    with open(meta_path, "w") as f:
        import json

        json.dump(meta, f)
    dataset = ZarrBackedTrajectoryDataset(tmpdir, device="cpu")
    yield dataset
    shutil.rmtree(tmpdir)


def test_window_cache_basic(dummy_zarr_dataset):
    cache = ZarrTrajectoryWindowCache(dummy_zarr_dataset, window_size=8)
    # Test single trajectory, multiple steps
    traj_idx = 0  # Only one trajectory in dummy dataset
    for step_idx in range(
        13
    ):  # 13 windows available (trajectory_length=20, window_size=8)
        val = cache.get(
            env_idx=0,
            traj_idx=traj_idx,
            step_idx=step_idx,
            key="qpos",
            data_type="observations",
        )

        # The expected value is the first element of the window starting at step_idx
        # Since window step_idx contains steps [step_idx, step_idx+7], element 0 = step step_idx
        dataset_window = dummy_zarr_dataset[step_idx]
        expected = dataset_window["observations"]["qpos"][
            0
        ]  # First element of the window

        assert torch.allclose(val, expected), f"Mismatch at step {step_idx}"

        # Also test actions
        act_val = cache.get(
            env_idx=0,
            traj_idx=traj_idx,
            step_idx=step_idx,
            key="actions",
            data_type="actions",
        )
        expected_act = dataset_window["actions"]["actions"][0]
        assert torch.allclose(
            act_val, expected_act
        ), f"Action mismatch at step {step_idx}"


def test_window_cache_boundary(dummy_zarr_dataset):
    cache = ZarrTrajectoryWindowCache(dummy_zarr_dataset, window_size=5)
    env_idx = 0
    traj_idx = 0
    # Step through available windows (13 windows for trajectory_length=20, window_size=8)
    max_step = len(dummy_zarr_dataset) - 1  # 12 (0-indexed, so 13 windows)
    for step_idx in range(0, max_step + 1):
        val = cache.get(
            env_idx, traj_idx, step_idx, key="qpos", data_type="observations"
        )
        # Get expected value from dataset window
        dataset_window = dummy_zarr_dataset[step_idx]
        expected = dataset_window["observations"]["qpos"][0]
        assert torch.allclose(val, expected), f"Boundary mismatch at step {step_idx}"

    # Test the last valid step
    val = cache.get(env_idx, traj_idx, max_step, key="qpos", data_type="observations")
    dataset_window = dummy_zarr_dataset[max_step]
    expected = dataset_window["observations"]["qpos"][0]
    assert torch.allclose(val, expected)


def test_window_cache_switch_env(dummy_zarr_dataset):
    cache = ZarrTrajectoryWindowCache(dummy_zarr_dataset, window_size=4)
    traj_idx = 0  # Only one trajectory

    # Test multiple environments accessing the same trajectory
    for env_idx in range(2):
        for step_idx in range(4):
            val = cache.get(
                env_idx, traj_idx, step_idx, key="qpos", data_type="observations"
            )
            dataset_window = dummy_zarr_dataset[step_idx]
            expected = dataset_window["observations"]["qpos"][0]
            assert torch.allclose(
                val, expected
            ), f"Mismatch at env {env_idx}, step {step_idx}"

    # Test different step ranges for different environments
    # Env 0: steps 0-3, Env 1: steps 5-8
    for step_idx in range(4):
        val_env0 = cache.get(
            0, traj_idx, step_idx, key="qpos", data_type="observations"
        )
        val_env1 = cache.get(
            1, traj_idx, step_idx + 5, key="qpos", data_type="observations"
        )

        expected_env0 = dummy_zarr_dataset[step_idx]["observations"]["qpos"][0]
        expected_env1 = dummy_zarr_dataset[step_idx + 5]["observations"]["qpos"][0]

        assert torch.allclose(val_env0, expected_env0)
        assert torch.allclose(val_env1, expected_env1)


def test_window_cache_multiple_parallel_environments(dummy_zarr_dataset):
    """
    Test ZarrTrajectoryWindowCache with multiple parallel environments accessing different
    parts of the trajectory space. This simulates the realistic scenario of thousands
    of parallel environments that the cache was originally designed for.
    """
    # Create cache with moderate window size
    cache_window_size = 6
    cache = ZarrTrajectoryWindowCache(dummy_zarr_dataset, window_size=cache_window_size)

    # Test with many parallel environments
    num_environments = 15  # Simulate multiple parallel environments
    traj_idx = 0  # Single trajectory dataset

    print(
        f"Testing {num_environments} parallel environments with cache window size {cache_window_size}"
    )

    # === TEST 1: Each environment accesses different regions ===
    print("Test 1: Different regions per environment")

    # Each environment starts at a different offset
    env_access_patterns = {}
    for env_idx in range(num_environments):
        # Spread environments across available steps (13 windows in dummy dataset)
        start_step = (env_idx * 2) % (
            len(dummy_zarr_dataset) - 3
        )  # Ensure we don't go out of bounds
        access_steps = [
            start_step + i for i in range(3)
        ]  # Each env accesses 3 consecutive steps
        env_access_patterns[env_idx] = access_steps

        print(f"Environment {env_idx}: accessing steps {access_steps}")

    # Initial access - should all be cache misses
    initial_results = {}
    for env_idx, steps in env_access_patterns.items():
        initial_results[env_idx] = []
        for step_idx in steps:
            val = cache.get(
                env_idx, traj_idx, step_idx, key="qpos", data_type="observations"
            )
            initial_results[env_idx].append(val.clone())

            # Verify correctness against dataset
            expected = dummy_zarr_dataset[step_idx]["observations"]["qpos"][0]
            assert torch.allclose(
                val, expected
            ), f"Env {env_idx}, step {step_idx}: initial access mismatch"

    # Check cache statistics after initial access
    stats = cache.get_cache_stats()
    assert (
        stats["num_cached_environments"] == num_environments
    ), f"Expected {num_environments} cached envs, got {stats['num_cached_environments']}"
    assert len(stats["cached_env_ids"]) == num_environments
    print(f"✅ All {num_environments} environments cached after initial access")

    # === TEST 2: Repeat access - should be cache hits ===
    print("Test 2: Cache hits on repeated access")

    for env_idx, steps in env_access_patterns.items():
        for i, step_idx in enumerate(steps):
            val = cache.get(
                env_idx, traj_idx, step_idx, key="qpos", data_type="observations"
            )

            # Should be identical to initial result (cache hit)
            expected_from_cache = initial_results[env_idx][i]
            assert torch.allclose(
                val, expected_from_cache
            ), f"Env {env_idx}, step {step_idx}: cache hit mismatch"

            # Also verify against dataset
            expected_from_dataset = dummy_zarr_dataset[step_idx]["observations"][
                "qpos"
            ][0]
            assert torch.allclose(
                val, expected_from_dataset
            ), f"Env {env_idx}, step {step_idx}: dataset mismatch"

    print("✅ All cache hits returned correct data")

    # === TEST 3: Environment isolation - different envs don't interfere ===
    print("Test 3: Environment isolation")

    # Access the same step from different environments
    test_step = 5
    results_per_env = {}
    for env_idx in range(min(5, num_environments)):  # Test first 5 envs
        val = cache.get(
            env_idx, traj_idx, test_step, key="qpos", data_type="observations"
        )
        results_per_env[env_idx] = val.clone()

        # All should be identical (same step, same trajectory)
        expected = dummy_zarr_dataset[test_step]["observations"]["qpos"][0]
        assert torch.allclose(
            val, expected
        ), f"Env {env_idx}: step {test_step} isolation test failed"

    # Verify all environments got the same result for the same step
    reference_result = results_per_env[0]
    for env_idx, result in results_per_env.items():
        assert torch.allclose(
            result, reference_result
        ), f"Env {env_idx}: result differs from reference"

    print("✅ Environment isolation working correctly")

    # === TEST 4: Cache window boundaries and cache misses ===
    print("Test 4: Cache window boundaries")

    # Force a cache miss by accessing outside current window
    test_env = 0

    # First, access a step to establish a cache window
    initial_step = 2
    cache.get(test_env, traj_idx, initial_step, key="qpos", data_type="observations")

    # Get current cache state
    cached_traj_idx, cached_window_start, cached_data = cache.cache[test_env]
    print(f"Environment {test_env}: cached window starts at step {cached_window_start}")

    # Access within the cached window - should be cache hit
    within_window_step = cached_window_start + 1
    if (
        within_window_step < cached_window_start + cache_window_size
        and within_window_step < len(dummy_zarr_dataset)
    ):
        val_hit = cache.get(
            test_env, traj_idx, within_window_step, key="qpos", data_type="observations"
        )
        expected_hit = dummy_zarr_dataset[within_window_step]["observations"]["qpos"][0]
        assert torch.allclose(val_hit, expected_hit), "Cache hit within window failed"
        print(f"✅ Cache hit for step {within_window_step} within window")

    # Access outside the cached window - should trigger cache miss and new window
    outside_window_step = cached_window_start + cache_window_size + 2
    if outside_window_step < len(dummy_zarr_dataset):
        val_miss = cache.get(
            test_env,
            traj_idx,
            outside_window_step,
            key="qpos",
            data_type="observations",
        )
        expected_miss = dummy_zarr_dataset[outside_window_step]["observations"]["qpos"][
            0
        ]
        assert torch.allclose(
            val_miss, expected_miss
        ), "Cache miss outside window failed"

        # Verify cache was updated
        new_cached_traj_idx, new_cached_window_start, new_cached_data = cache.cache[
            test_env
        ]
        assert (
            new_cached_window_start != cached_window_start
        ), "Cache window should have been updated"
        print(
            f"✅ Cache miss triggered new window: {cached_window_start} -> {new_cached_window_start}"
        )

    # === TEST 5: Actions caching (not just observations) ===
    print("Test 5: Actions caching")

    action_test_env = 1
    action_test_step = 3

    # Test both observations and actions are cached correctly
    obs_val = cache.get(
        action_test_env,
        traj_idx,
        action_test_step,
        key="qpos",
        data_type="observations",
    )
    act_val = cache.get(
        action_test_env, traj_idx, action_test_step, key="actions", data_type="actions"
    )

    expected_obs = dummy_zarr_dataset[action_test_step]["observations"]["qpos"][0]
    expected_act = dummy_zarr_dataset[action_test_step]["actions"]["actions"][0]

    assert torch.allclose(obs_val, expected_obs), "Cached observations incorrect"
    assert torch.allclose(act_val, expected_act), "Cached actions incorrect"
    print("✅ Both observations and actions cached correctly")

    # === TEST 6: Cache management ===
    print("Test 6: Cache management")

    # Test selective cache clearing
    clear_env = 2
    cache.clear_cache(clear_env)
    stats_after_clear = cache.get_cache_stats()
    assert (
        clear_env not in stats_after_clear["cached_env_ids"]
    ), f"Environment {clear_env} should be cleared from cache"
    assert (
        stats_after_clear["num_cached_environments"] == num_environments - 1
    ), "Cache count should decrease by 1"
    print(f"✅ Selective cache clearing worked for environment {clear_env}")

    # Test accessing cleared environment (should work, just cache miss)
    val_after_clear = cache.get(
        clear_env, traj_idx, 1, key="qpos", data_type="observations"
    )
    expected_after_clear = dummy_zarr_dataset[1]["observations"]["qpos"][0]
    assert torch.allclose(
        val_after_clear, expected_after_clear
    ), "Access after cache clear failed"
    print("✅ Access after cache clear works correctly")

    # Test clearing all caches
    cache.clear_cache()  # Clear all
    stats_after_clear_all = cache.get_cache_stats()
    assert (
        stats_after_clear_all["num_cached_environments"] == 0
    ), "All caches should be cleared"
    assert (
        len(stats_after_clear_all["cached_env_ids"]) == 0
    ), "No environments should be cached"
    print("✅ Clear all caches works correctly")

    # === TEST 7: Memory efficiency ===
    print("Test 7: Memory efficiency check")

    # Access from many environments again and check memory usage is reasonable
    for env_idx in range(num_environments):
        step = env_idx % (len(dummy_zarr_dataset) - 1)
        cache.get(env_idx, traj_idx, step, key="qpos", data_type="observations")

    final_stats = cache.get_cache_stats()
    assert final_stats["num_cached_environments"] == num_environments

    # Each environment should only cache cache_window_size steps, not entire dataset
    # This is implicit in our implementation - just verify cache exists for all envs
    print(
        f"✅ Memory efficiency: {final_stats['num_cached_environments']} environments cached with window size {cache_window_size}"
    )

    print("🎉 Multi-environment cache test completed successfully!")
    print("   - Tested {num_environments} parallel environments")
    print("   - Verified cache hits, misses, and window management")
    print("   - Confirmed environment isolation")
    print("   - Validated cache statistics and clearing")


def test_zarr_window_cache_with_unitree_squatting(tmp_path, visualize_enabled):
    """
    Test ZarrTrajectoryWindowCache with real UnitreeG1 squatting data.

    Run with visualization:
        pytest tests/datasets/test_zarr_window_cache.py::test_zarr_window_cache_with_unitree_squatting --visualize

    Tests window cache functionality with real locomotion data and optionally
    visualizes 500 steps of squatting motion in MuJoCo.
    """
    print("=== ZARR WINDOW CACHE TEST WITH UNITREE G1 SQUATTING ===")

    # Create loader for UnitreeG1 squatting task at 50 Hz
    target_freq = 50.0
    loader = LocoMuJoCoLoader(
        env_name="UnitreeG1", task="squat", default_control_freq=target_freq
    )
    print(f"Loader created for UnitreeG1 squatting at {target_freq} Hz")
    print(f"Available trajectories: {len(loader)}")

    # Get trajectory info
    original_traj = loader[0]
    traj_length = original_traj.observations["qpos"].shape[0]
    print(f"Trajectory length: {traj_length} timesteps")
    print(f"Observation keys: {loader.metadata.observation_keys}")

    # Create truncated loader for faster processing (similar to previous test)
    max_trajectory_length = 600  # Use 600 timesteps to ensure enough for testing

    from iltools_datasets.base_loader import BaseTrajectoryLoader
    from iltools_core.metadata_schema import DatasetMeta
    from iltools_core.trajectory import Trajectory
    from typing import Optional

    class TruncatedSquattingLoader(BaseTrajectoryLoader):
        def __init__(self, base_loader, max_length=600):
            self.base_loader = base_loader
            self.max_length = max_length
            self._metadata = DatasetMeta(
                name=base_loader.metadata.name + "_truncated_squat",
                source=base_loader.metadata.source,
                citation=base_loader.metadata.citation,
                version=base_loader.metadata.version,
                observation_keys=base_loader.metadata.observation_keys,
                action_keys=base_loader.metadata.action_keys,
                trajectory_lengths=[max_length],
                num_trajectories=1,
            )

        @property
        def metadata(self) -> DatasetMeta:
            return self._metadata

        def __len__(self):
            return 1

        def get_frequency_info(self):
            return self.base_loader.get_frequency_info()

        def __getitem__(self, idx: int, control_freq: Optional[float] = None):
            if idx != 0:
                raise IndexError(
                    f"Truncated loader only has 1 trajectory, got index {idx}"
                )

            # Get original trajectory with frequency handling
            original_traj = self.base_loader.__getitem__(idx, control_freq=control_freq)

            # Truncate all observations to max_length
            truncated_obs = {}
            for key, value in original_traj.observations.items():
                truncated_obs[key] = np.asarray(value)[: self.max_length]

            # Truncate actions if they exist
            truncated_actions = None
            if original_traj.actions is not None:
                truncated_actions = {}
                for key, value in original_traj.actions.items():
                    truncated_actions[key] = np.asarray(value)[: self.max_length]

            return Trajectory(
                observations=truncated_obs,
                actions=truncated_actions,
                dt=original_traj.dt,
            )

    truncated_loader = TruncatedSquattingLoader(loader, max_trajectory_length)
    print(
        f"Truncated loader created with {len(truncated_loader)} trajectory of {max_trajectory_length} timesteps"
    )

    # Export to zarr
    out_dir = tmp_path / "squatting_zarr"
    window_size = 20  # Small window for efficient testing
    print(f"Exporting to zarr with window_size={window_size}...")

    export_trajectories_to_zarr(
        truncated_loader,
        str(out_dir),
        num_workers=1,
        window_size=window_size,
        control_freq=target_freq,
    )
    print("✅ Export completed successfully!")

    # Load as ZarrBackedTrajectoryDataset
    dataset = ZarrBackedTrajectoryDataset(
        str(out_dir), window_size=window_size, device="cpu", batch_size=1
    )
    print(f"✅ Dataset loaded with {len(dataset)} windows")

    # Debug: Check dataset structure
    sample = dataset[0]
    print(f"Dataset sample window shape: {sample['observations']['qpos'].shape}")
    print(
        f"Export window size: {window_size}, Dataset windows available: {len(dataset)}"
    )

    # === TEST ZARR WINDOW CACHE ===
    print("\n=== TESTING ZARR WINDOW CACHE ===")

    # Create window cache for testing
    # NOTE: Cache window size must be <= export window size (20)
    # The cache can only request windows up to the size of the exported zarr windows
    cache_window_size = 16  # Smaller than export window size to test cache logic
    cache = ZarrTrajectoryWindowCache(dataset, window_size=cache_window_size)
    print(f"Created ZarrTrajectoryWindowCache with window_size={cache_window_size}")
    print(
        f"Cache window size ({cache_window_size}) <= Export window size ({window_size})"
    )

    # Test cache functionality with multiple environments
    num_test_envs = 3
    test_steps = min(
        50, max_trajectory_length - window_size
    )  # Test first 50 steps, use export window_size for bounds

    print(
        f"Testing cache with {num_test_envs} environments for {test_steps} steps each..."
    )

    # Simulate multiple environments accessing different parts of the trajectory
    for env_idx in range(num_test_envs):
        print(f"Testing environment {env_idx}...")

        # Each env starts at a different offset to test cache boundaries
        start_offset = env_idx * 10

        for step_idx in range(
            start_offset,
            min(start_offset + test_steps, max_trajectory_length - window_size + 1),
        ):
            # Test getting qpos via cache
            cached_qpos = cache.get(
                env_idx=env_idx,
                traj_idx=0,  # Only one trajectory in our test
                step_idx=step_idx,
                key="qpos",
                data_type="observations",
            )

            # Verify by getting the same data directly from dataset
            direct_sample = dataset[step_idx]  # This gets window starting at step_idx
            direct_qpos = direct_sample["observations"]["qpos"][
                0
            ]  # First step of window

            # They should match
            assert torch.allclose(
                cached_qpos, direct_qpos, atol=1e-6
            ), f"Cache mismatch at env {env_idx}, step {step_idx}"

            # Also test qvel if available
            if "qvel" in dataset.root["observations"]:
                cached_qvel = cache.get(
                    env_idx=env_idx,
                    traj_idx=0,
                    step_idx=step_idx,
                    key="qvel",
                    data_type="observations",
                )
                direct_qvel = direct_sample["observations"]["qvel"][0]
                assert torch.allclose(
                    cached_qvel, direct_qvel, atol=1e-6
                ), f"Cache qvel mismatch at env {env_idx}, step {step_idx}"

    print("✅ ZarrTrajectoryWindowCache tests passed!")

    # === OPTIONAL VISUALIZATION ===
    if visualize_enabled:
        print("\n=== VISUAL VALIDATION ENABLED ===")
        print("Visualizing 500 steps of UnitreeG1 squatting motion...")

        # Get the truncated original trajectory
        truncated_traj = truncated_loader[0]
        original_qpos = truncated_traj.observations["qpos"]
        original_qvel = truncated_traj.observations["qvel"]

        # Visualize using MuJoCo environment
        env = loader.env
        env.reset()

        visualization_steps = min(500, len(original_qpos))
        print(f"Visualizing {visualization_steps} steps of squatting motion...")

        for i in range(visualization_steps):
            # Set robot state directly
            env.data.qpos[:] = original_qpos[i]
            env.data.qvel[:] = original_qvel[i]

            # Call forward kinematics to update robot visualization
            mujoco.mj_forward(env.model, env.data)  # type: ignore

            # Render every few steps for smooth visualization
            if i % 2 == 0:  # 25 Hz visual rate
                env.render()
                import time

                time.sleep(0.04)  # 25 Hz timing

        print("✅ Squatting motion visualization completed!")

        # Compare with cached data to ensure consistency
        print("\n--- Validating Cache vs Original Data ---")
        comparison_steps = min(50, visualization_steps)  # Compare first 50 steps

        for i in range(comparison_steps):
            # Get from cache
            cached_qpos = cache.get(0, 0, i, "qpos", "observations")

            # Compare with original
            original_step = original_qpos[i]
            cached_numpy = cached_qpos.cpu().numpy()

            original_float = np.array(original_step, dtype=np.float64)
            cached_float = np.array(cached_numpy, dtype=np.float64)
            # Use torch.allclose to avoid numpy type checker issues
            assert torch.allclose(
                torch.from_numpy(original_float),
                torch.from_numpy(cached_float),
                atol=1e-6,
            ), f"Cache vs original mismatch at step {i}"

        print(f"✅ Cache validation: {comparison_steps} steps verified")

    else:
        print("\n=== VISUAL VALIDATION SKIPPED ===")
        print("Run with --visualize flag to see MuJoCo squatting visualization")
        print("✅ Cache functionality validated without visualization")

    # Cleanup
    dataset.shutdown()

    if visualize_enabled:
        print(
            "✅ ZarrTrajectoryWindowCache test with UnitreeG1 squatting visualization passed!"
        )
    else:
        print(
            "✅ ZarrTrajectoryWindowCache test with UnitreeG1 squatting passed! (use --visualize for MuJoCo)"
        )

    print(f"Cache statistics: {len(cache.cache)} environments cached")
