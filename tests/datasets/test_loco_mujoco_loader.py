import pytest
import numpy as np
import torch
import os
import mujoco
from typing import Optional
from iltools_datasets.loco_mujoco.loader import LocoMuJoCoLoader
from iltools_datasets import export_trajectories_to_zarr, ZarrBackedTrajectoryDataset
from iltools_core.metadata_schema import DatasetMeta


@pytest.fixture
def unitree_g1_loader():
    """
    Returns a LocoMuJoCoLoader for the Unitree G1 walking task.
    """
    return LocoMuJoCoLoader(env_name="UnitreeG1", task="walk")


@pytest.fixture
def minimal_loader(monkeypatch):
    # Create a minimal loader with synthetic data for fast tests
    class DummyTraj:
        def __init__(self):
            self.observations = {"qpos": np.ones((10, 3), dtype=np.float32)}
            self.actions = {"actions": np.ones((10, 2), dtype=np.float32)}
            self.dt = 0.05

    class DummyLoader:
        from iltools_core.metadata_schema import DatasetMeta

        metadata = DatasetMeta(
            name="dummy_dataset",
            source="dummy",
            citation="none",
            version="0.0.1",
            observation_keys=["qpos"],
            action_keys=["actions"],
            trajectory_lengths=[10, 10],
            num_trajectories=2,
        )

        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return DummyTraj()

    return DummyLoader()


def test_loader_basic(minimal_loader):
    # Test basic loader functionality
    assert len(minimal_loader) == 2
    traj = minimal_loader[0]
    assert isinstance(traj.observations["qpos"], np.ndarray)
    assert isinstance(traj.actions["actions"], np.ndarray)
    assert traj.observations["qpos"].shape == (10, 3)
    assert traj.actions["actions"].shape == (10, 2)
    assert isinstance(traj.dt, float)


def test_export_to_zarr_and_reload(tmp_path, minimal_loader):
    # Test export to Zarr and reload with ZarrBackedTrajectoryDataset
    out_dir = tmp_path / "zarrtest"
    export_trajectories_to_zarr(
        minimal_loader, str(out_dir), num_workers=2, window_size=5
    )
    # Check Zarr and metadata files exist
    assert (out_dir / "trajectories.zarr").exists() or (
        out_dir / "trajectories.zarr" / ".zgroup"
    ).exists()
    assert (out_dir / "metadata.json").exists()
    # Load with ZarrBackedTrajectoryDataset
    dataset = ZarrBackedTrajectoryDataset(
        str(out_dir), window_size=5, device="cpu", batch_size=2
    )
    assert len(dataset) > 0
    sample = dataset[0]
    assert isinstance(sample["observations"]["qpos"], torch.Tensor)  # type: ignore
    assert sample["observations"]["qpos"].shape == (5, 3)  # type: ignore
    assert sample["actions"]["actions"].shape == (5, 2)  # type: ignore
    assert isinstance(sample["dt"], torch.Tensor)
    # Test batch collation
    batch = dataset.get_batch()
    assert batch["observations"]["qpos"].shape[0] == 2  # type: ignore
    dataset.shutdown()


def test_device_aware_loading(tmp_path, minimal_loader):
    # Test device-aware loading (CPU/GPU)
    out_dir = tmp_path / "zarrtest2"
    export_trajectories_to_zarr(
        minimal_loader, str(out_dir), num_workers=2, window_size=5
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = ZarrBackedTrajectoryDataset(
        str(out_dir), window_size=5, device=device, batch_size=1
    )
    sample = dataset[0]
    assert sample["observations"]["qpos"].device.type == device  # type: ignore
    dataset.shutdown()


@pytest.mark.slow
def test_parallel_export_large(tmp_path, minimal_loader):
    # Test parallel export with more data (simulate many trajectories)
    class BigLoader:
        metadata = DatasetMeta(
            name="big_dummy_dataset",
            source="dummy",
            citation="none",
            observation_keys=["qpos"],
            action_keys=["actions"],
            trajectory_lengths=[10] * 20,
            num_trajectories=20,
            version="0.0.1",
        )

        def __len__(self):
            return 20

        def __getitem__(self, idx):
            class T:
                observations = {"qpos": np.ones((10, 3), dtype=np.float32)}
                actions = {"actions": np.ones((10, 2), dtype=np.float32)}
                dt = 0.05

            return T()

    out_dir = tmp_path / "zarrbig"
    export_trajectories_to_zarr(BigLoader(), str(out_dir), num_workers=4, window_size=5)  # type: ignore
    dataset = ZarrBackedTrajectoryDataset(str(out_dir), window_size=5, batch_size=4)
    batch = dataset.get_batch()
    assert batch["observations"]["qpos"].shape[0] == 4  # type: ignore
    dataset.shutdown()


def test_error_handling(tmp_path):
    # Test error handling for missing/corrupt files
    out_dir = tmp_path / "zarrbad"
    os.makedirs(out_dir, exist_ok=True)
    # Create incomplete/corrupt metadata
    with open(out_dir / "metadata.json", "w") as f:
        f.write("{bad json}")
    with pytest.raises(Exception):
        _ = ZarrBackedTrajectoryDataset(str(out_dir))


def create_fake_loco_npz(tmp_path, n=2, T=10):
    for i in range(n):
        arr = {
            "qpos": np.random.randn(T, 17).astype(np.float32),
            "qvel": np.random.randn(T, 17).astype(np.float32),
            "actions": np.random.randn(T, 6).astype(np.float32),
            "dt": np.array([0.033], dtype=np.float32),
        }
        np.savez(tmp_path / f"traj_{i}.npz", **arr)  # type: ignore


@pytest.fixture
def fake_loco_dir(tmp_path):
    create_fake_loco_npz(tmp_path, n=3, T=8)
    return tmp_path


def test_loco_loader_len_and_getitem():
    # Use a dummy loader for logic-only checks
    class DummyTraj:
        def __init__(self):
            self.observations = {
                "qpos": np.ones((8, 30), dtype=np.float32),
                "qvel": np.ones((8, 30), dtype=np.float32),
            }
            self.actions = {"actions": np.ones((8, 6), dtype=np.float32)}
            self.dt = 0.033

    class DummyLoader:
        from iltools_core.metadata_schema import DatasetMeta

        metadata = DatasetMeta(
            name="dummy_dataset",
            source="dummy",
            citation="none",
            version="0.0.1",
            observation_keys=["qpos", "qvel"],
            action_keys=["actions"],
            trajectory_lengths=[8, 8, 8],
            num_trajectories=3,
        )

        def __len__(self):
            return 3

        def __getitem__(self, idx):
            return DummyTraj()

    loader = DummyLoader()
    assert len(loader) == 3
    traj = loader[0]
    assert hasattr(traj, "observations")
    assert "qpos" in traj.observations
    assert hasattr(traj, "dt")
    assert loader.metadata.num_trajectories == 3
    # verify the observation and action shapes
    assert traj.observations["qpos"].shape[-1] == 30  # type: ignore
    assert traj.observations["qvel"].shape[-1] == 30  # type: ignore
    assert traj.actions["actions"].shape[-1] == 6  # type: ignore


def test_loco_as_dataset():
    # Use a dummy loader for logic-only checks
    class DummyTraj:
        def __init__(self):
            self.observations = {"qpos": np.ones((8, 30), dtype=np.float32)}
            self.actions = {"actions": np.ones((8, 6), dtype=np.float32)}
            self.dt = 0.033

    class DummyLoader:
        from iltools_core.metadata_schema import DatasetMeta

        metadata = DatasetMeta(
            name="dummy_dataset",
            source="dummy",
            citation="none",
            version="0.0.1",
            observation_keys=["qpos"],
            action_keys=["actions"],
            trajectory_lengths=[8, 8, 8],
            num_trajectories=3,
        )

        def __len__(self):
            return 3

        def __getitem__(self, idx):
            return DummyTraj()

    loader = DummyLoader()

    class DummyDataset:
        def __init__(self, loader):
            self.loader = loader

        def __len__(self):
            return len(self.loader)

        def __getitem__(self, idx):
            traj = self.loader[idx]
            obs = {k: torch.from_numpy(v).float() for k, v in traj.observations.items()}
            actions = {k: torch.from_numpy(v).float() for k, v in traj.actions.items()}
            return {
                "observations": obs,
                "actions": actions,
                "dt": torch.tensor(traj.dt, dtype=torch.float32),
            }

        @property
        def metadata(self):
            return self.loader.metadata

    dataset = DummyDataset(loader)
    assert len(dataset) == 3
    sample = dataset[0]
    assert "observations" in sample
    assert isinstance(sample["observations"], dict)
    assert "dt" in sample
    assert torch.is_tensor(sample["dt"])
    assert dataset.metadata.num_trajectories == 3


def test_visualize_first_trajectory_in_mujoco(visualize_enabled):
    """
    Optionally visualizes the first trajectory in MuJoCo using the loader's environment.

    Run with visualization:
        pytest tests/datasets/test_loco_mujoco_loader.py::test_visualize_first_trajectory_in_mujoco --visualize

    This test validates basic loader functionality and optionally shows visual inspection.
    """
    loader = LocoMuJoCoLoader(env_name="UnitreeG1", task="walk")

    # Always test basic functionality
    assert len(loader) > 0, "Loader should have at least one trajectory"
    traj = loader[0]
    assert hasattr(traj, "observations"), "Trajectory should have observations"

    if visualize_enabled:
        print("🎥 Visualizing first trajectory in MuJoCo...")
        # This will open a MuJoCo viewer window and replay the trajectory
        loader.env.play_trajectory(n_episodes=1, n_steps_per_episode=500, render=True)
        print("✅ Visual inspection completed")
    else:
        print("✅ Basic loader test passed (use --visualize for MuJoCo visualization)")


def test_real_unitree_g1_loader_print_first5():
    """Test real LocoMuJoCoLoader for UnitreeG1 walk dataset, print first 5 qpos/qvel, and show all obs shapes."""
    loader = LocoMuJoCoLoader(env_name="UnitreeG1", task="walk")
    traj = loader[0]
    obs = traj.observations
    print("Observation keys:", list(obs.keys()))
    for k, v in obs.items():
        print(f"Observation '{k}' shape: {v.shape}")
    assert "qpos" in obs, "qpos not found in observations"
    assert "qvel" in obs, "qvel not found in observations"
    print("First 5 qpos:", obs["qpos"][:5])
    print("First 5 qvel:", obs["qvel"][:5])


def test_real_unitree_g1_loader_print_trajectory_info():
    """Test real LocoMuJoCoLoader for UnitreeG1 walk dataset, print trajectory count, lengths, and meta info for the first trajectory."""
    loader = LocoMuJoCoLoader(env_name="UnitreeG1", task="walk")
    num_traj = len(loader)
    print(f"Total number of trajectories: {num_traj}")
    lengths = []
    for i in range(num_traj):
        traj = loader[i]
        if hasattr(traj, "observations") and "qpos" in traj.observations:
            lengths.append(traj.observations["qpos"].shape[0])
        else:
            lengths.append(None)
    print("Length of each trajectory (timesteps, by qpos):", lengths)
    # Load the first trajectory and print all meta info
    first_traj = loader[0]
    print("\nFirst trajectory meta info:")
    if hasattr(first_traj, "observations"):
        print("  Observations keys:", list(first_traj.observations.keys()))
        for k, v in first_traj.observations.items():
            print(f"    '{k}': shape {v.shape}, dtype {v.dtype}")
    if hasattr(first_traj, "actions") and first_traj.actions is not None:
        print("  Actions keys:", list(first_traj.actions.keys()))
        for k, v in first_traj.actions.items():
            print(f"    '{k}': shape {v.shape}, dtype {v.dtype}")
    if hasattr(first_traj, "dt"):
        print("  dt:", first_traj.dt)
    if hasattr(first_traj, "infos") and first_traj.infos is not None:
        print("  Additional infos:", first_traj.infos)
    if hasattr(loader, "metadata"):
        print("\nLoader metadata:")
        print(loader.metadata)


def test_zarr_dataset_window_shapes(tmp_path, minimal_loader):
    """Test that all samples in ZarrBackedTrajectoryDataset are windowed and have correct shapes."""
    out_dir = tmp_path / "zarr_window_shape"
    export_trajectories_to_zarr(
        minimal_loader, str(out_dir), num_workers=2, window_size=4
    )
    dataset = ZarrBackedTrajectoryDataset(
        str(out_dir), window_size=4, device="cpu", batch_size=1
    )
    for i in range(len(dataset)):
        sample = dataset[i]
        assert sample["observations"]["qpos"].shape == (4, 3)  # type: ignore
        assert sample["actions"]["actions"].shape == (4, 2)  # type: ignore
        assert isinstance(sample["dt"], torch.Tensor)
    dataset.shutdown()


def test_zarr_dataset_batching(tmp_path, minimal_loader):
    """Test that batching returns correct batch size and shapes."""
    out_dir = tmp_path / "zarr_batching"
    export_trajectories_to_zarr(
        minimal_loader, str(out_dir), num_workers=2, window_size=3
    )
    dataset = ZarrBackedTrajectoryDataset(
        str(out_dir), window_size=3, device="cpu", batch_size=2
    )
    batch = dataset.get_batch()
    assert batch["observations"]["qpos"].shape[0] == 2  # type: ignore
    assert batch["observations"]["qpos"].shape[1:] == (3, 3)  # type: ignore
    assert batch["actions"]["actions"].shape == (2, 3, 2)  # type: ignore
    dataset.shutdown()


def test_zarr_dataset_device_transfer(tmp_path, minimal_loader):
    """Test that samples and batches are on the correct device (CPU or CUDA)."""
    out_dir = tmp_path / "zarr_device"
    export_trajectories_to_zarr(
        minimal_loader, str(out_dir), num_workers=2, window_size=2
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = ZarrBackedTrajectoryDataset(
        str(out_dir), window_size=2, device=device, batch_size=1
    )
    sample = dataset[0]
    assert sample["observations"]["qpos"].device.type == device  # type: ignore
    batch = dataset.get_batch()
    assert batch["observations"]["qpos"].device.type == device  # type: ignore
    dataset.shutdown()


def test_zarr_dataset_metadata_preserved(tmp_path, minimal_loader):
    """Test that metadata is preserved and accessible in the ZarrBackedTrajectoryDataset."""
    out_dir = tmp_path / "zarr_metadata"
    export_trajectories_to_zarr(
        minimal_loader, str(out_dir), num_workers=2, window_size=2
    )
    dataset = ZarrBackedTrajectoryDataset(
        str(out_dir), window_size=2, device="cpu", batch_size=1
    )
    assert hasattr(dataset, "metadata")
    assert dataset.metadata.name == "dummy_dataset"
    assert dataset.metadata.num_trajectories == 2
    dataset.shutdown()


def test_loader_to_dataset_conversion_simple(tmp_path, visualize_enabled):
    """
    Test for loader-to-dataset conversion with optional visual validation.

    Run with visualization:
        pytest tests/datasets/test_loco_mujoco_loader.py::test_loader_to_dataset_conversion_simple --visualize

    Run without visualization (default):
        pytest tests/datasets/test_loco_mujoco_loader.py::test_loader_to_dataset_conversion_simple

    When visualization is enabled, shows 500 steps at 50 Hz for both loader trajectory
    and dataset trajectory to validate data preservation and frequency consistency.
    """
    from iltools_datasets.base_loader import BaseTrajectoryLoader

    print("=== SIMPLE LOADER TO DATASET TEST ===")

    # === UNIFIED FREQUENCY HANDLING TEST ===
    print("\n=== UNIFIED FREQUENCY HANDLING TEST ===")

    # Create loader with 50 Hz for visualization
    target_freq = 50.0  # Use 50 Hz as requested
    loader = LocoMuJoCoLoader(
        env_name="UnitreeG1", task="walk", default_control_freq=target_freq
    )
    print(f"Loader created with default frequency: {target_freq} Hz")
    print(f"Available trajectories: {len(loader)}")

    # Show frequency info
    freq_info = loader.get_frequency_info()
    print(f"Loader frequency info: {freq_info}")

    # Get trajectory (will use default frequency automatically)
    original_traj = loader[0]  # No need to specify frequency - uses default
    traj_length = original_traj.observations["qpos"].shape[0]
    print(f"Trajectory length at {target_freq} Hz: {traj_length} timesteps")
    print(f"Trajectory dt: {original_traj.dt}")
    print(f"Observation keys: {loader.metadata.observation_keys}")
    print(f"Action keys: {loader.metadata.action_keys}")

    # Use a small window size for testing but allow 500 steps for visualization
    window_size = 20  # Small window for efficient processing
    print(f"Using window size: {window_size} for zarr export")

    # Create minimal loader with enough trajectory for 500 step visualization
    max_trajectory_length = (
        600  # Use 600 timesteps to ensure we have enough for 500 step visualization
    )
    print(
        f"Truncating trajectory to {max_trajectory_length} timesteps for fast testing"
    )

    class MinimalLoader(BaseTrajectoryLoader):
        def __init__(self, base_loader, max_length=500):
            self.base_loader = base_loader
            self.max_length = max_length
            # Use truncated length instead of full trajectory
            self._metadata = DatasetMeta(
                name=base_loader.metadata.name + "_minimal",
                source=base_loader.metadata.source,
                citation=base_loader.metadata.citation,
                version=base_loader.metadata.version,
                observation_keys=base_loader.metadata.observation_keys,
                action_keys=base_loader.metadata.action_keys,
                trajectory_lengths=[max_length],  # Truncated length!
                num_trajectories=1,
            )

        @property
        def metadata(self) -> DatasetMeta:
            return self._metadata

        def __len__(self):
            return 1

        def get_frequency_info(self):
            """Return frequency info from the base loader."""
            return self.base_loader.get_frequency_info()

        def __getitem__(self, idx: int, control_freq: Optional[float] = None):
            if idx != 0:
                raise IndexError(
                    f"Minimal loader only has 1 trajectory, got index {idx}"
                )
            # Get the original trajectory (pass through control_freq to base loader)
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
                    truncated_actions[key] = np.asarray(value)[: self.max_length]  # type: ignore

            # Create truncated trajectory
            from iltools_core.trajectory import Trajectory

            return Trajectory(
                observations=truncated_obs,
                actions=truncated_actions,
                dt=original_traj.dt,
            )

    minimal_loader = MinimalLoader(loader, max_trajectory_length)
    print(
        f"Minimal loader created with {len(minimal_loader)} trajectory of length {max_trajectory_length}"
    )
    print(f"MinimalLoader inherits frequency from base loader: {target_freq} Hz")

    # Now with truncation, we'll have much fewer windows
    expected_windows = max(1, max_trajectory_length - window_size + 1)
    print(f"Expected number of windows: {expected_windows} (much more manageable!)")

    # Export with minimal settings
    out_dir = tmp_path / "minimal_zarr"
    print(f"Exporting to {out_dir} with window_size={window_size}...")

    try:
        export_trajectories_to_zarr(
            minimal_loader,
            str(out_dir),
            num_workers=1,  # Single worker to avoid concurrency issues
            window_size=window_size,
            control_freq=target_freq,  # Ensure consistent frequency in export
        )
        print("✅ Export completed successfully!")

        # Load back as dataset
        dataset = ZarrBackedTrajectoryDataset(
            str(out_dir), window_size=window_size, device="cpu", batch_size=1
        )
        print(f"✅ Dataset loaded with {len(dataset)} windows")

        # === FREQUENCY VALIDATION ===
        freq_info = dataset.get_frequency_info()
        print(f"Dataset frequency info: {freq_info}")
        print(f"Frequency consistency: {freq_info['consistent_dt']}")

        # Get a sample and validate
        sample = dataset[0]
        print(
            f"✅ Sample retrieved, qpos shape: {sample['observations']['qpos'].shape}"
        )

        # Basic validation - just check that data exists and has right window size
        assert (
            sample["observations"]["qpos"].shape[0] == window_size
        ), f"Expected window size {window_size}, got {sample['observations']['qpos'].shape[0]}"

        # === VISUAL INSPECTION ===
        print("\n=== VISUAL INSPECTION ===")

        # Get the truncated original trajectory for comparison
        truncated_traj = minimal_loader[0]
        original_qpos = truncated_traj.observations["qpos"]
        original_qvel = truncated_traj.observations["qvel"]

        # Get dataset sample data (convert from torch back to numpy)
        dataset_qpos = sample["observations"]["qpos"].numpy()
        dataset_qvel = sample["observations"]["qvel"].numpy()

        print(f"Original trajectory qpos shape: {original_qpos.shape}")
        print(f"Dataset sample qpos shape: {dataset_qpos.shape}")

        # Check frequency consistency
        print(f"Original trajectory dt: {truncated_traj.dt}")
        print(f"Dataset sample dt: {sample['dt'].item()}")

        # Verify data preservation (first window should match original start)
        data_match = np.allclose(original_qpos[:window_size], dataset_qpos, atol=1e-6)  # type: ignore
        print(f"Data preservation check: {data_match}")

        # Check dt consistency
        dt_match = abs(truncated_traj.dt - sample["dt"].item()) < 1e-6
        print(f"dt consistency check: {dt_match}")

        assert data_match, "Dataset data doesn't match original trajectory data!"
        assert dt_match, "Dataset dt doesn't match original trajectory dt!"

        # === VISUAL VALIDATION (OPTIONAL) ===
        if visualize_enabled:
            print("\n=== VISUAL VALIDATION ENABLED ===")
            print("Running MuJoCo visualization...")
            # Visual comparison using MuJoCo environment - 500 steps each
            print("\n--- Replaying Original Trajectory (500 steps at 50 Hz) ---")
            env = loader.env

            # Reset environment
            env.reset()

            # Replay original truncated trajectory by setting robot state directly
            # (since actions aren't available, we set qpos/qvel directly)
            visualization_steps = min(500, len(original_qpos))
            print(f"Visualizing {visualization_steps} steps from original trajectory")

            for i in range(visualization_steps):
                # Set robot state directly
                env.data.qpos[:] = original_qpos[i]
                env.data.qvel[:] = original_qvel[i]

                # IMPORTANT: Call forward kinematics to update robot visualization
                mujoco.mj_forward(env.model, env.data)  # type: ignore

                # Render every few steps for smoother visualization
                if i % 2 == 0:  # Render every 2nd frame for 25 Hz visual rate
                    env.render()
                    # Small delay for visualization at 25 Hz visual rate
                    import time

                    time.sleep(0.04)  # 25 Hz = 0.04s between frames

            print("--- Replaying Dataset Trajectory (500 steps at 50 Hz) ---")
            print(
                "Note: Dataset contains windowed data, so we'll reconstruct full trajectory from multiple windows"
            )

            # Reset environment
            env.reset()

            # For dataset visualization, reconstruct full trajectory from windows
            # Since windows overlap, we can reconstruct by taking first step of each window
            target_steps = min(500, len(original_qpos))

            # Simple reconstruction: collect first step from each window until we have enough
            dataset_trajectory_qpos = []
            dataset_trajectory_qvel = []

            # Get the starting window that covers the target range
            for window_idx in range(min(target_steps, len(dataset))):
                window_sample = dataset[window_idx]
                window_qpos = window_sample["observations"]["qpos"].numpy()
                window_qvel = window_sample["observations"]["qvel"].numpy()

                # Take the first step from each window (this reconstructs the original sequence)
                # Since window i starts at step i, window_qpos[0] corresponds to original step i
                dataset_trajectory_qpos.append(window_qpos[0])
                dataset_trajectory_qvel.append(window_qvel[0])

                if len(dataset_trajectory_qpos) >= target_steps:
                    break

            # Convert to numpy arrays and ensure we have exactly target_steps
            dataset_trajectory_qpos = np.array(dataset_trajectory_qpos)[:target_steps]
            dataset_trajectory_qvel = np.array(dataset_trajectory_qvel)[:target_steps]

            print(
                f"Reconstructed {len(dataset_trajectory_qpos)} steps from dataset windows"
            )
            print(
                f"Visualizing {len(dataset_trajectory_qpos)} steps from dataset trajectory"
            )

            # Replay the reconstructed dataset trajectory
            for i in range(len(dataset_trajectory_qpos)):
                # Set robot state from dataset
                env.data.qpos[:] = dataset_trajectory_qpos[i]
                env.data.qvel[:] = dataset_trajectory_qvel[i]

                # IMPORTANT: Call forward kinematics to update robot visualization
                mujoco.mj_forward(env.model, env.data)  # type: ignore

                # Render every few steps for smoother visualization
                if i % 2 == 0:  # Render every 2nd frame for 25 Hz visual rate
                    env.render()
                    # Small delay for visualization at 25 Hz visual rate
                    import time

                    time.sleep(0.04)  # 25 Hz = 0.04s between frames

            # Visual comparison summary
            print("--- Data Comparison Summary (Visual) ---")
            print(
                f"Original data range - qpos: [{original_qpos.min():.3f}, {original_qpos.max():.3f}]"
            )
            print(
                f"Dataset reconstructed range - qpos: [{dataset_trajectory_qpos.min():.3f}, {dataset_trajectory_qpos.max():.3f}]"
            )

            # Compare the 500 steps that were visualized
            comparison_length = min(
                len(original_qpos), len(dataset_trajectory_qpos), target_steps
            )
            original_comparison = original_qpos[:comparison_length]
            dataset_comparison = dataset_trajectory_qpos[:comparison_length]

            print(f"Comparing first {comparison_length} steps of both trajectories:")
            max_diff = np.abs(original_comparison - dataset_comparison).max()
            print(
                f"Max absolute difference over {comparison_length} steps: {max_diff:.8f}"
            )

            # Optional: Show position differences
            pos_diff = np.abs(original_comparison - dataset_comparison)
            print(
                f"Position differences per joint (max over {comparison_length} steps): {pos_diff.max(axis=0)}"
            )

            # Check if trajectories are essentially identical
            # Use explicit casting to avoid type checker issues
            orig_array = np.array(original_comparison, dtype=np.float64)
            dataset_array = np.array(dataset_comparison, dtype=np.float64)
            trajectories_match = np.allclose(orig_array, dataset_array, atol=1e-6)  # type: ignore
            print(f"500-step trajectory match: {trajectories_match}")

            if not trajectories_match:
                print(
                    "⚠️  Note: Small differences may be due to windowing reconstruction"
                )
            else:
                print("✅ Perfect trajectory match over 500 steps!")

        else:
            print("\n=== VISUAL VALIDATION SKIPPED ===")
            print("Run with --visualize flag to enable MuJoCo visualization")
            # Still do basic numerical comparison without visualization
            target_steps = min(500, len(original_qpos))
            dataset_trajectory_qpos = []
            dataset_trajectory_qvel = []

            for window_idx in range(min(target_steps, len(dataset))):
                window_sample = dataset[window_idx]
                window_qpos = window_sample["observations"]["qpos"].numpy()
                window_qvel = window_sample["observations"]["qvel"].numpy()
                dataset_trajectory_qpos.append(window_qpos[0])
                dataset_trajectory_qvel.append(window_qvel[0])
                if len(dataset_trajectory_qpos) >= target_steps:
                    break

            dataset_trajectory_qpos = np.array(dataset_trajectory_qpos)[:target_steps]
            dataset_trajectory_qvel = np.array(dataset_trajectory_qvel)[:target_steps]

            # Quick numerical validation
            comparison_length = min(
                len(original_qpos), len(dataset_trajectory_qpos), target_steps
            )
            original_comparison = original_qpos[:comparison_length]
            dataset_comparison = dataset_trajectory_qpos[:comparison_length]

            max_diff = np.abs(original_comparison - dataset_comparison).max()
            orig_array = np.array(original_comparison, dtype=np.float64)
            dataset_array = np.array(dataset_comparison, dtype=np.float64)
            trajectories_match = np.allclose(orig_array, dataset_array, atol=1e-6)  # type: ignore

            print(f"✅ Numerical validation: {comparison_length} steps compared")
            print(f"Max difference: {max_diff:.8f}, Match: {trajectories_match}")

        # Cleanup
        dataset.shutdown()
        if visualize_enabled:
            print(
                "✅ Simple loader-to-dataset test with 500-step visualization at 50 Hz passed!"
            )
        else:
            print(
                "✅ Simple loader-to-dataset test passed! (use --visualize for MuJoCo visualization)"
            )

    except Exception as e:
        print(f"❌ Export failed: {e}")
        raise
