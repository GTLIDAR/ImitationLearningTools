import pytest
import numpy as np
import torch
import os
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


# mujoco viewer is broken on arch or on ubuntu 25.04
# @pytest.mark.visual
# def test_visualize_first_trajectory_in_mujoco():
#     """
#     Visualizes the first trajectory in MuJoCo using the loader's environment.
#     This test is for manual/visual inspection and is skipped by default unless --visual is passed.
#     """
#     loader = LocoMuJoCoLoader(env_name="UnitreeG1", task="walk")
#     # Visualize the first trajectory (default behavior)
#     # This will open a MuJoCo viewer window and replay the trajectory
#     loader.env.play_trajectory(n_episodes=1, n_steps_per_episode=500, render=True)


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
