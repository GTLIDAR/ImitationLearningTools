import pytest
import numpy as np
import torch
import os
from iltools_datasets.loco_mujoco.loader import (
    LocoMuJoCoLoader,
    export_trajectories_to_zarr,
    ZarrBackedTrajectoryDataset,
)
from iltools_datasets.base_loader import BaseTrajectoryDataset, BaseTrajectoryLoader
from pathlib import Path


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
        metadata = type(
            "Meta",
            (),
            {
                "observation_keys": ["qpos"],
                "action_keys": ["actions"],
                "trajectory_lengths": [10, 10],
                "num_trajectories": 2,
            },
        )()

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
    assert isinstance(sample["observations"]["qpos"], torch.Tensor)
    assert sample["observations"]["qpos"].shape == (5, 3)
    assert sample["actions"]["actions"].shape == (5, 2)
    assert isinstance(sample["dt"], torch.Tensor)
    # Test batch collation
    batch = dataset.get_batch()
    assert batch["observations"]["qpos"].shape[0] == 2
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
    assert sample["observations"]["qpos"].device.type == device
    dataset.shutdown()


@pytest.mark.slow
def test_parallel_export_large(tmp_path, minimal_loader):
    # Test parallel export with more data (simulate many trajectories)
    class BigLoader:
        metadata = type(
            "Meta",
            (),
            {
                "observation_keys": ["qpos"],
                "action_keys": ["actions"],
                "trajectory_lengths": [10] * 20,
                "num_trajectories": 20,
            },
        )()

        def __len__(self):
            return 20

        def __getitem__(self, idx):
            class T:
                observations = {"qpos": np.ones((10, 3), dtype=np.float32)}
                actions = {"actions": np.ones((10, 2), dtype=np.float32)}
                dt = 0.05

            return T()

    out_dir = tmp_path / "zarrbig"
    export_trajectories_to_zarr(BigLoader(), str(out_dir), num_workers=4, window_size=5)
    dataset = ZarrBackedTrajectoryDataset(str(out_dir), window_size=5, batch_size=4)
    batch = dataset.get_batch()
    assert batch["observations"]["qpos"].shape[0] == 4
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
        np.savez(tmp_path / f"traj_{i}.npz", **arr)


@pytest.fixture
def fake_loco_dir(tmp_path):
    create_fake_loco_npz(tmp_path, n=3, T=8)
    return tmp_path


def test_loco_loader_len_and_getitem(fake_loco_dir):
    loader = LocoMuJoCoLoader(env_name="Humanoid", task="walk", control_freq=30)
    # Patch loader internals for test
    loader._file_list = sorted(list(Path(fake_loco_dir).glob("*.npz")))
    loader._trajectory_lengths = [8] * 3
    loader._metadata = loader._load_metadata()
    assert len(loader) == 3
    traj = loader[0]
    assert hasattr(traj, "observations")
    assert "qpos" in traj.observations
    assert hasattr(traj, "dt")
    assert loader.metadata.num_trajectories == 3


def test_loco_as_dataset(fake_loco_dir):
    loader = LocoMuJoCoLoader(env_name="Humanoid", task="walk", control_freq=30)
    loader._file_list = sorted(list(Path(fake_loco_dir).glob("*.npz")))
    loader._trajectory_lengths = [8] * 3
    loader._metadata = loader._load_metadata()
    dataset = loader.as_dataset(device="cpu")
    assert len(dataset) == 3
    sample = dataset[0]
    assert "observations" in sample
    assert isinstance(sample["observations"], dict)
    assert "dt" in sample
    assert torch.is_tensor(sample["dt"])
    assert dataset.metadata.num_trajectories == 3


@pytest.mark.visual
def test_visualize_first_trajectory_in_mujoco():
    """
    Visualizes the first trajectory in MuJoCo using the loader's environment.
    This test is for manual/visual inspection and is skipped by default unless --visual is passed.
    """
    loader = LocoMuJoCoLoader(env_name="UnitreeG1", task="walk")
    # Visualize the first trajectory (default behavior)
    # This will open a MuJoCo viewer window and replay the trajectory
    loader.env.play_trajectory(n_episodes=1, n_steps_per_episode=500, render=True)
