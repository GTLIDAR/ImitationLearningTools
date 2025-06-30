import pytest
import numpy as np
import torch
from pathlib import Path
from src.iltools_datasets.trajopt.loader import TrajoptLoader, TrajoptTrajectoryDataset


def create_fake_trajopt_npz(tmp_path, n=2, T=10):
    for i in range(n):
        arr = {
            "qpos": np.random.randn(T, 7).astype(np.float32),
            "qvel": np.random.randn(T, 7).astype(np.float32),
            "actions": np.random.randn(T, 3).astype(np.float32),
            "dt": np.array([0.05], dtype=np.float32),
        }
        np.savez(tmp_path / f"traj_{i}.npz", **arr)


@pytest.fixture
def fake_trajopt_dir(tmp_path):
    create_fake_trajopt_npz(tmp_path, n=4, T=6)
    return tmp_path


def test_trajopt_loader_len_and_getitem(fake_trajopt_dir):
    loader = TrajoptLoader(str(fake_trajopt_dir))
    assert len(loader) == 4
    traj = loader[0]
    assert hasattr(traj, "observations")
    assert "qpos" in traj.observations
    assert hasattr(traj, "dt")
    assert loader.metadata.num_trajectories == 4


def test_trajopt_as_dataset(fake_trajopt_dir):
    loader = TrajoptLoader(str(fake_trajopt_dir))
    dataset = loader.as_dataset(device="cpu")
    assert len(dataset) == 4
    sample = dataset[0]
    assert "observations" in sample
    assert isinstance(sample["observations"], dict)
    assert "dt" in sample
    assert torch.is_tensor(sample["dt"])
    assert dataset.metadata.num_trajectories == 4
