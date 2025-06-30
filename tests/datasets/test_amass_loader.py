import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import MagicMock
from src.iltools_datasets.amass.loader import AmassLoader, AmassTrajectoryDataset


def create_fake_amass_npz(tmp_path, n=2, T=10):
    for i in range(n):
        arr = {
            "poses": np.random.randn(T, 66).astype(np.float32),
            "betas": np.random.randn(10).astype(np.float32),
            "trans": np.random.randn(T, 3).astype(np.float32),
            "mocap_framerate": np.array([60.0], dtype=np.float32),
        }
        np.savez(tmp_path / f"seq_{i}.npz", **arr)


@pytest.fixture
def fake_amass_dir(tmp_path):
    create_fake_amass_npz(tmp_path, n=3, T=8)
    return tmp_path


def test_amass_loader_len_and_getitem(monkeypatch, fake_amass_dir):
    # Patch smplx.create to return a mock model
    monkeypatch.setattr(
        "smplx.create",
        lambda *a, **kw: MagicMock(return_value=MagicMock(joints=torch.randn(8, 55))),
    )
    loader = AmassLoader(str(fake_amass_dir), model_path="/dev/null")
    assert len(loader) == 3
    traj = loader[0]
    assert hasattr(traj, "observations")
    assert "qpos" in traj.observations
    assert hasattr(traj, "dt")
    assert loader.metadata.num_trajectories == 3


def test_amass_as_dataset(monkeypatch, fake_amass_dir):
    monkeypatch.setattr(
        "smplx.create",
        lambda *a, **kw: MagicMock(return_value=MagicMock(joints=torch.randn(8, 55))),
    )
    loader = AmassLoader(str(fake_amass_dir), model_path="/dev/null")
    dataset = loader.as_dataset(device="cpu")
    assert len(dataset) == 3
    sample = dataset[0]
    assert "observations" in sample
    assert isinstance(sample["observations"], dict)
    assert "dt" in sample
    assert torch.is_tensor(sample["dt"])
    assert dataset.metadata.num_trajectories == 3
