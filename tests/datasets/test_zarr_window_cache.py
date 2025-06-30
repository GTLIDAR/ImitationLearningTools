import pytest
import torch
import numpy as np
import tempfile
import shutil
import os
from iltools_datasets.dataset_types import ZarrBackedTrajectoryDataset
from iltools_datasets.utils import ZarrTrajectoryWindowCache
import zarr


@pytest.fixture(scope="module")
def dummy_zarr_dataset():
    # Create a temporary Zarr dataset with 2 trajectories, each 20 steps, 3-dim obs/action
    tmpdir = tempfile.mkdtemp()
    zarr_path = os.path.join(tmpdir, "trajectories.zarr")
    meta_path = os.path.join(tmpdir, "metadata.json")
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.open_group(store=store, mode="w")
    obs_shape = (2, 20, 3)
    act_shape = (2, 20, 2)
    root.create_dataset("observations/qpos", shape=obs_shape, dtype="f4")
    root.create_dataset("actions/actions", shape=act_shape, dtype="f4")
    root["observations/qpos"][:] = np.arange(
        np.prod(obs_shape), dtype=np.float32
    ).reshape(obs_shape)
    root["actions/actions"][:] = np.arange(
        np.prod(act_shape), dtype=np.float32
    ).reshape(act_shape)
    # dt
    root.create_dataset("dt", shape=(2, 20), dtype="f4")
    root["dt"][:] = 0.05
    # metadata
    meta = {
        "num_trajectories": 2,
        "trajectory_lengths": [20, 20],
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
    # Simulate 2 envs, each stepping through their own trajectory
    for env_idx in range(2):
        traj_idx = env_idx
        for step_idx in range(20):
            val = cache.get(
                env_idx, traj_idx, step_idx, key="qpos", data_type="observations"
            )
            # Should match the value in the dataset
            expected = dummy_zarr_dataset.root["observations/qpos"][traj_idx, step_idx]
            assert torch.allclose(
                val, torch.from_numpy(expected)
            ), f"Mismatch at env {env_idx}, step {step_idx}"
            # Also test actions
            act_val = cache.get(
                env_idx, traj_idx, step_idx, key="actions", data_type="actions"
            )
            expected_act = dummy_zarr_dataset.root["actions/actions"][
                traj_idx, step_idx
            ]
            assert torch.allclose(
                act_val, torch.from_numpy(expected_act)
            ), f"Action mismatch at env {env_idx}, step {step_idx}"


def test_window_cache_boundary(dummy_zarr_dataset):
    cache = ZarrTrajectoryWindowCache(dummy_zarr_dataset, window_size=5)
    env_idx = 0
    traj_idx = 0
    # Step through, crossing window boundaries
    for step_idx in range(0, 20):
        val = cache.get(
            env_idx, traj_idx, step_idx, key="qpos", data_type="observations"
        )
        expected = dummy_zarr_dataset.root["observations/qpos"][traj_idx, step_idx]
        assert torch.allclose(
            val, torch.from_numpy(expected)
        ), f"Boundary mismatch at step {step_idx}"
    # Out-of-bounds (should truncate to end)
    val = cache.get(env_idx, traj_idx, 19, key="qpos", data_type="observations")
    expected = dummy_zarr_dataset.root["observations/qpos"][traj_idx, 19]
    assert torch.allclose(val, torch.from_numpy(expected))


def test_window_cache_switch_traj(dummy_zarr_dataset):
    cache = ZarrTrajectoryWindowCache(dummy_zarr_dataset, window_size=4)
    env_idx = 0
    # Start with traj 0
    for step_idx in range(4):
        val = cache.get(env_idx, 0, step_idx, key="qpos", data_type="observations")
        expected = dummy_zarr_dataset.root["observations/qpos"][0, step_idx]
        assert torch.allclose(val, torch.from_numpy(expected))
    # Switch to traj 1
    for step_idx in range(4):
        val = cache.get(env_idx, 1, step_idx, key="qpos", data_type="observations")
        expected = dummy_zarr_dataset.root["observations/qpos"][1, step_idx]
        assert torch.allclose(val, torch.from_numpy(expected))
