import os
from typing import Tuple

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore
from omegaconf import OmegaConf

from iltools_datasets.storage import VectorizedTrajectoryDataset


def _create_simple_zarr(root_dir: str) -> Tuple[str, list[int]]:
    zarr_path = os.path.join(root_dir, "trajectories.zarr")
    store = LocalStore(zarr_path)
    root = zarr.group(store=store, overwrite=True)

    # Structure: ds1/walk/traj0, ds1/walk/traj1
    g_ds = root.create_group("ds1")
    g_motion = g_ds.create_group("walk")

    lengths = [10, 12]
    qpos_dim, qvel_dim = 8, 6

    for i, L in enumerate(lengths):
        g_traj = g_motion.create_group(f"traj{i}")
        qpos = g_traj.zeros(name="qpos", shape=(L, qpos_dim), dtype=np.float32)
        qvel = g_traj.zeros(name="qvel", shape=(L, qvel_dim), dtype=np.float32)
        # also create an action key for potential replay tests
        g_traj.zeros(name="action", shape=(L, 3), dtype=np.float32)
        # Fill with recognizable values: qpos[t, 0] = 100*i + t; qvel[t, 0] = 200*i + t
        for t in range(L):
            qpos[t, 0] = 100 * i + t
            qvel[t, 0] = 200 * i + t

    return zarr_path, lengths


def test_vectorized_dataset_listing_and_fetch(tmp_path):
    zarr_path, lengths = _create_simple_zarr(str(tmp_path))
    cfg = OmegaConf.create({"window_size": 8, "buffer_size": 4})

    ds = VectorizedTrajectoryDataset(zarr_path=zarr_path, num_envs=2, cfg=cfg)

    # Listing
    assert ds.available_dataset_sources == ["ds1"]
    assert ds.available_motions == ["walk"]
    assert len(ds.available_trajectories) == 2

    # Assign envs to trajectories and steps
    ds.update_references(env_to_traj={0: 0, 1: 1}, env_to_step={0: 0, 1: 3})

    # Fetch qpos values for current steps
    out_qpos = ds.fetch([0, 1], key="qpos")
    assert out_qpos.shape == (2, 8)
    # env0->traj0@0 => 100*0 + 0
    assert np.allclose(out_qpos[0, 0], 0.0)
    # env1->traj1@3 => 100*1 + 3 = 103
    assert np.allclose(out_qpos[1, 0], 103.0)

    # Move steps to force window refresh (buffer_size=4)
    ds.update_references(env_to_step={0: 3, 1: 6})
    out_qvel = ds.fetch([0, 1], key="qvel")
    assert out_qvel.shape == (2, 6)
    # env0->traj0@3 => 200*0 + 3
    assert np.allclose(out_qvel[0, 0], 3.0)
    # env1->traj1@6 => 200*1 + 6 = 206
    assert np.allclose(out_qvel[1, 0], 206.0)


def test_vectorized_dataset_missing_key_raises(tmp_path):
    zarr_path, _ = _create_simple_zarr(str(tmp_path))
    cfg = OmegaConf.create({"window_size": 8, "buffer_size": 4})
    ds = VectorizedTrajectoryDataset(zarr_path=zarr_path, num_envs=1, cfg=cfg)
    ds.update_references(env_to_traj={0: 0}, env_to_step={0: 0})
    with pytest.raises(KeyError):
        ds.fetch([0], key="nonexistent_key")


def test_vectorized_dataset_step_out_of_range(tmp_path):
    zarr_path, lengths = _create_simple_zarr(str(tmp_path))
    cfg = OmegaConf.create({"window_size": 8, "buffer_size": 4})
    ds = VectorizedTrajectoryDataset(zarr_path=zarr_path, num_envs=1, cfg=cfg)
    ds.update_references(env_to_traj={0: 0}, env_to_step={0: lengths[0]})
    with pytest.raises(IndexError):
        ds.fetch([0], key="qpos")


def test_vectorized_dataset_wrap_steps(tmp_path):
    zarr_path, lengths = _create_simple_zarr(str(tmp_path))
    cfg = OmegaConf.create({"window_size": 8, "buffer_size": 4, "allow_wrap": True})
    ds = VectorizedTrajectoryDataset(zarr_path=zarr_path, num_envs=1, cfg=cfg)
    # Set step beyond end -> should wrap
    ds.update_references(env_to_traj={0: 0}, env_to_step={0: lengths[0] + 2})
    out = ds.fetch([0], key="qpos")
    # Expected wrapped step: (L+2) % L = 2 -> qpos[2,0] = 2
    assert np.allclose(out[0, 0], 2.0)
