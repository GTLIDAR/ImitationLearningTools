import os
from typing import Tuple

import numpy as np
import pytest
from omegaconf import OmegaConf

from iltools_datasets.trajopt.loader import TrajoptLoader
from iltools_datasets.storage import VectorizedTrajectoryDataset


def _write_trajopt_npz(root_dir: str) -> Tuple[str, list[int]]:
    os.makedirs(root_dir, exist_ok=True)
    lengths = [10, 12]
    qpos_dim, qvel_dim, act_dim = 8, 6, 3

    for i, L in enumerate(lengths):
        qpos = np.zeros((L, qpos_dim), dtype=np.float32)
        qvel = np.zeros((L, qvel_dim), dtype=np.float32)
        act = np.zeros((L, act_dim), dtype=np.float32)
        for t in range(L):
            qpos[t, :] = 100 * i + t
            qvel[t, :] = 200 * i + t
            act[t, :] = 300 * i + t
        np.savez(
            os.path.join(root_dir, f"traj_{i}.npz"), qpos=qpos, qvel=qvel, action=act
        )

    return root_dir, lengths


def test_trajopt_save_to_vectorized_and_fetch(tmp_path):
    npz_dir, lengths = _write_trajopt_npz(os.path.join(tmp_path, "npz"))

    loader = TrajoptLoader(data_path=npz_dir)
    zarr_path = loader.save(
        out_dir=str(tmp_path), dataset_source="trajopt", motion="digit_v3"
    )

    cfg = OmegaConf.create({"window_size": 8, "buffer_size": 4})
    ds = VectorizedTrajectoryDataset(zarr_path=zarr_path, num_envs=2, cfg=cfg)

    # Listing mirrors vectorized tests
    assert ds.available_dataset_sources == ["trajopt"]
    assert ds.available_motions == ["digit_v3"]
    assert len(ds.available_trajectories) == 2

    ds.update_references(env_to_traj={0: 0, 1: 1}, env_to_step={0: 0, 1: 3})
    out_qpos = ds.fetch([0, 1], key="qpos")
    assert out_qpos.shape == (2, 8)
    assert np.allclose(out_qpos[0, 0], 0.0)
    assert np.allclose(out_qpos[1, 0], 103.0)

    ds.update_references(env_to_step={0: 3, 1: 6})
    out_qvel = ds.fetch([0, 1], key="qvel")
    assert out_qvel.shape == (2, 6)
    assert np.allclose(out_qvel[0, 0], 3.0)
    assert np.allclose(out_qvel[1, 0], 206.0)
