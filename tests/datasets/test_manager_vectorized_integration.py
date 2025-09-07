import os
import tempfile
from typing import Tuple

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore
from omegaconf import OmegaConf

from iltools_datasets.manager import TrajectoryDatasetManager

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    import pytest

    pytest.skip("PyTorch not available", allow_module_level=True)


def _make_traj(
    root,
    dataset_source: str,
    motion: str,
    traj_name: str,
    length: int,
    ref_joint_count: int,
) -> None:
    ds_g = root.require_group(dataset_source)
    mot_g = ds_g.require_group(motion)
    traj_g = mot_g.create_group(traj_name)

    qpos_dim = 7 + ref_joint_count
    qvel_dim = 6 + ref_joint_count
    qpos = traj_g.zeros(name="qpos", shape=(length, qpos_dim), dtype=np.float32)
    qvel = traj_g.zeros(name="qvel", shape=(length, qvel_dim), dtype=np.float32)
    # Fill root pos/quat and joint data with patterns
    for t in range(length):
        qpos[t, :3] = [t * 0.1, 0.0, 0.5]
        qpos[t, 3:7] = [1.0, 0.0, 0.0, 0.0]
        qpos[t, 7:] = np.arange(ref_joint_count) + t
        qvel[t, :3] = [0.1, 0.0, 0.0]
        qvel[t, 3:6] = [0.0, 0.0, 0.01]
        qvel[t, 6:] = (np.arange(ref_joint_count) + t) * 0.1


def _build_zarr_dataset(tmp_dir: str, ref_joint_count: int = 5) -> str:
    zarr_path = os.path.join(tmp_dir, "trajectories.zarr")
    store = LocalStore(zarr_path)
    root = zarr.group(store=store, overwrite=True)

    _make_traj(root, "ds1", "walk", "traj0", length=20, ref_joint_count=ref_joint_count)
    _make_traj(
        root, "ds1", "dance", "traj0", length=15, ref_joint_count=ref_joint_count
    )
    return zarr_path


def test_manager_with_vectorized_dataset_end_to_end(tmp_path):
    zarr_path = _build_zarr_dataset(str(tmp_path), ref_joint_count=4)

    # Minimal config for manager
    cfg = OmegaConf.create(
        {
            "dataset_path": str(tmp_path),
            "window_size": 8,
            "buffer_size": 4,
            "assignment_strategy": "round_robin",
            # reference/target joint names overlap partially
            "reference_joint_names": [f"j{i}" for i in range(4)],
            "target_joint_names": ["j0", "jX", "j2", "jY"],
        }
    )

    mgr = TrajectoryDatasetManager(cfg=cfg, num_envs=2, device=torch.device("cpu"))
    # Round-robin assigns predictable trajectories per env
    mgr.reset_trajectories()

    td = mgr.get_reference_data()
    assert td.batch_size == (2,)
    # Root tensors present
    for k in ("root_pos", "root_quat", "root_lin_vel", "root_ang_vel"):
        assert k in td.keys()
    # Joint tensors masked for non-overlapping joints; shape equals len(target_joint_names)
    assert td["joint_pos"].shape[1] == len(cfg.target_joint_names)
    assert td["joint_vel"].shape[1] == len(cfg.target_joint_names)
    # Non-overlapping target joints are NaN
    jp = td["joint_pos"][0]
    # target order: ["j0", "jX", "j2", "jY"] -> expect j0=0, j2=2; jX/jY=nan
    assert jp.shape[0] == 4
    assert torch.isfinite(jp[0]) and torch.allclose(jp[0], torch.tensor(0.0))
    assert torch.isnan(jp[1])
    assert torch.isfinite(jp[2]) and torch.allclose(jp[2], torch.tensor(2.0))
    assert torch.isnan(jp[3])
    # Steps advanced
    assert int(mgr.env2step[0]) == 1 and int(mgr.env2step[1]) == 1
