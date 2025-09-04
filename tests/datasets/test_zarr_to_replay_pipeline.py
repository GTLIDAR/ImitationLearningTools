import os
import numpy as np
import pytest
import zarr

try:
    import torch
    from iltools_datasets.replay_export import build_replay_from_zarr
    TORCH_STACK_AVAILABLE = True
except Exception:
    TORCH_STACK_AVAILABLE = False


pytestmark = pytest.mark.skipif(not TORCH_STACK_AVAILABLE, reason="torch/tensordict/torchrl not available")


def _make_small_zarr(path: str):
    store = zarr.DirectoryStore(path)
    root = zarr.group(store=store, overwrite=True)
    g = root.create_group("ds1").create_group("walk").create_group("traj0")
    T = 6
    # Simple increasing signal for obs; actions are constant ones
    qpos = g.zeros("qpos", shape=(T, 3), dtype=np.float32)
    # Original observation vector from loco-mujoco (for this test, distinct from qpos)
    obs = g.zeros("obs", shape=(T, 4), dtype=np.float32)
    action = g.zeros("action", shape=(T, 2), dtype=np.float32)
    for t in range(T):
        qpos[t, :] = t
        obs[t, :] = t + 100  # distinguish from qpos
        action[t, :] = 1.0


def test_build_replay_from_zarr_and_sample(tmp_path):
    zarr_path = os.path.join(str(tmp_path), "trajectories.zarr")
    _make_small_zarr(zarr_path)

    mgr = build_replay_from_zarr(
        zarr_path=zarr_path,
        scratch_dir=str(tmp_path),
        obs_keys=["qpos"],
        act_key="action",
        concat_obs_to_key="observation",
        include_terminated=True,
        include_truncated=True,
    )

    # Uniform sampler default; sample a batch and verify next_obs - obs == 1 (since qpos increments by 1)
    batch = mgr.buffer.sample()
    obs = batch["observation"]
    next_obs = batch[("next", "observation")]
    diff = next_obs - obs
    assert torch.allclose(diff, torch.ones_like(diff))

    # Terminated/truncated keys exist and are boolean
    assert "terminated" in batch and batch["terminated"].dtype == torch.bool
    assert "truncated" in batch and batch["truncated"].dtype == torch.bool

    # Additional keys present
    assert "qpos" in batch
    assert ("next", "qpos") in batch.keys(True)
    # Original loco-mujoco observation exported
    assert "lmj_observation" in batch
    assert ("next", "lmj_observation") in batch.keys(True)
    # lmj_observation increments by 1 as well (constructed as obs[t]=100+t)
    lmj = batch["lmj_observation"]
    next_lmj = batch[("next", "lmj_observation")]
    assert torch.allclose(next_lmj - lmj, torch.ones_like(lmj))
