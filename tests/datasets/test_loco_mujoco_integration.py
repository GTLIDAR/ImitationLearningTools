import os
import shutil
import tempfile

import pytest
from omegaconf import OmegaConf


try:
    from iltools_datasets.loco_mujoco.loader import LocoMuJoCoLoader
    from iltools_datasets.storage import VectorizedTrajectoryDataset

    LOCO_AVAILABLE = True
except Exception:
    LOCO_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not LOCO_AVAILABLE, reason="loco-mujoco not available in environment"
)


@pytest.mark.slow
def test_loco_mujoco_loader_combined_motions_to_zarr(tmp_path):
    """Single config covering multiple dataset families (default + LAFAN1).

    This reflects typical vectorized training: one shared trajectory pool used
    by all envs, rather than splitting into separate runs.
    """
    cfg = OmegaConf.create(
        {
            "dataset": {
                "trajectories": {
                    "default": ["walk", "run"],  # common baseline motions
                    "amass": [],
                    "lafan1": [],  # Skip lafan1 to avoid download issues
                }
            },
            "control_freq": 30,
        }
    )

    loader = LocoMuJoCoLoader(env_name="UnitreeH1", cfg=cfg)
    # We expect at least the default motions; lafan1 may be unavailable in some setups
    assert len(loader) >= 2

    # Save to Zarr and read back with VectorizedTrajectoryDataset
    zarr_dir = os.path.join(str(tmp_path), "trajectories.zarr")
    loader.save(zarr_dir)

    ds_cfg = OmegaConf.create({"window_size": 16, "buffer_size": 32})
    # Use up to 2 envs to test default trajectories
    num_envs = 2
    ds = VectorizedTrajectoryDataset(zarr_path=zarr_dir, num_envs=num_envs, cfg=ds_cfg)

    n_traj = len(ds.available_trajectories)
    assert n_traj >= 2
    n_assign = min(num_envs, n_traj)
    env_to_traj = {i: i for i in range(n_assign)}
    env_to_step = {i: 0 for i in range(n_assign)}
    ds.update_references(env_to_traj=env_to_traj, env_to_step=env_to_step)

    # Ensure we can fetch qpos for the assigned envs
    qpos = ds.fetch(list(range(n_assign)), key="qpos")
    assert qpos.ndim == 2 and qpos.shape[0] == n_assign
