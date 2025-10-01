import os
import pytest

from iltools_datasets.trajopt.loader import TrajoptLoader
from iltools_datasets.storage import VectorizedTrajectoryDataset


REAL_DATA_ROOT = (
    "/home/fwu/Documents/Research/ThirdArm/mujoco_playground/"
    "mujoco_playground/_src/locomotion/digit_v3/data/thirdarm_0901"
)


@pytest.mark.slow
def test_real_thirdarm_discover_and_load(tmp_path):
    """Discover motions in the real dataset and verify 10 trajectories per motion.

    - Scans REAL_DATA_ROOT for motion subdirs
    - Exports all motions to a single Zarr via TrajoptLoader(motion=None)
    - Verifies each motion exposes exactly 10 trajectories
    - Creates a minimal VectorizedTrajectoryDataset and fetches a small window
    """
    if not os.path.isdir(REAL_DATA_ROOT):
        pytest.skip(f"Real dataset not found at {REAL_DATA_ROOT}")

    # Export all motions
    zarr_path = TrajoptLoader(REAL_DATA_ROOT).save(
        out_dir=os.path.join(tmp_path, "zarr_out"),
        dataset_source="thirdarm_0901",
        motion=None,
        max_trajs_per_motion=10,
    )

    # Load dataset and verify counts
    from omegaconf import OmegaConf

    ds = VectorizedTrajectoryDataset(
        zarr_path=zarr_path,
        num_envs=2,
        cfg=OmegaConf.create({"window_size": 8, "buffer_size": 16, "allow_wrap": True}),
    )

    motions = ds.available_motions_in("thirdarm_0901")
    assert len(motions) > 0

    for motion in motions:
        trajs = ds.available_trajectories_in("thirdarm_0901", motion)
        assert (
            len(trajs) == 10
        ), f"Expected 10 trajectories in {motion}, got {len(trajs)}"

    # Assign two envs to the first two trajectories of the first motion and fetch a small window
    first_motion = motions[0]
    first_trajs = ds.available_trajectories_in("thirdarm_0901", first_motion)[:2]
    all_trajs = ds.available_trajectories
    traj_indices = [
        all_trajs.index(f"thirdarm_0901/{first_motion}/{t}") for t in first_trajs
    ]
    ds.update_references(
        env_to_traj={0: traj_indices[0], 1: traj_indices[1]}, env_to_step={0: 0, 1: 0}
    )

    qpos = ds.fetch_window([0, 1], key="qpos", window_size=8)
    assert qpos.shape[0] == 2 and qpos.shape[1] == 8
