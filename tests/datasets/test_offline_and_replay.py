import json
import os
from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore

import torch
from iltools_datasets.offline import OfflineDataset
from iltools_datasets.reference import JointMapping, ReferenceManager


def _make_simple_offline_root(root_dir: Path) -> OfflineDataset:
    """Create a tiny offline dataset with one dataset_source, one motion, two trajs.

    Layout:
        <root>/trajectories.zarr/ds1/walk/traj0/{qpos, obs, action}
        <root>/trajectories.zarr/ds1/walk/traj1/{qpos, obs, action}
    """
    zarr_path = root_dir / "trajectories.zarr"
    store = LocalStore(str(zarr_path))
    root = zarr.group(store=store, overwrite=True)
    g0 = root.create_group("ds1").create_group("walk").create_group("traj0")
    g1 = root["ds1"]["walk"].create_group("traj1")

    T0, T1 = 6, 4
    for g, T, base in [(g0, T0, 0), (g1, T1, 100)]:
        qpos = g.zeros(name="qpos", shape=(T, 3), dtype=np.float32)
        obs = g.zeros(name="obs", shape=(T, 4), dtype=np.float32)
        action = g.zeros(name="action", shape=(T, 2), dtype=np.float32)
        for t in range(T):
            qpos[t, :] = t + base
            obs[t, :] = t + base + 0.5
            action[t, :] = 1.0

    # Minimal metadata file
    meta = {
        "name": "toy_loco",
        "source": "unit_test",
        "version": "0.0.1",
        "citation": "N/A",
        "num_trajectories": 2,
        "keys": ["qpos", "obs", "action"],
        "trajectory_lengths": [T0, T1],
        "dt": 0.0167,
        "joint_names": [],
        "body_names": [],
        "site_names": [],
        "metadata": {},
    }
    with (root_dir / "meta.json").open("w") as f:
        json.dump(meta, f)

    return OfflineDataset.from_zarr_root(root_dir)


def test_offline_dataset_introspection(tmp_path: Path):
    offline = _make_simple_offline_root(tmp_path)

    # Introspection of trajectories
    assert offline.num_trajectories == 2
    assert offline.total_steps == 6 + 4

    # Check that trajectories are properly discovered with lengths
    lengths = sorted(t.length for t in offline.iter_trajectories())
    assert lengths == [4, 6]

    # Metadata round-trip
    assert offline.metadata is not None
    assert offline.metadata.name == "toy_loco"
    assert offline.metadata.num_trajectories == 2


def test_offline_to_replay_uniform_sampling(tmp_path: Path):
    root = tmp_path / "offline_root"
    root.mkdir()
    offline = _make_simple_offline_root(root)

    # Build replay manager; concat qpos into main observation
    mgr = offline.build_replay_manager(
        scratch_dir=str(tmp_path / "replay_scratch"),
        obs_keys=["qpos"],
        act_key="action",
        concat_obs_to_key="observation",
        include_terminated=True,
        include_truncated=True,
    )

    batch = mgr.buffer.sample()
    assert "observation" in batch
    obs = batch["observation"]
    next_obs = batch[("next", "observation")]
    assert obs.shape == next_obs.shape

    # Termination flags exist and are boolean
    assert "terminated" in batch and batch["terminated"].dtype == torch.bool
    assert "truncated" in batch and batch["truncated"].dtype == torch.bool

    # Additional keys present
    assert "qpos" in batch
    assert ("next", "qpos") in batch.keys(True)


def test_offline_make_sequential_replay(tmp_path: Path):
    root = tmp_path / "offline_root"
    root.mkdir()
    offline = _make_simple_offline_root(root)

    num_envs = 3
    mgr, assignments = offline.make_sequential_replay(
        scratch_dir=str(tmp_path / "replay_scratch"),
        num_envs=num_envs,
        obs_keys=["qpos"],
        act_key="action",
        concat_obs_to_key="observation",
        include_terminated=False,
        include_truncated=False,
    )

    assert len(assignments) == num_envs
    # Each env is mapped to some (task_id, traj_id)
    for a in assignments:
        assert isinstance(a.task_id, int)
        assert isinstance(a.traj_id, int)

    # First sample: one transition per env
    batch1 = mgr.buffer.sample()
    assert batch1.batch_size[0] == num_envs

    # Second sample should advance per-env pointers (sequential sampling semantics)
    batch2 = mgr.buffer.sample()
    assert batch2.batch_size[0] == num_envs

    # Observations must differ across at least one env between calls
    diff = (batch2["observation"] - batch1["observation"]).abs()
    assert torch.any(diff > 0)

    # Per-env window helper: does not advance steps but returns [num_envs, window, ...]
    window = mgr.sample_per_env_window(window_size=3, wrap=True)
    assert window.batch_size == (num_envs, 3)


def test_reference_manager_on_replay(tmp_path: Path):
    root = tmp_path / "offline_root"
    root.mkdir()
    offline = _make_simple_offline_root(root)

    num_envs = 2
    mgr, assignments = offline.make_sequential_replay(
        scratch_dir=str(tmp_path / "replay_scratch"),
        num_envs=num_envs,
        obs_keys=["qpos"],
        act_key="action",
        concat_obs_to_key="observation",
        include_terminated=False,
        include_truncated=False,
    )

    # For this toy dataset, qpos is just a 3D signal; we do not have real joint
    # names here, so use an empty mapping to exercise the plumbing without
    # assuming a particular layout.
    mapping = JointMapping(reference_joint_names=[], target_joint_names=[])
    ref_mgr = ReferenceManager(mgr, joint_mapping=mapping, qpos_key="qpos", qvel_key="qvel")

    ref_td = ref_mgr.get_reference_window(window_size=1, wrap=True)
    # Windowed interface always returns [num_envs, window_size, ...]
    assert ref_td.batch_size == (num_envs, 1)
    assert "root_pos" in ref_td and "joint_pos" in ref_td
    assert ref_td["root_pos"].shape[-1] == 3
    assert ref_td["joint_pos"].shape[-1] == len(mapping.target_joint_names)
