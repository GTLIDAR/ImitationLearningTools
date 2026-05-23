from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import zarr
from zarr.storage import LocalStore

from iltools.datasets.lerobot.loader import LeRobotLoader
from iltools.datasets.utils import make_rb_from


def _make_lerobot_rows(
    *,
    episode_index: int,
    frames: int,
    joints: int = 29,
) -> list[dict[str, object]]:
    t = np.arange(frames, dtype=np.float32)
    root_pos = np.stack([0.05 * t, -0.02 * t, np.ones_like(t)], axis=1)
    root_quat = np.tile(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (frames, 1)
    )
    joint_pos = np.stack(
        [0.1 * np.sin(0.2 * t + float(index)) for index in range(joints)],
        axis=1,
    ).astype(np.float32)
    q_current = np.concatenate([root_pos, root_quat, joint_pos], axis=1)
    q_desired = q_current.copy()
    q_desired[:, 7:] += 0.05

    return [
        {
            "episode_index": episode_index,
            "frame_index": frame,
            "timestamp": float(frame) / 30.0,
            "observation.state.robot_q_current": q_current[frame],
            "action.robot_q_desired": q_desired[frame],
        }
        for frame in range(frames)
    ]


def test_lerobot_loader_builds_manifest_and_zarr_from_rows(tmp_path: Path) -> None:
    rows = _make_lerobot_rows(episode_index=0, frames=6) + _make_lerobot_rows(
        episode_index=1, frames=5
    )
    zarr_path = tmp_path / "lerobot.zarr"
    cfg = {
        "dataset_name": "unitree_lerobot",
        "control_freq": 30,
    }

    loader = LeRobotLoader(
        cfg=cfg,
        source=rows,
        build_zarr_dataset=True,
        zarr_path=str(zarr_path),
    )

    assert len(loader) == 2
    assert loader.metadata.name == "unitree_lerobot"
    assert loader.metadata.dt == 1.0 / 30.0
    assert loader.motion_info_dict["unitree_lerobot"]["in_memory"][
        "episode_indices"
    ] == [0, 1]

    store = LocalStore(str(zarr_path))
    root = zarr.group(store=store, overwrite=False)
    dataset_group = root["unitree_lerobot"]
    assert dataset_group.attrs["transition_format"] == "flat_next_keys_v1"
    assert dataset_group.attrs["num_trajectories"] == 2

    traj_group = dataset_group["in_memory"]["trajectory_0"]
    qpos = np.asarray(traj_group["qpos"][:])
    qvel = np.asarray(traj_group["qvel"][:])
    next_qpos = np.asarray(traj_group["next_qpos"][:])
    action = np.asarray(traj_group["action"][:])
    target_joint_pos = np.asarray(traj_group["target_joint_pos"][:])

    assert qpos.shape == (6, 36)
    assert qvel.shape == (6, 35)
    assert next_qpos.shape == (5, 36)
    assert action.shape == (6, 36)
    assert target_joint_pos.shape == (6, 29)
    np.testing.assert_allclose(next_qpos, qpos[1:], atol=1e-6)
    np.testing.assert_allclose(target_joint_pos, action[:, 7:], atol=1e-6)


def test_lerobot_loader_make_rb_alignment(tmp_path: Path) -> None:
    rows = _make_lerobot_rows(episode_index=4, frames=7)
    zarr_path = tmp_path / "lerobot_rb.zarr"
    _ = LeRobotLoader(
        cfg={"control_freq": 30},
        source=rows,
        build_zarr_dataset=True,
        zarr_path=str(zarr_path),
    )

    rb, info = make_rb_from(zarr_path=zarr_path, device="cpu", verbose_tree=False)
    assert info["written"] == 6

    store = LocalStore(str(zarr_path))
    root = zarr.group(store=store, overwrite=False)
    traj_group = root["lerobot"]["in_memory"]["trajectory_0"]
    qpos = np.asarray(traj_group["qpos"][:])
    next_qpos = np.asarray(traj_group["next_qpos"][:])

    first = rb[0]
    np.testing.assert_allclose(first["qpos"].cpu().numpy(), qpos[0], atol=1e-6)
    np.testing.assert_allclose(
        first["next_qpos"].cpu().numpy(), next_qpos[0], atol=1e-6
    )
    np.testing.assert_allclose(first["next_qpos"].cpu().numpy(), qpos[1], atol=1e-6)


def test_lerobot_loader_uses_lerobot_dataset_package(monkeypatch) -> None:
    rows = _make_lerobot_rows(episode_index=2, frames=4)
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeLeRobotDataset:
        fps = 30
        features = {"observation.state.robot_q_current": {"shape": (36,)}}

        def __init__(self, repo_id: str, **kwargs: object) -> None:
            calls.append((repo_id, kwargs))

        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int) -> dict[str, object]:
            return rows[index]

    lerobot_module = types.ModuleType("lerobot")
    datasets_module = types.ModuleType("lerobot.datasets")
    datasets_module.LeRobotDataset = FakeLeRobotDataset
    monkeypatch.setitem(sys.modules, "lerobot", lerobot_module)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets_module)

    loader = LeRobotLoader(
        cfg={
            "repo_id": "fake/repo",
            "root": "/tmp/fake_lerobot_root",
            "episodes": [2],
        },
        build_zarr_dataset=False,
    )

    assert len(loader) == 1
    assert calls[0][0] == "fake/repo"
    assert calls[0][1]["root"] == "/tmp/fake_lerobot_root"
    assert loader.metadata.metadata["source_metadata"]["fake_repo"]["fps"] == 30
