from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from zarr.storage import LocalStore

from iltools.datasets.lafan1.loader import Lafan1CsvLoader


def _write_motion_csv(path: Path, *, frames: int = 12, joints: int = 4) -> None:
    t = np.arange(frames, dtype=np.float32)
    root_pos = np.stack([0.1 * t, 0.05 * t, np.ones_like(t)], axis=1)
    root_quat_xyzw = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (frames, 1))
    joint_pos = np.stack(
        [np.sin(0.15 * t + float(j)) for j in range(joints)],
        axis=1,
    ).astype(np.float32)
    motion = np.concatenate([root_pos, root_quat_xyzw, joint_pos], axis=1)
    np.savetxt(path, motion, delimiter=",")


def _write_commands_style_npz(path: Path, *, frames: int = 10, joints: int = 6, fps: float = 50.0) -> None:
    t = np.arange(frames, dtype=np.float32)
    joint_pos = np.stack(
        [0.2 * np.sin(0.1 * t + i) for i in range(joints)],
        axis=1,
    ).astype(np.float32)
    joint_vel = np.gradient(joint_pos, 1.0 / fps, axis=0).astype(np.float32)

    body_pos_w = np.zeros((frames, 1, 3), dtype=np.float32)
    body_pos_w[:, 0, 0] = 0.05 * t
    body_pos_w[:, 0, 2] = 0.9
    body_quat_w = np.zeros((frames, 1, 4), dtype=np.float32)
    body_quat_w[:, 0, 0] = 1.0
    body_lin_vel_w = np.gradient(body_pos_w, 1.0 / fps, axis=0).astype(np.float32)
    body_ang_vel_w = np.zeros((frames, 1, 3), dtype=np.float32)

    np.savez(
        path,
        fps=np.array([fps], dtype=np.float32),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )


def test_lafan1_csv_loader_builds_manifest_and_metadata(tmp_path: Path) -> None:
    csv_a = tmp_path / "walk_a.csv"
    csv_b = tmp_path / "walk_b.csv"
    _write_motion_csv(csv_a, frames=15, joints=5)
    _write_motion_csv(csv_b, frames=13, joints=5)

    cfg = {
        "dataset": {"trajectories": {"lafan1_csv": [str(csv_a), str(csv_b)]}},
        "input_fps": 60,
        "control_freq": 30,
    }
    loader = Lafan1CsvLoader(cfg=cfg, build_zarr_dataset=False)

    assert len(loader) == 2
    assert loader.metadata.name == "lafan1"
    assert loader.metadata.source == "lafan1_csv"
    assert loader.metadata.dt == 1.0 / 30.0
    assert loader.metadata.num_trajectories == 2
    assert "qpos" in loader.metadata.keys
    assert "qvel" in loader.metadata.keys
    assert len(loader.metadata.joint_names) == 5


def test_lafan1_csv_loader_writes_zarr(tmp_path: Path) -> None:
    csv_a = tmp_path / "dance_a.csv"
    csv_b = tmp_path / "dance_b.csv"
    _write_motion_csv(csv_a, frames=21, joints=3)
    _write_motion_csv(csv_b, frames=19, joints=3)

    zarr_path = tmp_path / "lafan1_csv.zarr"
    cfg = {
        "dataset": {"trajectories": {"lafan1_csv": [str(csv_a), str(csv_b)]}},
        "input_fps": 60,
        "control_freq": 50,
    }
    loader = Lafan1CsvLoader(cfg=cfg, build_zarr_dataset=True, zarr_path=str(zarr_path))

    store = LocalStore(str(zarr_path))
    root = zarr.group(store=store, overwrite=False)
    assert "lafan1" in root

    dataset_group = root["lafan1"]
    assert dataset_group.attrs["num_trajectories"] == 2
    assert np.isclose(dataset_group.attrs["dt"], 1.0 / 50.0)
    assert "walk_a" not in dataset_group  # names come from file stems for this test
    assert "dance_a" in dataset_group
    assert "dance_b" in dataset_group

    traj_group = dataset_group["dance_a"]["trajectory_0"]
    assert "qpos" in traj_group
    assert "qvel" in traj_group
    qpos = np.asarray(traj_group["qpos"][:])
    qvel = np.asarray(traj_group["qvel"][:])
    assert qpos.ndim == 2
    assert qvel.ndim == 2
    assert qpos.shape[0] == qvel.shape[0]
    assert qpos.shape[1] == 10  # 7 root + 3 joints
    assert len(loader.trajectory_info_list) == 2


def test_lafan1_commands_npz_input(tmp_path: Path) -> None:
    npz_file = tmp_path / "commands_style.npz"
    _write_commands_style_npz(npz_file, frames=11, joints=4, fps=50.0)

    zarr_path = tmp_path / "lafan1_npz.zarr"
    cfg = {
        "dataset": {"trajectories": {"lafan1_csv": [str(npz_file)]}},
        "control_freq": 50,
    }
    loader = Lafan1CsvLoader(cfg=cfg, build_zarr_dataset=True, zarr_path=str(zarr_path))

    store = LocalStore(str(zarr_path))
    root = zarr.group(store=store, overwrite=False)
    traj_group = root["lafan1"]["commands_style"]["trajectory_0"]
    qpos = np.asarray(traj_group["qpos"][:])
    root_pos = np.asarray(traj_group["root_pos"][:])
    root_quat = np.asarray(traj_group["root_quat"][:])
    joint_pos = np.asarray(traj_group["joint_pos"][:])
    body_pos = np.asarray(traj_group["body_pos_w"][:])

    assert qpos.shape[1] == 11  # 7 root + 4 joints
    np.testing.assert_allclose(qpos[:, :3], root_pos, atol=1e-6)
    np.testing.assert_allclose(qpos[:, 3:7], root_quat, atol=1e-6)
    np.testing.assert_allclose(qpos[:, 7:], joint_pos, atol=1e-6)
    np.testing.assert_allclose(root_pos, body_pos[:, 0], atol=1e-6)
    assert loader.metadata.num_trajectories == 1
    assert "body_pos_w" in loader.metadata.keys


def test_lafan1_csv_loader_honors_frame_range(tmp_path: Path) -> None:
    csv_file = tmp_path / "slice_test.csv"
    _write_motion_csv(csv_file, frames=14, joints=2)

    # frame_range is 1-indexed and inclusive => [3, 10] has 8 frames.
    cfg = {
        "dataset": {"trajectories": {"lafan1_csv": [str(csv_file)]}},
        "input_fps": 60,
        "control_freq": 60,
        "frame_range": [3, 10],
    }
    loader = Lafan1CsvLoader(cfg=cfg, build_zarr_dataset=False)
    assert loader.metadata.trajectory_lengths == [8]


def test_lafan1_csv_loader_groups_multiple_files_under_one_motion(tmp_path: Path) -> None:
    dance_a = tmp_path / "dance_a.csv"
    dance_b = tmp_path / "dance_b.csv"
    walk_a = tmp_path / "walk_a.csv"
    _write_motion_csv(dance_a, frames=10, joints=3)
    _write_motion_csv(dance_b, frames=12, joints=3)
    _write_motion_csv(walk_a, frames=11, joints=3)

    zarr_path = tmp_path / "lafan1_grouped.zarr"
    cfg = {
        "dataset": {
            "trajectories": {
                "lafan1_csv": [
                    {"name": "dance_combo", "paths": [str(dance_a), str(dance_b)], "input_fps": 60},
                    {"name": "walk_combo", "path": str(walk_a), "input_fps": 60},
                ]
            }
        },
        "control_freq": 60,
    }

    loader = Lafan1CsvLoader(cfg=cfg, build_zarr_dataset=True, zarr_path=str(zarr_path))
    assert len(loader) == 3
    assert loader.metadata.num_trajectories == 3

    motion_info = loader.motion_info_dict["lafan1"]
    assert "dance_combo" in motion_info
    assert "walk_combo" in motion_info
    assert motion_info["dance_combo"]["trajectory_indices"] == [0, 1]
    assert motion_info["walk_combo"]["trajectory_indices"] == [2]
    assert motion_info["dance_combo"]["trajectory_local_start_indices"] == [0, 10]
    assert motion_info["dance_combo"]["trajectory_local_end_indices"] == [10, 22]

    traj_info = loader.trajectory_info_list
    assert traj_info[0]["motion"] == "dance_combo"
    assert traj_info[0]["trajectory_in_motion"] == 0
    assert traj_info[1]["motion"] == "dance_combo"
    assert traj_info[1]["trajectory_in_motion"] == 1
    assert traj_info[2]["motion"] == "walk_combo"
    assert traj_info[2]["trajectory_in_motion"] == 0

    store = LocalStore(str(zarr_path))
    root = zarr.group(store=store, overwrite=False)
    dataset_group = root["lafan1"]
    assert "dance_combo" in dataset_group
    assert "walk_combo" in dataset_group

    dance_group = dataset_group["dance_combo"]
    assert dance_group.attrs["num_trajectories"] == 2
    assert "trajectory_0" in dance_group
    assert "trajectory_1" in dance_group
