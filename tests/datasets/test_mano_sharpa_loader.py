"""Tests for the video-to-data MANO/Sharpa adapter."""

from __future__ import annotations

import numpy as np
import pytest

from iltools.core import load_dexterous_reference_manifest
from iltools.datasets.mano_sharpa import (
    ManoSharpaLoader,
    make_rigid_proxy_row,
    resample_mano_sharpa_row,
    retarget_provenance,
)


def _pose_frames(frame_count: int) -> list:
    value = np.zeros((frame_count, 67, 7), dtype=np.float32)
    value[..., 3] = 1.0
    return value.tolist()


def _row(*, articulated: bool = False) -> dict:
    frame_count = 2
    left_names = [f"left_joint_{index}" for index in range(22)]
    right_names = [f"right_joint_{index}" for index in range(22)]
    frame_names = [f"frame_{index}" for index in range(67)]
    identity = [[1.0, 0.0, 0.0, 0.0]] * frame_count
    return {
        "sequence_id": "fixture",
        "raw_motion_file": "fixture.npz",
        "robot_name": "sharpa_wave",
        "fps": 20.0,
        "mano_to_robot_scale": 1.0,
        "mano_link_names": [],
        "left_robot_finger_joint_names": left_names,
        "right_robot_finger_joint_names": right_names,
        "left_robot_frame_names": frame_names,
        "right_robot_frame_names": frame_names,
        "left_robot_frame_task_names": ["left_hand_C_MC"],
        "right_robot_frame_task_names": ["right_hand_C_MC"],
        "robot_left_finger_joints": np.zeros((frame_count, 22)).tolist(),
        "robot_right_finger_joints": np.zeros((frame_count, 22)).tolist(),
        "robot_left_wrist_position": [[0.0, -0.2, 0.5], [0.1, -0.2, 0.5]],
        "robot_right_wrist_position": [[0.0, 0.2, 0.5], [0.1, 0.2, 0.5]],
        "robot_left_wrist_wxyz": identity,
        "robot_right_wrist_wxyz": identity,
        "robot_left_frames": _pose_frames(frame_count),
        "robot_right_frames": _pose_frames(frame_count),
        "object_name": "box",
        "object_body_names": ["box"],
        "object_body_position": [[[0.0, 0.0, 0.4]], [[0.1, 0.0, 0.4]]],
        "object_body_wxyz": [[identity[0]], [identity[1]]],
        "object_articulation": [[0.1], [0.1]] if articulated else [[], []],
        "object_mesh_radius": [0.05],
    }


def test_processed_row_builds_dual_floating_hand_reference(tmp_path) -> None:
    object_path = tmp_path / "box.obj"
    object_path.write_text("o box\n", encoding="utf-8")
    reference = ManoSharpaLoader.to_reference(_row(), object_asset_path=object_path)
    assert reference.robot_layout == "dual_floating_hand"
    assert reference.qpos.shape == (2, 44)
    assert reference.left_wrist_twist_w.shape == (2, 6)
    assert (
        reference.joint_names
        == reference.left_joint_names + reference.right_joint_names
    )


def test_articulated_row_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="articulated"):
        ManoSharpaLoader.to_reference(
            _row(articulated=True), object_asset_path=tmp_path / "box.obj"
        )


def test_rigid_proxy_selects_one_body_and_clears_articulation() -> None:
    row = _row()
    row["object_body_names"] = ["bottom", "lid"]
    row["object_body_position"] = [
        [[0.0, 0.0, 0.4], [0.0, 0.0, 0.5]],
        [[0.1, 0.0, 0.4], [0.1, 0.0, 0.5]],
    ]
    row["object_body_wxyz"] = [
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
    ]
    row["object_articulation"] = [[0.2], [0.3]]
    row["mano_left_object_contact_part_ids"] = [[1] * 16, [1] * 16]
    result = make_rigid_proxy_row(row, object_body_index=1)
    assert result["object_body_names"] == ["lid"]
    assert np.asarray(result["object_body_position"]).shape == (2, 1, 3)
    assert result["object_articulation"] == [[], []]
    assert np.asarray(result["mano_left_object_contact_part_ids"]).max() == 0
    assert result["rigid_proxy_source_body_index"] == 1
    assert result["physics_backend"] == "physx"


def test_rigid_proxy_rejects_invalid_body_index() -> None:
    with pytest.raises(ValueError, match="outside"):
        make_rigid_proxy_row(_row(), object_body_index=1)


def test_write_dataset_rejects_articulated_urdf(tmp_path, monkeypatch) -> None:
    pytest.importorskip("zarr")
    object_path = tmp_path / "articulated.urdf"
    object_path.write_text(
        "<robot name='box'><link name='base'/><link name='lid'/>"
        "<joint name='hinge' type='revolute'><parent link='base'/>"
        "<child link='lid'/></joint></robot>",
        encoding="utf-8",
    )
    loader = ManoSharpaLoader(tmp_path / "unused.parquet")
    monkeypatch.setattr(ManoSharpaLoader, "rows", lambda self, filters=None: (_row(),))
    with pytest.raises(ValueError, match="articulated joints"):
        loader.write_dataset(
            tmp_path / "output",
            object_asset_path=object_path,
            dataset_name="fixture",
        )


def test_source_resampling_advances_immediately() -> None:
    pytest.importorskip("scipy")
    row = _row()
    result = resample_mano_sharpa_row(row, target_fps=40.0)
    positions = np.asarray(result["robot_left_wrist_position"])
    assert len(positions) == 2
    assert positions[1, 0] > positions[0, 0]


def test_write_dataset_produces_manifest_and_zarr(tmp_path, monkeypatch) -> None:
    pytest.importorskip("zarr")
    object_path = tmp_path / "box.obj"
    object_path.write_text("o box\n", encoding="utf-8")
    loader = ManoSharpaLoader(tmp_path / "unused.parquet")
    monkeypatch.setattr(ManoSharpaLoader, "rows", lambda self, filters=None: (_row(),))
    manifest_path = loader.write_dataset(
        tmp_path / "output",
        object_asset_path=object_path,
        dataset_name="fixture",
    )
    manifest = load_dexterous_reference_manifest(manifest_path)
    references = manifest.load_references()
    assert references[0].robot_layout == "dual_floating_hand"
    assert (tmp_path / "output" / "trajectories.zarr").is_dir()


def _ik_row(*, placeholder: bool) -> dict:
    row = _row()
    frame_count = 2
    if placeholder:
        errors = np.zeros((frame_count, 11)).tolist()
        iterations = [1, 1]
    else:
        errors = (np.full((frame_count, 11), 0.002) + [[0.0], [0.001]]).tolist()
        iterations = [12, 7]
    for side in ("left", "right"):
        row[f"robot_{side}_frame_task_errors"] = errors
        row[f"robot_{side}_num_optimization_iterations"] = iterations
    return row


def test_provenance_flags_fabricated_placeholder_solution() -> None:
    provenance = retarget_provenance(_ik_row(placeholder=True))
    assert provenance["retarget_source"] == "parquet_robot_columns"
    assert provenance["retarget_solution_kind"] == "placeholder"


def test_provenance_records_ik_statistics_and_source() -> None:
    row = _ik_row(placeholder=False)
    row["retarget_source"] = "iltools_pink"
    row["retarget_max_iterations"] = 100
    row["retarget_mjcf_sha256"] = {"left": "a", "right": "b"}
    provenance = retarget_provenance(row)
    assert provenance["retarget_source"] == "iltools_pink"
    assert provenance["retarget_solution_kind"] == "ik"
    assert provenance["retarget_max_iterations"] == 100
    assert provenance["retarget_mjcf_sha256"] == {"left": "a", "right": "b"}
    assert provenance["ik_left_max_task_error_m"] == pytest.approx(0.003)
    assert provenance["ik_left_max_iterations"] == 12
    assert provenance["ik_right_mean_iterations"] == pytest.approx(9.5)


def test_provenance_without_ik_columns_is_unknown() -> None:
    assert retarget_provenance(_row())["retarget_solution_kind"] == "unknown"


def test_reference_metadata_carries_retarget_provenance(tmp_path) -> None:
    object_path = tmp_path / "box.obj"
    object_path.write_text("o box\n", encoding="utf-8")
    reference = ManoSharpaLoader.to_reference(
        _ik_row(placeholder=True), object_asset_path=object_path
    )
    assert reference.metadata["retarget_solution_kind"] == "placeholder"
    assert reference.metadata["retarget_source"] == "parquet_robot_columns"
