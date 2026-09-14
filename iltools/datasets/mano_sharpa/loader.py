"""Load the video-to-data ``ManoSharpaData`` Parquet format.

The released video-to-data runtime loads one Parquet row into memory and then
samples independent start frames for simulation environments.  This module
keeps those data semantics but converts the row to the validated ILTools
reference contract.  PyArrow and SciPy are optional imports so normal ILTools
users do not need the Sharpa retarget environment.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import xml.etree.ElementTree as ET

import numpy as np

from iltools.core import (
    ContactSequence,
    DexterousReference,
    create_dexterous_reference_manifest,
    save_dexterous_reference_npz,
    sha256_file,
)


def _pyarrow_dataset() -> Any:
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "ManoSharpaLoader requires the optional 'sharpa' dependencies."
        ) from exc
    return ds


def _rotation_imports() -> tuple[Any, Any]:
    try:
        from scipy.spatial.transform import Rotation, Slerp
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("Sharpa interpolation requires SciPy.") from exc
    return Rotation, Slerp


def _validate_rigid_object_asset(path: Path) -> None:
    """Reject articulated URDFs before they reach the Sharpa writer/runtime.

    The source Parquet can contain articulated bodies, but the Sharpa v1
    reference contract deliberately reduces those observations to one rigid
    proxy.  Passing the original articulated URDF as the simulation asset is
    therefore almost always a configuration error and otherwise fails much
    later inside Isaac Sim with an opaque rigid-body-count exception.
    """

    if path.suffix.lower() != ".urdf":
        return
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"Could not parse rigid object URDF {path}: {exc}") from exc
    moving_joints = [
        str(joint.attrib.get("name", "<unnamed>"))
        for joint in root.findall("joint")
        if joint.attrib.get("type", "fixed") != "fixed"
    ]
    if moving_joints:
        raise ValueError(
            "Sharpa v1 requires a rigid object asset; the supplied URDF "
            f"contains articulated joints {moving_joints}. Select a rigid "
            "proxy URDF/USD instead."
        )


def _linear(values: Any, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    flat = array.reshape(array.shape[0], -1)
    out = np.stack(
        [np.interp(dst, src, flat[:, index]) for index in range(flat.shape[1])], axis=1
    )
    return out.reshape((len(dst),) + array.shape[1:]).astype(np.float32)


def _slerp(values: Any, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    Rotation, Slerp = _rotation_imports()
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2:
        return (
            Slerp(src, Rotation.from_quat(array, scalar_first=True))(dst)
            .as_quat(scalar_first=True)
            .astype(np.float32)
        )
    result = np.empty((len(dst), array.shape[1], 4), dtype=np.float32)
    for index in range(array.shape[1]):
        result[:, index] = _slerp(array[:, index], src, dst)
    return result


def _contact_linear(values: Any, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    result = _linear(array, src, dst)
    lower = np.clip(np.searchsorted(src, dst, side="right") - 1, 0, len(src) - 1)
    upper = np.minimum(lower + 1, len(src) - 1)
    valid = (np.abs(array[lower]) > 1.0e-8) & (np.abs(array[upper]) > 1.0e-8)
    return np.where(valid, result, 0.0).astype(np.float32)


def resample_mano_sharpa_row(
    row: Mapping[str, Any], *, target_fps: float, motion_speed: float = 1.0
) -> dict[str, Any]:
    """Resample one row with the interpolation rules from video-to-data."""

    if target_fps <= 0.0 or motion_speed <= 0.0:
        raise ValueError("target_fps and motion_speed must be positive.")
    result = deepcopy(dict(row))
    frame_count = len(result["robot_right_wrist_position"])
    if frame_count < 2:
        raise ValueError("A MANO/Sharpa trajectory needs at least two frames.")
    source_fps = float(result["fps"])
    src = np.arange(frame_count, dtype=np.float64) / source_fps
    duration = src[-1]
    # Match the source implementation: target_num_frames is the effective
    # samples-per-source-second, while the stored data still advances once per
    # policy step at target_fps.
    output_count = max(2, int(duration * target_fps / motion_speed))
    dst = np.linspace(0.0, duration, output_count)

    linear_fields = (
        "object_articulation",
        "object_body_position",
        "robot_right_finger_joints",
        "robot_left_finger_joints",
        "robot_right_wrist_position",
        "robot_left_wrist_position",
    )
    for name in linear_fields:
        if result.get(name) is not None and np.asarray(result[name]).size:
            result[name] = _linear(result[name], src, dst).tolist()
    for name in ("robot_right_wrist_wxyz", "robot_left_wrist_wxyz", "object_body_wxyz"):
        result[name] = _slerp(result[name], src, dst).tolist()
    for name in ("robot_right_frames", "robot_left_frames"):
        frames = np.asarray(result[name])
        result[name] = np.concatenate(
            (_linear(frames[..., :3], src, dst), _slerp(frames[..., 3:7], src, dst)),
            axis=-1,
        ).tolist()
    for side in ("right", "left"):
        for suffix in (
            "link_contact_positions",
            "link_contact_normals",
            "object_contact_positions",
            "object_contact_normals",
        ):
            name = f"mano_{side}_{suffix}"
            if result.get(name):
                result[name] = _contact_linear(result[name], src, dst).tolist()
        name = f"mano_{side}_object_contact_part_ids"
        if result.get(name):
            nearest = np.abs(src[:, None] - dst[None, :]).argmin(axis=0)
            result[name] = np.asarray(result[name])[nearest].astype(np.int32).tolist()
    result["fps"] = float(target_fps)
    result["resample"] = {
        "source_fps": source_fps,
        "source_frames": int(frame_count),
        "target_fps": float(target_fps),
        "motion_speed": float(motion_speed),
        "output_frames": int(output_count),
    }
    return result


def make_rigid_proxy_row(
    row: Mapping[str, Any], *, object_body_index: int = 0
) -> dict[str, Any]:
    """Reduce one source row to a single rigid object body.

    The released source fixtures may contain an articulated object represented
    by multiple rigid bodies.  Sharpa v1 intentionally accepts only one rigid
    object, so this helper keeps one body's metric pose and contact geometry,
    drops articulation values, and maps contact part IDs to the remaining
    rigid object (index zero).  It is an explicit documented proxy operation;
    it does not claim to preserve articulated-object dynamics.
    """

    result = deepcopy(dict(row))
    names = tuple(str(name) for name in result.get("object_body_names", ()))
    if len(names) < 1:
        raise ValueError("A rigid proxy needs at least one source object body.")
    if isinstance(object_body_index, bool) or not isinstance(
        object_body_index, (int, np.integer)
    ):
        raise ValueError("object_body_index must be an integer.")
    index = int(object_body_index)
    if index < 0 or index >= len(names):
        raise ValueError(
            f"object_body_index {index} is outside the {len(names)} source bodies."
        )

    selected_name = names[index]
    for field in ("object_body_position", "object_body_wxyz"):
        values = np.asarray(result.get(field, ()))
        if values.ndim < 3 or values.shape[1] != len(names):
            raise ValueError(
                f"{field} must have shape [frames, {len(names)}, ...] for a proxy."
            )
        result[field] = values[:, index : index + 1].tolist()
    result["object_body_names"] = [selected_name]
    result["object_name"] = selected_name
    # The selected body is now a rigid proxy, regardless of source articulation.
    frame_count = len(result["object_body_position"])
    result["object_articulation"] = [[] for _ in range(frame_count)]
    for side in ("left", "right"):
        part_ids = f"mano_{side}_object_contact_part_ids"
        if result.get(part_ids) is not None:
            values = np.asarray(result[part_ids], dtype=np.int32)
            result[part_ids] = np.zeros_like(values).tolist()
    sequence_id = str(result.get("sequence_id", "trajectory"))
    result["sequence_id"] = (
        sequence_id
        if sequence_id.endswith("_rigid_proxy")
        else f"{sequence_id}_rigid_proxy"
    )
    result["rigid_proxy_source_body"] = selected_name
    result["rigid_proxy_source_body_index"] = index
    result["object_kind"] = "rigid"
    result["physics_backend"] = "physx"
    result["physics_dt"] = 0.01
    result["control_dt"] = 0.05
    return result


RETARGET_PROVENANCE_KEYS = (
    "retarget_solver",
    "retarget_max_iterations",
    "retarget_frequency_hz",
    "retarget_mjcf_sha256",
)


def retarget_provenance(row: Mapping[str, Any]) -> dict[str, Any]:
    """Describe where a row's Sharpa solution came from and how well it converged.

    ``retarget_source`` is ``iltools_pink`` when :class:`SharpaPinkRetargeter`
    solved the row and ``parquet_robot_columns`` when the stored ``robot_*``
    columns are trusted as-is.  ``retarget_solution_kind`` is ``placeholder``
    when every frame reports a zero task error after exactly one iteration,
    which is the signature of a fabricated fixture rather than an IK solve.
    """

    provenance: dict[str, Any] = {
        "retarget_source": str(row.get("retarget_source", "parquet_robot_columns"))
    }
    for key in RETARGET_PROVENANCE_KEYS:
        if key in row:
            provenance[key] = row[key]
    errors: list[np.ndarray] = []
    iterations: list[np.ndarray] = []
    for side in ("left", "right"):
        error = np.asarray(
            row.get(f"robot_{side}_frame_task_errors") or (), dtype=np.float64
        )
        count = np.asarray(
            row.get(f"robot_{side}_num_optimization_iterations") or (),
            dtype=np.float64,
        )
        if error.size:
            provenance[f"ik_{side}_max_task_error_m"] = float(np.max(error))
            provenance[f"ik_{side}_mean_task_error_m"] = float(np.mean(error))
            errors.append(error)
        if count.size:
            provenance[f"ik_{side}_mean_iterations"] = float(np.mean(count))
            provenance[f"ik_{side}_max_iterations"] = int(np.max(count))
            iterations.append(count)
    if errors and iterations:
        placeholder = all(np.all(item == 0.0) for item in errors) and all(
            np.all(item == 1.0) for item in iterations
        )
        provenance["retarget_solution_kind"] = "placeholder" if placeholder else "ik"
    else:
        provenance["retarget_solution_kind"] = "unknown"
    return provenance


def _contacts_from_row(
    row: Mapping[str, Any], frame_count: int
) -> ContactSequence | None:
    names = tuple(str(name) for name in row.get("mano_link_names", ()))
    if not names:
        return None
    blocks: list[
        tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    for side in ("left", "right"):
        link_pos = np.asarray(
            row.get(f"mano_{side}_link_contact_positions", ()), dtype=np.float32
        )
        if link_pos.size == 0:
            continue
        blocks.append(
            (
                side,
                link_pos,
                np.asarray(row[f"mano_{side}_link_contact_normals"], dtype=np.float32),
                np.asarray(
                    row[f"mano_{side}_object_contact_positions"], dtype=np.float32
                ),
                np.asarray(
                    row[f"mano_{side}_object_contact_normals"], dtype=np.float32
                ),
                np.asarray(row[f"mano_{side}_object_contact_part_ids"], dtype=np.int32),
            )
        )
    if not blocks:
        return None
    slot_names: list[tuple[str, ...]] = []
    hand_sides: list[str] = []
    arrays = [[] for _ in range(5)]
    for side, link_pos, link_normal, object_pos, object_normal, part_ids in blocks:
        count = link_pos.shape[1]
        side_names = names[:count]
        slot_names.append(tuple(side_names))
        hand_sides.append(side)
        for target, value in zip(
            arrays,
            (link_pos, link_normal, object_pos, object_normal, part_ids),
            strict=True,
        ):
            target.append(value)
    link_pos, link_normal, object_pos, object_normal, part_ids = [
        np.stack(value, axis=1) for value in arrays
    ]
    active = np.linalg.norm(link_pos, axis=-1) > 1.0e-8
    return ContactSequence(
        hand_sides=tuple(hand_sides),
        link_names=np.asarray(slot_names),
        link_positions_w=link_pos,
        link_normals_w=link_normal,
        object_positions_w=object_pos,
        object_normals_w=object_normal,
        object_indices=part_ids,
        active=active,
    )


@dataclass(slots=True)
class ManoSharpaLoader:
    """Read processed or MANO-only legacy V2D Parquet rows."""

    parquet_path: str | Path

    def rows(self, *, filters: Any = None) -> tuple[dict[str, Any], ...]:
        ds = _pyarrow_dataset()
        dataset = ds.dataset(
            str(Path(self.parquet_path).expanduser()),
            format="parquet",
            partitioning="hive",
        )
        table = dataset.to_table(filter=filters)
        return tuple(dict(row) for row in table.to_pylist())

    @staticmethod
    def to_reference(
        row: Mapping[str, Any],
        *,
        object_asset_path: str | Path,
        object_asset_sha256: str = "",
    ) -> DexterousReference:
        """Convert one processed row to a single-rigid-object reference."""

        required = (
            "robot_left_finger_joints",
            "robot_right_finger_joints",
            "robot_left_wrist_position",
            "robot_right_wrist_position",
            "robot_left_wrist_wxyz",
            "robot_right_wrist_wxyz",
            "robot_left_frames",
            "robot_right_frames",
        )
        missing = [name for name in required if not row.get(name)]
        if missing:
            raise ValueError(
                "The row has no processed Sharpa solution. Retarget it first; "
                f"missing fields: {missing}."
            )
        object_body_names = tuple(
            str(name) for name in row.get("object_body_names", ())
        )
        if len(object_body_names) != 1:
            raise ValueError("Sharpa v1 supports exactly one rigid object body.")
        articulation = np.asarray(row.get("object_articulation", ()), dtype=np.float32)
        if articulation.size and np.any(np.abs(articulation) > 1.0e-8):
            raise ValueError("Sharpa v1 does not support articulated objects.")

        left_names = tuple(str(name) for name in row["left_robot_finger_joint_names"])
        right_names = tuple(str(name) for name in row["right_robot_finger_joint_names"])
        if len(left_names) != 22 or len(right_names) != 22:
            raise ValueError("Sharpa references require 22 finger joints per hand.")
        left_qpos = np.asarray(row["robot_left_finger_joints"], dtype=np.float32)
        right_qpos = np.asarray(row["robot_right_finger_joints"], dtype=np.float32)
        frame_count = left_qpos.shape[0]
        if right_qpos.shape[0] != frame_count:
            raise ValueError("Left and right trajectories must have the same length.")
        object_pos = np.asarray(row["object_body_position"], dtype=np.float32)
        object_quat = np.asarray(row["object_body_wxyz"], dtype=np.float32)
        if object_pos.shape != (frame_count, 1, 3) or object_quat.shape != (
            frame_count,
            1,
            4,
        ):
            raise ValueError("Sharpa v1 needs one object body pose per frame.")

        left_wrist = np.concatenate(
            (
                np.asarray(row["robot_left_wrist_position"]),
                np.asarray(row["robot_left_wrist_wxyz"]),
            ),
            axis=-1,
        )
        right_wrist = np.concatenate(
            (
                np.asarray(row["robot_right_wrist_position"]),
                np.asarray(row["robot_right_wrist_wxyz"]),
            ),
            axis=-1,
        )
        metadata = {
            "source_format": "video_to_data/ManoSharpaData",
            "raw_motion_file": str(row.get("raw_motion_file", "")),
            "mano_to_robot_scale": float(row.get("mano_to_robot_scale") or 1.0),
            "object_kind": "rigid",
        }
        for key in (
            "physics_backend",
            "physics_dt",
            "control_dt",
            "rigid_proxy_source_body",
            "rigid_proxy_source_body_index",
        ):
            if key in row:
                metadata[key] = row[key]
        metadata.update(retarget_provenance(row))
        if row.get("resample") is not None:
            metadata["resample"] = {
                key: (
                    int(value) if isinstance(value, (int, np.integer)) else float(value)
                )
                for key, value in dict(row["resample"]).items()
            }
        return DexterousReference(
            sequence_id=str(row["sequence_id"]),
            robot_name="sharpa_wave",
            robot_layout="dual_floating_hand",
            fps=float(row["fps"]),
            joint_names=left_names + right_names,
            left_joint_names=left_names,
            right_joint_names=right_names,
            qpos=np.concatenate((left_qpos, right_qpos), axis=-1),
            fixed_root_pose_w=np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            left_wrist_pose_w=left_wrist,
            right_wrist_pose_w=right_wrist,
            left_wrist_frame_name=str(
                row.get("left_robot_frame_task_names", ["left_hand_C_MC"])[0]
            ),
            right_wrist_frame_name=str(
                row.get("right_robot_frame_task_names", ["right_hand_C_MC"])[0]
            ),
            left_hand_frame_names=tuple(
                str(name) for name in row["left_robot_frame_names"]
            ),
            left_hand_frame_poses_w=np.asarray(row["robot_left_frames"]),
            right_hand_frame_names=tuple(
                str(name) for name in row["right_robot_frame_names"]
            ),
            right_hand_frame_poses_w=np.asarray(row["robot_right_frames"]),
            object_names=(str(row.get("object_name") or object_body_names[0]),),
            object_poses_w=np.concatenate((object_pos, object_quat), axis=-1),
            object_asset_paths=(str(Path(object_asset_path).expanduser().resolve()),),
            object_asset_sha256=(object_asset_sha256,),
            object_radii=np.asarray(
                row.get("object_mesh_radius", [0.05])[:1], dtype=np.float32
            ),
            contacts=_contacts_from_row(row, frame_count),
            metadata=metadata,
        )

    def references(
        self,
        *,
        object_asset_path: str | Path,
        object_asset_sha256: str = "",
        filters: Any = None,
    ) -> tuple[DexterousReference, ...]:
        return tuple(
            self.to_reference(
                row,
                object_asset_path=object_asset_path,
                object_asset_sha256=object_asset_sha256,
            )
            for row in self.rows(filters=filters)
        )

    def write_dataset(
        self,
        output_dir: str | Path,
        *,
        object_asset_path: str | Path,
        dataset_name: str,
        filters: Any = None,
    ) -> Path:
        """Write NPZ references, a hash-bound manifest, and canonical Zarr data."""

        try:
            import zarr
        except ImportError as exc:  # pragma: no cover - required base dependency
            raise ImportError("Writing a Sharpa dataset requires zarr.") from exc
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        object_path = Path(object_asset_path).expanduser().resolve()
        if not object_path.is_file():
            raise ValueError(f"Rigid object asset does not exist: {object_path}")
        _validate_rigid_object_asset(object_path)
        object_hash = sha256_file(object_path)
        references = self.references(
            object_asset_path=object_path,
            object_asset_sha256=object_hash,
            filters=filters,
        )
        if not references:
            raise ValueError("The Parquet selection contains no trajectories.")
        npz_paths = []
        zarr_root = zarr.open_group(str(output / "trajectories.zarr"), mode="w")
        trajectories = zarr_root.create_group("trajectories")
        for index, reference in enumerate(references):
            safe_name = f"{index:05d}_{reference.sequence_id.replace('/', '_')}"
            npz_path = save_dexterous_reference_npz(
                reference, output / "references" / f"{safe_name}.npz"
            )
            npz_paths.append(npz_path)
            group = trajectories.create_group(safe_name)
            group.attrs.update(
                {
                    "sequence_id": reference.sequence_id,
                    "fps": reference.fps,
                    "robot_layout": reference.robot_layout,
                    "joint_names": list(reference.joint_names),
                }
            )
            arrays = {
                "qpos": reference.qpos,
                "qvel": reference.qvel,
                "left_wrist_pose_w": reference.left_wrist_pose_w,
                "right_wrist_pose_w": reference.right_wrist_pose_w,
                "left_wrist_twist_w": reference.left_wrist_twist_w,
                "right_wrist_twist_w": reference.right_wrist_twist_w,
                "object_poses_w": reference.object_poses_w,
                "object_twists_w": reference.object_twists_w,
                "left_hand_frame_poses_w": reference.left_hand_frame_poses_w,
                "right_hand_frame_poses_w": reference.right_hand_frame_poses_w,
            }
            for name, value in arrays.items():
                group.create_array(name, data=np.asarray(value), overwrite=True)
            if reference.contacts is not None:
                contact_group = group.create_group("contacts")
                for name, value in reference.contacts.to_npz_fields().items():
                    contact_group.create_array(
                        name, data=np.asarray(value), overwrite=True
                    )
        return create_dexterous_reference_manifest(
            npz_paths,
            output / "manifest.json",
            dataset_name=dataset_name,
            metadata={
                "zarr": "trajectories.zarr",
                "source_format": "video_to_data/ManoSharpaData",
                "object_scope": "single_rigid",
            },
        )


__all__ = [
    "ManoSharpaLoader",
    "make_rigid_proxy_row",
    "resample_mano_sharpa_row",
    "retarget_provenance",
]
