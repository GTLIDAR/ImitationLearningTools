"""Read Ego-Exo4D EgoPose 3-D keypoint annotations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np

from iltools.core.trajectory import Trajectory


def load_ego_pose_trajectory(
    annotation_path: str | Path,
    *,
    keypoint_names: Sequence[str],
    fps: float,
    strict: bool = True,
    resample_gaps: bool = False,
) -> Trajectory:
    """Load one EgoPose JSON file into a named world-frame trajectory.

    Ego-Exo4D stores one annotation list per frame number.  This function uses
    the first annotation in each list, keeps frame numbers explicit, and reads
    the 3-D coordinates only.  With ``strict=False``, missing landmarks are
    represented by NaNs for downstream filtering; strict mode fails early.
    """

    path = Path(annotation_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"EgoPose annotation file not found: {path}")
    if fps <= 0.0:
        raise ValueError("fps must be positive.")
    names = tuple(keypoint_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("keypoint_names must be non-empty and unique.")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"EgoPose root must be an object: {path}")

    frame_numbers: list[int] = []
    frames: list[np.ndarray] = []
    payload_items = cast(dict[str, Any], payload).items()
    for frame_key, annotation_rows in sorted(
        payload_items, key=lambda item: int(item[0])
    ):
        if not isinstance(annotation_rows, list) or not annotation_rows:
            if strict:
                raise ValueError(f"Frame {frame_key} has no EgoPose annotation.")
            continue
        annotation = annotation_rows[0]
        points_value = (
            annotation.get("annotation3D") if isinstance(annotation, dict) else None
        )
        points = (
            cast(dict[str, Any], points_value)
            if isinstance(points_value, dict)
            else None
        )
        if not isinstance(points, dict):
            if strict:
                raise ValueError(f"Frame {frame_key} has no annotation3D object.")
            continue
        frame = np.full((len(names), 3), np.nan, dtype=np.float64)
        missing: list[str] = []
        for index, name in enumerate(names):
            point_value = points.get(name)
            if not isinstance(point_value, dict) or not all(
                axis in point_value for axis in ("x", "y", "z")
            ):
                missing.append(name)
                continue
            point = cast(dict[str, Any], point_value)
            frame[index] = [float(point["x"]), float(point["y"]), float(point["z"])]
        if missing and strict:
            raise ValueError(
                f"Frame {frame_key} is missing EgoPose landmarks: {missing}."
            )
        frame_numbers.append(int(frame_key))
        frames.append(frame)

    if not frames:
        raise ValueError(f"No usable EgoPose frames found in {path}.")
    if strict and len(frame_numbers) > 1:
        frame_delta = np.diff(np.asarray(frame_numbers, dtype=np.int64))
        if not np.all(frame_delta == 1) and not resample_gaps:
            raise ValueError(
                "EgoPose frame numbers are not contiguous; resample the "
                f"annotation before retargeting: {path}"
            )
    trajectory = Trajectory(
        observations={
            "keypoints": np.stack(frames, axis=0),
            "frame_number": np.asarray(frame_numbers, dtype=np.int64),
        },
        infos={
            "source": "Ego-Exo4D EgoPose",
            "coordinate_frame": "world",
            "keypoint_names": list(names),
            "annotation_path": str(path),
        },
        dt=1.0 / float(fps),
    )
    return resample_ego_pose_trajectory(trajectory) if resample_gaps else trajectory


def resample_ego_pose_trajectory(trajectory: Trajectory) -> Trajectory:
    """Linearly fill valid frame gaps onto the native integer frame grid."""

    if "keypoints" not in trajectory.observations:
        raise KeyError("EgoPose trajectory must contain keypoints.")
    if "frame_number" not in trajectory.observations:
        raise KeyError("EgoPose trajectory must contain frame_number.")
    points = np.asarray(trajectory.observations["keypoints"], dtype=np.float64)
    frame_numbers = np.asarray(trajectory.observations["frame_number"], dtype=np.int64)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("EgoPose keypoints must have shape [frames, keypoints, 3].")
    if frame_numbers.shape != (len(points),):
        raise ValueError("EgoPose frame_number must match the frame axis.")
    if len(frame_numbers) == 0 or np.any(np.diff(frame_numbers) <= 0):
        raise ValueError("EgoPose frame_number must be strictly increasing.")
    if not np.isfinite(points).all():
        raise ValueError("EgoPose resampling requires finite valid keypoints.")
    target_frames = np.arange(frame_numbers[0], frame_numbers[-1] + 1, dtype=np.int64)
    if len(target_frames) == len(frame_numbers):
        return trajectory

    resampled = np.empty((len(target_frames), points.shape[1], 3), dtype=np.float64)
    for keypoint_index in range(points.shape[1]):
        for axis in range(3):
            resampled[:, keypoint_index, axis] = np.interp(
                target_frames,
                frame_numbers,
                points[:, keypoint_index, axis],
            )
    infos = dict(trajectory.infos or {})
    infos["resampled_frame_gaps"] = True
    return Trajectory(
        observations={
            "keypoints": resampled,
            "frame_number": target_frames,
        },
        infos=infos,
        dt=trajectory.dt,
    )


def merge_ego_pose_trajectories(trajectories: Sequence[Trajectory]) -> Trajectory:
    """Merge body and hand EgoPose files on their common frame numbers."""

    values = tuple(trajectories)
    if not values:
        raise ValueError("At least one EgoPose trajectory is required.")
    first = values[0]
    if first.dt is None or first.dt <= 0.0:
        raise ValueError("EgoPose trajectories must have a positive dt.")
    frame_sets = []
    for trajectory in values:
        if trajectory.dt is None or not np.isclose(trajectory.dt, first.dt):
            raise ValueError("EgoPose trajectories must use the same FPS.")
        if "frame_number" not in trajectory.observations:
            raise ValueError("EgoPose trajectory is missing frame_number.")
        frame_sets.append(
            set(np.asarray(trajectory.observations["frame_number"]).tolist())
        )
    common_frames = sorted(set.intersection(*frame_sets))
    if not common_frames:
        raise ValueError("EgoPose trajectories have no common frame numbers.")
    if len(common_frames) > 1:
        frame_delta = np.diff(np.asarray(common_frames, dtype=np.int64))
        if not np.all(frame_delta == 1):
            raise ValueError(
                "Merged EgoPose frame numbers are not contiguous; resample the "
                "annotations before retargeting."
            )

    names: list[str] = []
    for trajectory in values:
        trajectory_names = tuple((trajectory.infos or {}).get("keypoint_names", ()))
        if not trajectory_names:
            raise ValueError("EgoPose trajectory is missing keypoint_names metadata.")
        for name in trajectory_names:
            if name not in names:
                names.append(name)

    merged = np.full((len(common_frames), len(names), 3), np.nan, dtype=np.float64)
    name_index = {name: index for index, name in enumerate(names)}
    for trajectory in values:
        source_names = tuple((trajectory.infos or {})["keypoint_names"])
        source_frames = np.asarray(trajectory.observations["frame_number"])
        source_points = np.asarray(trajectory.observations["keypoints"])
        frame_index = {int(frame): index for index, frame in enumerate(source_frames)}
        for merged_frame_index, frame_number in enumerate(common_frames):
            source_frame = source_points[frame_index[frame_number]]
            for source_index, name in enumerate(source_names):
                target_index = name_index[name]
                existing = merged[merged_frame_index, target_index]
                point = source_frame[source_index]
                if np.isfinite(existing).all() and not np.allclose(existing, point):
                    raise ValueError(
                        f"Conflicting EgoPose values for keypoint {name!r} at "
                        f"frame {frame_number}."
                    )
                merged[merged_frame_index, target_index] = point

    if not np.isfinite(merged).all():
        raise ValueError("Merged EgoPose trajectory contains missing keypoints.")
    return Trajectory(
        observations={
            "keypoints": merged,
            "frame_number": np.asarray(common_frames, dtype=np.int64),
        },
        infos={
            "source": "Ego-Exo4D EgoPose body+hand",
            "coordinate_frame": "world",
            "keypoint_names": names,
            "annotation_paths": [
                (trajectory.infos or {}).get("annotation_path") for trajectory in values
            ],
        },
        dt=first.dt,
    )


def load_ego_pose_bundle(
    annotation_paths: Sequence[str | Path],
    *,
    keypoint_names_by_file: Sequence[Sequence[str]],
    fps: float,
    strict: bool = True,
    resample_gaps: bool = False,
) -> Trajectory:
    """Load and merge separate EgoPose body and hand annotation files."""

    paths = tuple(annotation_paths)
    groups = tuple(tuple(names) for names in keypoint_names_by_file)
    if len(paths) != len(groups):
        raise ValueError("annotation_paths and keypoint_names_by_file must align.")
    if not paths:
        raise ValueError("At least one EgoPose annotation path is required.")
    trajectories = tuple(
        load_ego_pose_trajectory(
            path,
            keypoint_names=names,
            fps=fps,
            strict=strict,
            resample_gaps=resample_gaps,
        )
        for path, names in zip(paths, groups, strict=True)
    )
    return merge_ego_pose_trajectories(trajectories)


__all__ = [
    "load_ego_pose_bundle",
    "load_ego_pose_trajectory",
    "merge_ego_pose_trajectories",
    "resample_ego_pose_trajectory",
]
