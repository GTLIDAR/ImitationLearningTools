from __future__ import annotations

import inspect
import os
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from zarr.storage import LocalStore

from iltools.core.metadata_schema import DatasetMeta
from iltools.datasets.base_loader import BaseLoader

TrajectoryEntry = dict[str, Any]
MotionIndex = dict[str, dict[str, dict[str, Any]]]

BASE_EXPORT_KEYS: frozenset[str] = frozenset(
    [
        "qpos",
        "qvel",
        "root_pos",
        "root_quat",
        "root_lin_vel",
        "root_ang_vel",
        "joint_pos",
        "joint_vel",
        "episode_index",
        "frame_index",
    ]
)
TRANSITION_EXPORT_KEYS: frozenset[str] = frozenset(
    [
        "next_qpos",
        "next_qvel",
        "next_root_pos",
        "next_root_quat",
        "next_root_lin_vel",
        "next_root_ang_vel",
        "next_joint_pos",
        "next_joint_vel",
        "next_episode_index",
        "next_frame_index",
    ]
)
OPTIONAL_ACTION_KEYS: frozenset[str] = frozenset(
    [
        "action",
        "next_action",
        "target_qpos",
        "next_target_qpos",
        "target_joint_pos",
        "next_target_joint_pos",
        "target_root_pos",
        "next_target_root_pos",
        "target_root_quat",
        "next_target_root_quat",
    ]
)
OPTIONAL_TIME_KEYS: frozenset[str] = frozenset(["timestamp", "next_timestamp"])

UNITREE_G1_WBT_DEFAULT_REPO_ID = "unitreerobotics/G1_WBT_Brainco_Pickup_Pillow"
UNITREE_G1_WBT_29DOF_DATASET_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

EPS = 1.0e-8
_MISSING = object()


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_get_nested(cfg: Any, keys: Sequence[str], default: Any = None) -> Any:
    current = cfg
    for key in keys:
        if current is None:
            return default
        if isinstance(current, Mapping):
            current = current.get(key, None)
        else:
            current = getattr(current, key, None)
    return default if current is None else current


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, os.PathLike)):
        return [value]
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _maybe_list_of_str(value: Any) -> list[str] | None:
    if value is None:
        return None
    return [str(v) for v in _as_list(value)]


def _maybe_list_of_int(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(int(v) for v in _as_list(value))


def _normalize_frame_range(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        start = value.get("start")
        end = value.get("end")
    else:
        seq = list(value)
        if len(seq) != 2:
            raise ValueError(
                "frame_range must have exactly two elements: [start, end]."
            )
        start, end = seq
    start_i = int(start)
    end_i = int(end)
    if start_i < 1:
        raise ValueError("frame_range start must be >= 1 (1-indexed inclusive).")
    if end_i < start_i:
        raise ValueError("frame_range end must be >= start.")
    return start_i, end_i


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(name)).strip("_")
    return cleaned or "motion"


def _row_get(row: Mapping[str, Any], key: str, default: Any = _MISSING) -> Any:
    if key in row:
        return row[key]

    current: Any = row
    for part in key.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            if default is _MISSING:
                raise KeyError(key)
            return default
    return current


def _row_get_first(
    row: Mapping[str, Any], keys: Sequence[str], default: Any = _MISSING
) -> tuple[Any, str | None]:
    for key in keys:
        try:
            return _row_get(row, key), key
        except KeyError:
            continue
    if default is _MISSING:
        raise KeyError(f"Missing required row field; tried {list(keys)}.")
    return default, None


def _to_numpy(
    value: Any, *, dtype: np.dtype[Any] | type | None = np.float32
) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu().numpy()
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype)
    return array


def _stack_rows(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    *,
    required: bool,
    dtype: np.dtype[Any] | type | None = np.float32,
) -> tuple[np.ndarray | None, str | None]:
    values: list[np.ndarray] = []
    selected_key: str | None = None
    for row in rows:
        value, key = _row_get_first(row, keys, default=None)
        if key is None:
            if required:
                raise KeyError(f"Missing required row field; tried {list(keys)}.")
            return None, None
        if selected_key is None:
            selected_key = key
        elif selected_key != key:
            raise ValueError(
                "Rows mix multiple keys for the same field: "
                f"first={selected_key!r}, current={key!r}."
            )
        array = _to_numpy(value, dtype=dtype)
        if array.ndim == 0:
            array = array.reshape(1)
        values.append(array)
    if not values:
        if required:
            raise ValueError("Cannot stack an empty episode.")
        return None, None
    return np.stack(values, axis=0), selected_key


def _normalize_quat_wxyz(quat: np.ndarray, quat_order: str) -> np.ndarray:
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternion width 4, got {tuple(quat.shape)}.")
    if quat_order == "wxyz":
        quat_wxyz = quat
    elif quat_order == "xyzw":
        quat_wxyz = quat[..., [3, 0, 1, 2]]
    else:
        raise ValueError(f"Unsupported quat_order={quat_order!r}.")
    norm = np.linalg.norm(quat_wxyz, axis=-1, keepdims=True)
    norm = np.where(norm < EPS, 1.0, norm)
    return (quat_wxyz / norm).astype(np.float32)


def _quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    out = quat.copy()
    out[..., 1:] *= -1.0
    return out


def _quat_mul_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _axis_angle_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    q = _normalize_quat_wxyz(quat, "wxyz")
    w = np.clip(q[..., 0], -1.0, 1.0)
    vector = q[..., 1:]
    vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    axis = np.divide(
        vector,
        np.where(vector_norm < EPS, 1.0, vector_norm),
        out=np.zeros_like(vector),
        where=vector_norm >= EPS,
    )
    angle = 2.0 * np.arctan2(vector_norm[..., 0], w)
    axis_angle = axis * angle[..., None]
    axis_angle[vector_norm[..., 0] < EPS] = 0.0
    return axis_angle.astype(np.float32)


def _so3_derivative_wxyz(rotations: np.ndarray, dt: float) -> np.ndarray:
    q = _normalize_quat_wxyz(rotations, "wxyz")
    n = q.shape[0]
    if n <= 1:
        return np.zeros((n, 3), dtype=np.float32)
    if n == 2:
        q_rel = _quat_mul_wxyz(q[1:2], _quat_conjugate_wxyz(q[0:1]))
        omega = _axis_angle_from_quat_wxyz(q_rel) / float(dt)
        return np.repeat(omega, 2, axis=0).astype(np.float32)
    q_prev = q[:-2]
    q_next = q[2:]
    q_rel = _quat_mul_wxyz(q_next, _quat_conjugate_wxyz(q_prev))
    omega = _axis_angle_from_quat_wxyz(q_rel) / (2.0 * float(dt))
    return np.concatenate([omega[:1], omega, omega[-1:]], axis=0).astype(np.float32)


def _lerp(lhs: np.ndarray, rhs: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return lhs * (1.0 - blend) + rhs * blend


def _quat_slerp_wxyz(q0: np.ndarray, q1: np.ndarray, blend: np.ndarray) -> np.ndarray:
    qa = _normalize_quat_wxyz(q0.astype(np.float32), "wxyz")
    qb = _normalize_quat_wxyz(q1.astype(np.float32), "wxyz")
    t = np.asarray(blend, dtype=np.float32)

    dot = np.sum(qa * qb, axis=-1)
    neg_mask = dot < 0.0
    qb = qb.copy()
    qb[neg_mask] *= -1.0
    dot = np.abs(dot)
    dot = np.clip(dot, -1.0, 1.0)

    out = np.empty_like(qa)
    linear_mask = dot > 0.9995
    if np.any(linear_mask):
        t_linear = t[linear_mask][:, None]
        out[linear_mask] = _normalize_quat_wxyz(
            qa[linear_mask] * (1.0 - t_linear) + qb[linear_mask] * t_linear,
            "wxyz",
        )

    if np.any(~linear_mask):
        theta_0 = np.arccos(dot[~linear_mask])
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t[~linear_mask]
        s0 = np.sin(theta_0 - theta) / np.maximum(sin_theta_0, EPS)
        s1 = np.sin(theta) / np.maximum(sin_theta_0, EPS)
        out[~linear_mask] = (
            qa[~linear_mask] * s0[:, None] + qb[~linear_mask] * s1[:, None]
        )
        out[~linear_mask] = _normalize_quat_wxyz(out[~linear_mask], "wxyz")
    return out.astype(np.float32)


def _filter_kwargs(callable_obj: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return {key: value for key, value in kwargs.items() if value is not None}
    parameters = signature.parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    return {
        key: value
        for key, value in kwargs.items()
        if value is not None and (accepts_kwargs or key in parameters)
    }


@dataclass(frozen=True)
class LeRobotSource:
    repo_id: str
    motion_name: str
    split: str
    root: str | None
    revision: str | None
    episodes: tuple[int, ...] | None
    streaming: bool
    fps: float | None
    max_episodes: int | None
    max_rows: int | None
    source_rows_key: str | None = None


@dataclass(frozen=True)
class TrajectoryInfo:
    dataset: str
    motion: str
    motion_name: str
    trajectory_index: int
    trajectory_in_motion: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start

    def to_dict(self) -> TrajectoryEntry:
        return {
            "dataset": self.dataset,
            "motion": self.motion,
            "motion_name": self.motion_name,
            "trajectory_index": self.trajectory_index,
            "trajectory_in_motion": self.trajectory_in_motion,
            "start": self.start,
            "end": self.end,
            "length": self.length,
        }


class LeRobotLoader(BaseLoader):
    """Load low-dimensional LeRobot episodes into ILTools Zarr trajectories.

    By default this targets the public Unitree G1 WBT LeRobot schema, where
    ``observation.state.robot_q_current`` and ``action.robot_q_desired`` are
    36-wide vectors: root position, root quaternion, then 29 joint positions.
    The state/action keys, joint names, and quaternion order are configurable so
    the loader can also ingest standard LeRobot rows that expose
    ``observation.state`` and ``action``.
    """

    def __init__(
        self,
        cfg: Any,
        build_zarr_dataset: bool = True,
        zarr_path: str | None = None,
        *,
        source: Iterable[Mapping[str, Any]]
        | Mapping[str, Iterable[Mapping[str, Any]]]
        | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset_name = str(
            _cfg_get(
                cfg,
                "dataset_name",
                _cfg_get_nested(cfg, ("dataset", "name"), "lerobot"),
            )
        )
        self.dataset_source = str(_cfg_get(cfg, "source_name", "lerobot"))
        self.default_split = str(
            _cfg_get(cfg, "split", _cfg_get_nested(cfg, ("dataset", "split"), "train"))
        )
        self.default_fps = float(_cfg_get(cfg, "fps", 30.0))
        if self.default_fps <= 0.0:
            raise ValueError("fps must be positive.")
        self.control_freq = self._resolve_control_freq()
        self.frame_range = _normalize_frame_range(_cfg_get(cfg, "frame_range", None))

        self.state_key_candidates = self._resolve_key_candidates(
            primary_key="state_key",
            nested_path=("dataset", "state_key"),
            defaults=(
                "observation.state.robot_q_current",
                "observation.state",
            ),
        )
        self.action_key_candidates = self._resolve_key_candidates(
            primary_key="action_key",
            nested_path=("dataset", "action_key"),
            defaults=(
                "action.robot_q_desired",
                "action",
            ),
            allow_none=True,
        )
        self.episode_key = str(_cfg_get(cfg, "episode_key", "episode_index"))
        self.frame_key = str(_cfg_get(cfg, "frame_key", "frame_index"))
        self.timestamp_key = str(_cfg_get(cfg, "timestamp_key", "timestamp"))
        self.quat_order = str(_cfg_get(cfg, "quat_order", "wxyz"))
        self.align_root_z_to_default = bool(
            _cfg_get(cfg, "align_root_z_to_default", False)
        )
        self.default_root_height = float(_cfg_get(cfg, "default_root_height", 0.0))
        self.drop_short_episodes = bool(_cfg_get(cfg, "drop_short_episodes", True))

        self.default_root = self._optional_path_str(
            _cfg_get(cfg, "root", _cfg_get_nested(cfg, ("dataset", "root"), None))
        )
        self.default_revision = _cfg_get(
            cfg, "revision", _cfg_get_nested(cfg, ("dataset", "revision"), None)
        )
        self.default_episodes = _maybe_list_of_int(
            _cfg_get(
                cfg,
                "episodes",
                _cfg_get_nested(cfg, ("dataset", "episodes"), None),
            )
        )
        self.default_streaming = bool(_cfg_get(cfg, "streaming", False))
        self.force_cache_sync = bool(_cfg_get(cfg, "force_cache_sync", False))
        self.download_videos = bool(_cfg_get(cfg, "download_videos", False))
        self.video_backend = _cfg_get(cfg, "video_backend", None)
        self.max_episodes = self._optional_positive_int(
            _cfg_get(cfg, "max_episodes", None), "max_episodes"
        )
        self.max_rows = self._optional_positive_int(
            _cfg_get(cfg, "max_rows", None), "max_rows"
        )

        self._configured_joint_names = self._read_optional_name_list(
            ("joint_names",), ("dataset", "joint_names")
        )
        self._configured_body_names = self._read_optional_name_list(
            ("body_names",), ("dataset", "body_names")
        )
        self._configured_site_names = self._read_optional_name_list(
            ("site_names",), ("dataset", "site_names")
        )
        self._joint_names: list[str] | None = (
            list(self._configured_joint_names)
            if self._configured_joint_names is not None
            else None
        )
        self._body_names: list[str] | None = (
            list(self._configured_body_names)
            if self._configured_body_names is not None
            else None
        )
        self._site_names: list[str] | None = (
            list(self._configured_site_names)
            if self._configured_site_names is not None
            else None
        )
        self.dataset_joint_names = tuple(
            _maybe_list_of_str(_cfg_get(cfg, "dataset_joint_names", None))
            or UNITREE_G1_WBT_29DOF_DATASET_JOINT_NAMES
        )
        self.target_joint_names = tuple(
            _maybe_list_of_str(_cfg_get(cfg, "target_joint_names", None)) or ()
        )
        self._joint_reorder_index: np.ndarray | None = None
        self._validate_joint_name_config()

        self._source_rows = self._normalize_source_override(source)
        self.sources = self._collect_sources()
        self._available_keys: set[str] = set()
        self._trajectory_output_fps: list[float] = []
        self._source_metadata: dict[str, dict[str, Any]] = {}

        self.logger.info(
            "Initializing LeRobotLoader with %d source(s)", len(self.sources)
        )
        self._trajectory_info_list, self._motion_info_dict = self._get_trajectories(
            build_zarr_dataset=build_zarr_dataset,
            path=zarr_path or kwargs.pop("path", None),
            **kwargs,
        )
        self._metadata = self._discover_metadata()

    def _resolve_key_candidates(
        self,
        *,
        primary_key: str,
        nested_path: Sequence[str],
        defaults: Sequence[str],
        allow_none: bool = False,
    ) -> tuple[str, ...]:
        value = _cfg_get(self.cfg, primary_key, _cfg_get_nested(self.cfg, nested_path))
        if value is None:
            return () if allow_none and defaults == () else tuple(defaults)
        if allow_none and str(value).lower() in {"", "none", "null"}:
            return ()
        return tuple(str(item) for item in _as_list(value))

    def _resolve_control_freq(self) -> float | None:
        control_freq = _cfg_get(self.cfg, "control_freq", None)
        if control_freq is None:
            control_freq = _cfg_get(self.cfg, "output_fps", None)

        if control_freq is None:
            sim_dt = _cfg_get_nested(self.cfg, ("sim", "dt"), None)
            decimation = _cfg_get(self.cfg, "decimation", None)
            if sim_dt is not None and decimation is not None:
                control_freq = 1.0 / (float(sim_dt) * float(decimation))

        if control_freq is None:
            sim_dt = _cfg_get_nested(self.cfg, ("sim", "dt"), None)
            n_substeps = _cfg_get(self.cfg, "n_substeps", None)
            if sim_dt is not None and n_substeps is not None:
                control_freq = 1.0 / (float(sim_dt) * float(n_substeps))

        if control_freq is None:
            return None

        control_freq = float(control_freq)
        if control_freq <= 0.0:
            raise ValueError("control_freq must be positive.")
        return control_freq

    def _optional_path_str(self, value: Any) -> str | None:
        if value is None:
            return None
        return str(Path(str(value)).expanduser())

    def _optional_positive_int(self, value: Any, label: str) -> int | None:
        if value is None:
            return None
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"{label} must be positive when provided.")
        return parsed

    def _read_optional_name_list(self, *paths: Sequence[str]) -> list[str] | None:
        for path in paths:
            value = _cfg_get_nested(self.cfg, path, None)
            parsed = _maybe_list_of_str(value)
            if parsed is not None:
                return parsed
        return None

    def _validate_joint_name_config(self) -> None:
        if len(self.dataset_joint_names) == 0:
            return
        if len(set(self.dataset_joint_names)) != len(self.dataset_joint_names):
            raise ValueError("dataset_joint_names must not contain duplicate names.")
        if not self.target_joint_names:
            return
        if len(set(self.target_joint_names)) != len(self.target_joint_names):
            raise ValueError("target_joint_names must not contain duplicate names.")
        missing = [
            joint_name
            for joint_name in self.target_joint_names
            if joint_name not in self.dataset_joint_names
        ]
        extra = [
            joint_name
            for joint_name in self.dataset_joint_names
            if joint_name not in self.target_joint_names
        ]
        if missing or extra:
            raise ValueError(
                "target_joint_names must contain the same joints as "
                f"dataset_joint_names; missing={missing}, extra={extra}."
            )
        self._joint_reorder_index = np.asarray(
            [
                self.dataset_joint_names.index(joint_name)
                for joint_name in self.target_joint_names
            ],
            dtype=np.int64,
        )

    def _normalize_source_override(
        self,
        source: Iterable[Mapping[str, Any]]
        | Mapping[str, Iterable[Mapping[str, Any]]]
        | None,
    ) -> dict[str, list[Mapping[str, Any]]] | None:
        if source is None:
            return None
        if isinstance(source, Mapping):
            return {
                _sanitize_name(str(name)): list(rows)
                for name, rows in source.items()
            }
        return {"in_memory": list(source)}

    def _collect_sources(self) -> list[LeRobotSource]:
        if self._source_rows is not None:
            return [
                LeRobotSource(
                    repo_id=name,
                    motion_name=name,
                    split=self.default_split,
                    root=None,
                    revision=None,
                    episodes=None,
                    streaming=False,
                    fps=self.default_fps,
                    max_episodes=self.max_episodes,
                    max_rows=self.max_rows,
                    source_rows_key=name,
                )
                for name in self._source_rows
            ]

        entries = self._collect_source_entries()
        sources = [self._resolve_source_entry(entry) for entry in entries]
        if not sources:
            raise ValueError("No LeRobot sources resolved from config.")
        return sources

    def _collect_source_entries(self) -> list[Any]:
        candidates = [
            ("dataset", "trajectories", "lerobot"),
            ("dataset", "lerobot"),
            ("dataset", "repo_ids"),
            ("dataset", "repo_id"),
            ("repo_ids",),
            ("repo_id",),
        ]
        for candidate in candidates:
            value = _cfg_get_nested(self.cfg, candidate, None)
            if value is not None:
                return self._normalize_source_entries(value)
        return [UNITREE_G1_WBT_DEFAULT_REPO_ID]

    def _normalize_source_entries(self, value: Any) -> list[Any]:
        if isinstance(value, Mapping):
            if self._looks_like_source_entry(value):
                return [value]
            return [
                {"name": str(name), "repo_id": repo_id}
                for name, repo_id in value.items()
            ]
        return _as_list(value)

    def _looks_like_source_entry(self, value: Mapping[str, Any]) -> bool:
        entry_keys = {
            "repo_id",
            "id",
            "name",
            "root",
            "split",
            "revision",
            "episodes",
            "streaming",
            "fps",
            "max_episodes",
            "max_rows",
        }
        return any(key in value for key in entry_keys)

    def _resolve_source_entry(self, entry: Any) -> LeRobotSource:
        if isinstance(entry, Mapping):
            repo_id_value = entry.get("repo_id", entry.get("id", None))
            if repo_id_value is None:
                raise ValueError("LeRobot source entry must include 'repo_id'.")
            repo_id = str(repo_id_value)
            motion_name = _sanitize_name(str(entry.get("name", repo_id)))
            split = str(entry.get("split", self.default_split))
            root = self._optional_path_str(entry.get("root", self.default_root))
            revision_value = entry.get("revision", self.default_revision)
            revision = None if revision_value is None else str(revision_value)
            episodes = _maybe_list_of_int(entry.get("episodes", self.default_episodes))
            streaming = bool(entry.get("streaming", self.default_streaming))
            fps_value = entry.get("fps", None)
            fps = None if fps_value is None else float(fps_value)
            max_episodes = self._optional_positive_int(
                entry.get("max_episodes", self.max_episodes),
                "max_episodes",
            )
            max_rows = self._optional_positive_int(
                entry.get("max_rows", self.max_rows),
                "max_rows",
            )
        else:
            repo_id = str(entry)
            motion_name = _sanitize_name(repo_id)
            split = self.default_split
            root = self.default_root
            revision = (
                None
                if self.default_revision is None
                else str(self.default_revision)
            )
            episodes = self.default_episodes
            streaming = self.default_streaming
            fps = None
            max_episodes = self.max_episodes
            max_rows = self.max_rows

        if fps is not None and fps <= 0.0:
            raise ValueError("source fps must be positive.")

        return LeRobotSource(
            repo_id=repo_id,
            motion_name=motion_name,
            split=split,
            root=root,
            revision=revision,
            episodes=episodes,
            streaming=streaming,
            fps=fps,
            max_episodes=max_episodes,
            max_rows=max_rows,
        )

    @property
    def num_traj(self) -> int:
        return len(self._trajectory_info_list)

    @property
    def control_dt(self) -> float | list[float]:
        if not self._trajectory_output_fps:
            fps = self.control_freq or self.default_fps
            return 1.0 / float(fps)
        if len(set(self._trajectory_output_fps)) == 1:
            return 1.0 / float(self._trajectory_output_fps[0])
        return [1.0 / float(fps) for fps in self._trajectory_output_fps]

    @property
    def metadata(self) -> DatasetMeta:
        return self._metadata

    def __len__(self) -> int:
        return self.num_traj

    @property
    def trajectory_info_list(self) -> list[TrajectoryEntry]:
        return list(self._trajectory_info_list)

    @property
    def motion_info_dict(self) -> MotionIndex:
        return dict(self._motion_info_dict)

    def _get_trajectories(
        self,
        build_zarr_dataset: bool = False,
        path: str | None = None,
        **kwargs: Any,
    ) -> tuple[list[TrajectoryEntry], MotionIndex]:
        if build_zarr_dataset and path is None:
            raise ValueError("path must be provided when build_zarr_dataset is True")

        trajectory_info_list: list[TrajectoryEntry] = []
        motion_info_dict: MotionIndex = {}
        global_idx = 0

        dataset_group: zarr.Group | None = None
        if build_zarr_dataset:
            chunk_size = int(kwargs.get("chunk_size", 64))
            shard_size = int(kwargs.get("shard_size", 512))
            overwrite = bool(kwargs.get("overwrite", False))
            os.makedirs(path, exist_ok=True)
            store = LocalStore(path)
            root = zarr.group(store=store, overwrite=overwrite)
            if self.dataset_name in root:
                if not overwrite:
                    raise ValueError(
                        f"Group '{self.dataset_name}' already exists in {path}. "
                        "Use overwrite=True to rebuild."
                    )
                del root[self.dataset_name]
            dataset_group = root.create_group(self.dataset_name)
        else:
            chunk_size = 64
            shard_size = 512

        motion_local_cursors: dict[str, int] = {}
        motion_local_counts: dict[str, int] = {}
        motion_groups: dict[str, zarr.Group] = {}
        motion_metadata: dict[str, dict[str, Any]] = {}

        for source in self.sources:
            motion_name = source.motion_name
            motion_group = motion_groups.get(motion_name)
            if (
                motion_group is None
                and build_zarr_dataset
                and dataset_group is not None
            ):
                motion_group = dataset_group.create_group(motion_name)
                motion_groups[motion_name] = motion_group

            motion_entry = motion_info_dict.setdefault(
                self.dataset_name, {}
            ).setdefault(
                motion_name,
                {
                    "motion_name": motion_name,
                    "repo_ids": [],
                    "splits": [],
                    "trajectory_indices": [],
                    "trajectory_lengths": [],
                    "trajectory_local_start_indices": [],
                    "trajectory_local_end_indices": [],
                    "source_fps": [],
                    "output_fps": [],
                    "episode_indices": [],
                    "state_keys": [],
                    "action_keys": [],
                },
            )
            if source.repo_id not in motion_entry["repo_ids"]:
                motion_entry["repo_ids"].append(source.repo_id)
            if source.split not in motion_entry["splits"]:
                motion_entry["splits"].append(source.split)

            local_start_cursor = motion_local_cursors.get(motion_name, 0)
            local_count = motion_local_counts.get(motion_name, 0)

            source_episode_count = 0
            for episode_id, rows, source_fps in self._iter_source_episodes(source):
                if source.max_episodes is not None and source_episode_count >= int(
                    source.max_episodes
                ):
                    break
                traj_data, output_fps, state_key, action_key = self._load_episode_rows(
                    rows=rows,
                    source_fps=source_fps,
                )
                if traj_data is None:
                    continue
                self._available_keys.update(traj_data.keys())
                self._infer_or_validate_names(traj_data)

                traj_len = int(traj_data["qpos"].shape[0])
                local_start = local_start_cursor
                local_end = local_start + traj_len
                local_start_cursor = local_end

                traj_info = TrajectoryInfo(
                    dataset=self.dataset_name,
                    motion=motion_name,
                    motion_name=motion_name,
                    trajectory_index=global_idx,
                    trajectory_in_motion=local_count,
                    start=local_start,
                    end=local_end,
                )
                trajectory_info_list.append(traj_info.to_dict())

                motion_entry["trajectory_indices"].append(global_idx)
                motion_entry["trajectory_lengths"].append(traj_len)
                motion_entry["trajectory_local_start_indices"].append(local_start)
                motion_entry["trajectory_local_end_indices"].append(local_end)
                motion_entry["source_fps"].append(float(source_fps))
                motion_entry["output_fps"].append(float(output_fps))
                motion_entry["episode_indices"].append(int(episode_id))
                motion_entry["state_keys"].append(state_key)
                motion_entry["action_keys"].append(action_key)
                self._trajectory_output_fps.append(float(output_fps))

                if motion_group is not None:
                    traj_group = motion_group.create_group(f"trajectory_{local_count}")
                    self._save_trajectory_data(
                        traj_group,
                        traj_data,
                        chunk_size=chunk_size,
                        shard_size=shard_size,
                    )

                source_episode_count += 1
                local_count += 1
                global_idx += 1

            motion_local_cursors[motion_name] = local_start_cursor
            motion_local_counts[motion_name] = local_count

            if motion_group is not None:
                motion_group.attrs["num_trajectories"] = local_count
                motion_group.attrs["trajectory_lengths"] = motion_entry[
                    "trajectory_lengths"
                ]
                motion_group.attrs["repo_ids"] = motion_entry["repo_ids"]
                motion_group.attrs["splits"] = motion_entry["splits"]
                motion_group.attrs["episode_indices"] = motion_entry["episode_indices"]
                motion_group.attrs["source_fps"] = motion_entry["source_fps"]
                motion_group.attrs["output_fps"] = motion_entry["output_fps"]

            motion_metadata[motion_name] = {
                "repo_ids": motion_entry["repo_ids"],
                "splits": motion_entry["splits"],
                "num_trajectories": len(motion_entry["trajectory_indices"]),
                "trajectory_lengths": motion_entry["trajectory_lengths"],
                "episode_indices": motion_entry["episode_indices"],
            }

        if not trajectory_info_list:
            raise ValueError("No LeRobot trajectories were loaded.")

        if build_zarr_dataset and dataset_group is not None:
            dataset_group.attrs["num_trajectories"] = len(trajectory_info_list)
            dataset_group.attrs["trajectory_lengths"] = [
                e["length"] for e in trajectory_info_list
            ]
            dataset_group.attrs["keys"] = sorted(self._available_keys)
            dataset_group.attrs["joint_names"] = self._joint_names or []
            dataset_group.attrs["body_names"] = self._body_names or []
            dataset_group.attrs["site_names"] = self._site_names or []
            dataset_group.attrs["dt"] = self.control_dt
            dataset_group.attrs["control_freq"] = self._metadata_control_freq()
            dataset_group.attrs["transition_format"] = "flat_next_keys_v1"
            dataset_group.attrs["transition_keys"] = sorted(
                key for key in self._available_keys if key.startswith("next_")
            )
            dataset_group.attrs["motion_metadata"] = motion_metadata
            dataset_group.attrs["source_metadata"] = self._source_metadata
            dataset_group.attrs["trajectory_info_list"] = trajectory_info_list
            dataset_group.attrs["motion_info_dict"] = motion_info_dict
            self.logger.info("Saved trajectories to Zarr store at %s", path)

        self.logger.info(
            "Built LeRobot trajectory manifest with %d entries across %d motions",
            len(trajectory_info_list),
            sum(len(motions) for motions in motion_info_dict.values()),
        )
        return trajectory_info_list, motion_info_dict

    def _iter_source_episodes(
        self, source: LeRobotSource
    ) -> Iterator[tuple[int, list[Mapping[str, Any]], float]]:
        iterable, source_fps = self._make_source_iterable(source)
        source_fps = float(
            source.fps or source_fps or self.control_freq or self.default_fps
        )
        if source_fps <= 0.0:
            raise ValueError(f"Invalid LeRobot source fps: {source_fps}.")

        current_episode_id: int | None = None
        current_rows: list[Mapping[str, Any]] = []
        rows_seen = 0
        yielded = 0

        for row in iterable:
            rows_seen += 1
            if source.max_rows is not None and rows_seen > int(source.max_rows):
                break
            episode_id = int(
                _to_numpy(_row_get(row, self.episode_key), dtype=None).reshape(-1)[0]
            )
            if source.episodes is not None and episode_id not in source.episodes:
                if current_rows and current_episode_id is not None:
                    yield (
                        current_episode_id,
                        self._finalize_episode_rows(current_rows),
                        source_fps,
                    )
                    yielded += 1
                    current_rows = []
                    current_episode_id = None
                continue
            if current_episode_id is None:
                current_episode_id = episode_id
            if episode_id != current_episode_id:
                yield (
                    current_episode_id,
                    self._finalize_episode_rows(current_rows),
                    source_fps,
                )
                yielded += 1
                if source.max_episodes is not None and yielded >= int(
                    source.max_episodes
                ):
                    return
                current_rows = []
                current_episode_id = episode_id
            current_rows.append(row)

        if current_rows and current_episode_id is not None:
            yield (
                current_episode_id,
                self._finalize_episode_rows(current_rows),
                source_fps,
            )

    def _make_source_iterable(
        self, source: LeRobotSource
    ) -> tuple[Iterable[Mapping[str, Any]], float | None]:
        if self._source_rows is not None:
            if source.source_rows_key is None:
                raise RuntimeError(
                    "Internal source_rows_key missing for in-memory source."
                )
            return self._source_rows[source.source_rows_key], source.fps

        if source.streaming:
            return self._make_streaming_lerobot_iterable(source), source.fps
        return self._make_lerobot_dataset_iterable(source)

    def _make_lerobot_dataset_iterable(
        self, source: LeRobotSource
    ) -> tuple[Iterable[Mapping[str, Any]], float | None]:
        try:
            from lerobot.datasets import LeRobotDataset
        except ImportError as exc:
            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset
            except ImportError:
                raise ImportError(
                    "LeRobotLoader requires the optional 'lerobot' package. "
                    "Install iltools[lerobot] or install lerobot directly."
                ) from exc

        kwargs = _filter_kwargs(
            LeRobotDataset,
            {
                "root": source.root,
                "episodes": list(source.episodes)
                if source.episodes is not None
                else None,
                "revision": source.revision,
                "force_cache_sync": self.force_cache_sync,
                "download_videos": self.download_videos,
                "video_backend": self.video_backend,
                "split": source.split,
            },
        )
        dataset = LeRobotDataset(source.repo_id, **kwargs)
        self._source_metadata[source.motion_name] = self._extract_lerobot_metadata(
            dataset
        )
        fps = self._read_dataset_fps(dataset)

        def _iter_dataset_rows() -> Iterator[Mapping[str, Any]]:
            for index in range(len(dataset)):
                yield dataset[index]

        return _iter_dataset_rows(), fps

    def _make_streaming_lerobot_iterable(
        self, source: LeRobotSource
    ) -> Iterable[Mapping[str, Any]]:
        try:
            from lerobot.datasets import StreamingLeRobotDataset
        except ImportError as exc:
            try:
                from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
            except ImportError:
                raise ImportError(
                    "LeRobotLoader streaming mode requires "
                    "lerobot.datasets.StreamingLeRobotDataset."
                ) from exc
        kwargs = _filter_kwargs(
            StreamingLeRobotDataset,
            {
                "split": source.split,
                "revision": source.revision,
            },
        )
        return StreamingLeRobotDataset(source.repo_id, **kwargs)

    def _read_dataset_fps(self, dataset: Any) -> float | None:
        fps = getattr(dataset, "fps", None)
        if fps is not None:
            return float(fps)
        meta = getattr(dataset, "meta", None)
        fps = getattr(meta, "fps", None)
        if fps is not None:
            return float(fps)
        if isinstance(meta, Mapping) and meta.get("fps") is not None:
            return float(meta["fps"])
        return None

    def _extract_lerobot_metadata(self, dataset: Any) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for attr in ("fps", "num_frames", "num_episodes", "features"):
            value = getattr(dataset, attr, None)
            if value is not None:
                metadata[attr] = self._json_safe(value)
        meta = getattr(dataset, "meta", None)
        if meta is not None:
            for attr in ("repo_id", "total_frames", "total_episodes"):
                value = getattr(meta, attr, None)
                if value is not None:
                    metadata[attr] = self._json_safe(value)
        return metadata

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, Mapping):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [self._json_safe(v) for v in value]
        if hasattr(value, "tolist"):
            return self._json_safe(value.tolist())
        return str(value)

    def _finalize_episode_rows(
        self, rows: list[Mapping[str, Any]]
    ) -> list[Mapping[str, Any]]:
        if not rows:
            return rows
        rows = list(rows)
        try:
            rows.sort(
                key=lambda row: int(
                    _to_numpy(_row_get(row, self.frame_key), dtype=None).reshape(-1)[0]
                )
            )
        except KeyError:
            pass
        if self.frame_range is None:
            return rows
        start, end = self.frame_range
        if end > len(rows):
            raise ValueError(
                f"frame_range {self.frame_range} exceeds episode length {len(rows)}."
            )
        return rows[start - 1 : end]

    def _load_episode_rows(
        self,
        *,
        rows: Sequence[Mapping[str, Any]],
        source_fps: float,
    ) -> tuple[dict[str, np.ndarray] | None, float, str, str | None]:
        if len(rows) < 2:
            if self.drop_short_episodes:
                self.logger.debug("Skipping LeRobot episode with fewer than 2 rows.")
                return None, source_fps, "", None
            raise ValueError("A LeRobot episode must contain at least two rows.")

        state, state_key = _stack_rows(
            rows,
            self.state_key_candidates,
            required=True,
            dtype=np.float32,
        )
        if state is None or state_key is None:
            raise KeyError("Missing required LeRobot state field.")
        if state.ndim != 2 or state.shape[-1] < 8:
            raise ValueError(
                f"{state_key} must have shape [T, >=8], got {tuple(state.shape)}."
            )

        action: np.ndarray | None
        action_key: str | None
        if self.action_key_candidates:
            action, action_key = _stack_rows(
                rows,
                self.action_key_candidates,
                required=False,
                dtype=np.float32,
            )
        else:
            action, action_key = None, None

        episode_index, _ = _stack_rows(
            rows,
            (self.episode_key,),
            required=True,
            dtype=np.int64,
        )
        frame_index, _ = _stack_rows(
            rows,
            (self.frame_key,),
            required=False,
            dtype=np.int64,
        )
        timestamp, _ = _stack_rows(
            rows,
            (self.timestamp_key,),
            required=False,
            dtype=np.float32,
        )

        root_pos = state[:, :3].astype(np.float32)
        root_quat = _normalize_quat_wxyz(
            state[:, 3:7].astype(np.float32), self.quat_order
        )
        if self.align_root_z_to_default:
            root_pos = root_pos.copy()
            root_pos[:, 2] += float(self.default_root_height) - float(root_pos[0, 2])
        joint_pos = self._reorder_joints(state[:, 7:].astype(np.float32))

        target_root_pos = None
        target_root_quat = None
        target_joint_pos = None
        if action is not None:
            if action.ndim == 1:
                action = action[:, None]
            if action.ndim != 2:
                raise ValueError(
                    f"{action_key} must have shape [T, A], got {tuple(action.shape)}."
                )
            if action.shape[0] != state.shape[0]:
                raise ValueError(
                    f"{action_key} length {action.shape[0]} does not match "
                    f"{state_key} length {state.shape[0]}."
                )
            if action.shape[-1] == state.shape[-1]:
                target_root_pos = action[:, :3].astype(np.float32)
                target_root_quat = _normalize_quat_wxyz(
                    action[:, 3:7].astype(np.float32), self.quat_order
                )
                target_joint_pos = self._reorder_joints(
                    action[:, 7:].astype(np.float32)
                )
            elif action.shape[-1] == joint_pos.shape[-1]:
                target_joint_pos = action.astype(np.float32)

        output_fps = float(self.control_freq or source_fps)
        if output_fps <= 0.0:
            raise ValueError("output_fps must be positive.")
        (
            root_pos,
            root_quat,
            joint_pos,
            action,
            target_root_pos,
            target_root_quat,
            target_joint_pos,
            episode_index,
            frame_index,
            timestamp,
        ) = self._maybe_resample_episode(
            source_fps=source_fps,
            output_fps=output_fps,
            root_pos=root_pos,
            root_quat=root_quat,
            joint_pos=joint_pos,
            action=action,
            target_root_pos=target_root_pos,
            target_root_quat=target_root_quat,
            target_joint_pos=target_joint_pos,
            episode_index=episode_index,
            frame_index=frame_index,
            timestamp=timestamp,
        )

        root_lin_vel, root_ang_vel, joint_vel = self._compute_velocities(
            root_pos=root_pos,
            root_quat=root_quat,
            joint_pos=joint_pos,
            dt=1.0 / output_fps,
        )
        traj_data = self._build_trajectory_dict(
            root_pos=root_pos,
            root_quat=root_quat,
            joint_pos=joint_pos,
            root_lin_vel=root_lin_vel,
            root_ang_vel=root_ang_vel,
            joint_vel=joint_vel,
            episode_index=episode_index,
            frame_index=frame_index,
            timestamp=timestamp,
            action=action,
            target_root_pos=target_root_pos,
            target_root_quat=target_root_quat,
            target_joint_pos=target_joint_pos,
        )
        return traj_data, output_fps, state_key, action_key

    def _reorder_joints(self, joint_data: np.ndarray) -> np.ndarray:
        if self._joint_reorder_index is None:
            return joint_data.astype(np.float32)
        if joint_data.shape[-1] != len(self.dataset_joint_names):
            raise ValueError(
                "Cannot apply target_joint_names because joint width "
                f"{joint_data.shape[-1]} does not match dataset_joint_names length "
                f"{len(self.dataset_joint_names)}."
            )
        return joint_data[:, self._joint_reorder_index].astype(np.float32)

    def _maybe_resample_episode(
        self,
        *,
        source_fps: float,
        output_fps: float,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
        joint_pos: np.ndarray,
        action: np.ndarray | None,
        target_root_pos: np.ndarray | None,
        target_root_quat: np.ndarray | None,
        target_joint_pos: np.ndarray | None,
        episode_index: np.ndarray,
        frame_index: np.ndarray | None,
        timestamp: np.ndarray | None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        if root_pos.shape[0] == 0:
            raise ValueError("Cannot load empty LeRobot episode.")
        if root_pos.shape[0] == 1 or np.isclose(source_fps, output_fps):
            return (
                root_pos.astype(np.float32),
                root_quat.astype(np.float32),
                joint_pos.astype(np.float32),
                None if action is None else action.astype(np.float32),
                None
                if target_root_pos is None
                else target_root_pos.astype(np.float32),
                None
                if target_root_quat is None
                else target_root_quat.astype(np.float32),
                None
                if target_joint_pos is None
                else target_joint_pos.astype(np.float32),
                episode_index.astype(np.int64),
                None if frame_index is None else frame_index.astype(np.int64),
                None if timestamp is None else timestamp.astype(np.float32),
            )

        input_dt = 1.0 / float(source_fps)
        output_dt = 1.0 / float(output_fps)
        duration = (root_pos.shape[0] - 1) * input_dt
        if duration <= 0.0:
            raise ValueError("Cannot resample a zero-duration LeRobot episode.")

        times = np.arange(0.0, duration, output_dt, dtype=np.float64)
        if times.size < 2:
            times = np.array([0.0, duration], dtype=np.float64)
        phase = times / duration
        index_0 = np.floor(phase * (root_pos.shape[0] - 1)).astype(np.int64)
        index_1 = np.minimum(index_0 + 1, root_pos.shape[0] - 1)
        blend = (phase * (root_pos.shape[0] - 1) - index_0).astype(np.float32)
        blend_col = blend[:, None]

        def maybe_lerp(value: np.ndarray | None) -> np.ndarray | None:
            if value is None:
                return None
            return _lerp(value[index_0], value[index_1], blend_col).astype(
                np.float32
            )

        def maybe_slerp(value: np.ndarray | None) -> np.ndarray | None:
            if value is None:
                return None
            return _quat_slerp_wxyz(value[index_0], value[index_1], blend)

        out_episode = np.full(
            (times.shape[0], 1), int(episode_index[0, 0]), dtype=np.int64
        )
        out_frame = None
        if frame_index is not None:
            out_frame = np.rint(
                _lerp(frame_index[index_0], frame_index[index_1], blend_col)
            ).astype(np.int64)
        out_timestamp = None
        if timestamp is not None:
            out_timestamp = _lerp(
                timestamp[index_0], timestamp[index_1], blend_col
            ).astype(np.float32)

        return (
            _lerp(root_pos[index_0], root_pos[index_1], blend_col).astype(
                np.float32
            ),
            _quat_slerp_wxyz(root_quat[index_0], root_quat[index_1], blend),
            _lerp(joint_pos[index_0], joint_pos[index_1], blend_col).astype(
                np.float32
            ),
            maybe_lerp(action),
            maybe_lerp(target_root_pos),
            maybe_slerp(target_root_quat),
            maybe_lerp(target_joint_pos),
            out_episode,
            out_frame,
            out_timestamp,
        )

    def _compute_velocities(
        self,
        *,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
        joint_pos: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if dt <= 0.0:
            raise ValueError("dt must be positive when computing velocities.")
        if root_pos.shape[0] <= 1:
            zeros_root = np.zeros_like(root_pos, dtype=np.float32)
            zeros_joint = np.zeros_like(joint_pos, dtype=np.float32)
            zeros_ang = np.zeros((root_pos.shape[0], 3), dtype=np.float32)
            return zeros_root, zeros_ang, zeros_joint
        root_lin_vel = np.gradient(root_pos, dt, axis=0).astype(np.float32)
        joint_vel = np.gradient(joint_pos, dt, axis=0).astype(np.float32)
        root_ang_vel = _so3_derivative_wxyz(root_quat, dt).astype(np.float32)
        return root_lin_vel, root_ang_vel, joint_vel

    def _build_trajectory_dict(
        self,
        *,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
        joint_pos: np.ndarray,
        root_lin_vel: np.ndarray,
        root_ang_vel: np.ndarray,
        joint_vel: np.ndarray,
        episode_index: np.ndarray,
        frame_index: np.ndarray | None,
        timestamp: np.ndarray | None,
        action: np.ndarray | None,
        target_root_pos: np.ndarray | None,
        target_root_quat: np.ndarray | None,
        target_joint_pos: np.ndarray | None,
    ) -> dict[str, np.ndarray]:
        qpos = np.concatenate([root_pos, root_quat, joint_pos], axis=-1).astype(
            np.float32
        )
        qvel = np.concatenate([root_lin_vel, root_ang_vel, joint_vel], axis=-1).astype(
            np.float32
        )
        if qpos.shape[0] < 2:
            raise ValueError("A LeRobot trajectory must contain at least two frames.")

        traj_data: dict[str, np.ndarray] = {
            "qpos": qpos,
            "qvel": qvel,
            "root_pos": root_pos.astype(np.float32),
            "root_quat": root_quat.astype(np.float32),
            "root_lin_vel": root_lin_vel.astype(np.float32),
            "root_ang_vel": root_ang_vel.astype(np.float32),
            "joint_pos": joint_pos.astype(np.float32),
            "joint_vel": joint_vel.astype(np.float32),
            "episode_index": episode_index.astype(np.int64).reshape(
                qpos.shape[0], -1
            ),
            "next_qpos": qpos[1:].astype(np.float32),
            "next_qvel": qvel[1:].astype(np.float32),
            "next_root_pos": root_pos[1:].astype(np.float32),
            "next_root_quat": root_quat[1:].astype(np.float32),
            "next_root_lin_vel": root_lin_vel[1:].astype(np.float32),
            "next_root_ang_vel": root_ang_vel[1:].astype(np.float32),
            "next_joint_pos": joint_pos[1:].astype(np.float32),
            "next_joint_vel": joint_vel[1:].astype(np.float32),
            "next_episode_index": episode_index[1:]
            .astype(np.int64)
            .reshape(qpos.shape[0] - 1, -1),
        }
        if frame_index is not None:
            frame_index = frame_index.astype(np.int64).reshape(qpos.shape[0], -1)
            traj_data["frame_index"] = frame_index
            traj_data["next_frame_index"] = frame_index[1:]
        if timestamp is not None:
            timestamp = timestamp.astype(np.float32).reshape(qpos.shape[0], -1)
            traj_data["timestamp"] = timestamp
            traj_data["next_timestamp"] = timestamp[1:]
        if action is not None:
            action = action.astype(np.float32)
            traj_data["action"] = action
            traj_data["next_action"] = action[1:]
        if target_root_pos is not None:
            traj_data["target_root_pos"] = target_root_pos.astype(np.float32)
            traj_data["next_target_root_pos"] = target_root_pos[1:].astype(np.float32)
        if target_root_quat is not None:
            traj_data["target_root_quat"] = target_root_quat.astype(np.float32)
            traj_data["next_target_root_quat"] = target_root_quat[1:].astype(np.float32)
        if target_joint_pos is not None:
            traj_data["target_joint_pos"] = target_joint_pos.astype(np.float32)
            traj_data["next_target_joint_pos"] = target_joint_pos[1:].astype(np.float32)
        if (
            target_root_pos is not None
            and target_root_quat is not None
            and target_joint_pos is not None
        ):
            target_qpos = np.concatenate(
                [target_root_pos, target_root_quat, target_joint_pos], axis=-1
            ).astype(np.float32)
            traj_data["target_qpos"] = target_qpos
            traj_data["next_target_qpos"] = target_qpos[1:]
        return traj_data

    def _infer_or_validate_names(self, traj_data: Mapping[str, np.ndarray]) -> None:
        joint_count = int(traj_data["joint_pos"].shape[-1])
        if self._joint_names is None:
            if joint_count == len(UNITREE_G1_WBT_29DOF_DATASET_JOINT_NAMES):
                if self.target_joint_names:
                    self._joint_names = list(self.target_joint_names)
                else:
                    self._joint_names = list(UNITREE_G1_WBT_29DOF_DATASET_JOINT_NAMES)
            else:
                self._joint_names = [f"joint_{index}" for index in range(joint_count)]
        elif len(self._joint_names) != joint_count:
            raise ValueError(
                f"joint_names length mismatch: expected {joint_count}, "
                f"got {len(self._joint_names)}."
            )
        if self._body_names is None:
            self._body_names = []
        if self._site_names is None:
            self._site_names = []

    def _save_trajectory_data(
        self,
        traj_group: zarr.Group,
        traj_data: Mapping[str, np.ndarray],
        *,
        chunk_size: int,
        shard_size: int,
    ) -> None:
        for key, value in traj_data.items():
            array = np.asarray(value)
            if array.ndim == 0 or array.shape[0] == 0:
                continue
            chunks = [min(chunk_size, array.shape[0])] + list(array.shape[1:])
            shards = [min(shard_size, array.shape[0])] + list(array.shape[1:])
            ds = traj_group.create_array(
                key,
                shape=array.shape,
                dtype=array.dtype,
                chunks=chunks,
                shards=shards,
            )
            ds[:] = array

    def _discover_metadata(self) -> DatasetMeta:
        trajectory_lengths = [int(e["length"]) for e in self._trajectory_info_list]
        return DatasetMeta(
            name=self.dataset_name,
            source=self.dataset_source,
            version="1.0.0",
            citation=(
                "LeRobot datasets loaded through the optional lerobot package and "
                "converted to ILTools qpos/qvel trajectories."
            ),
            num_trajectories=len(self._trajectory_info_list),
            keys=sorted(self._available_keys),
            trajectory_lengths=trajectory_lengths,
            dt=self.control_dt,
            joint_names=self._joint_names or [],
            body_names=self._body_names or [],
            site_names=self._site_names or [],
            metadata={
                "trajectory_info_list": self._trajectory_info_list,
                "motion_info_dict": self._motion_info_dict,
                "source_metadata": self._source_metadata,
                "state_key_candidates": list(self.state_key_candidates),
                "action_key_candidates": list(self.action_key_candidates),
                "control_freq": self._metadata_control_freq(),
                "sources": [
                    {
                        "repo_id": source.repo_id,
                        "motion_name": source.motion_name,
                        "split": source.split,
                        "root": source.root,
                        "revision": source.revision,
                        "episodes": source.episodes,
                        "streaming": source.streaming,
                        "fps": source.fps,
                    }
                    for source in self.sources
                ],
            },
        )

    def _metadata_control_freq(self) -> float | list[float]:
        if not self._trajectory_output_fps:
            return float(self.control_freq or self.default_fps)
        if len(set(self._trajectory_output_fps)) == 1:
            return float(self._trajectory_output_fps[0])
        return [float(fps) for fps in self._trajectory_output_fps]
