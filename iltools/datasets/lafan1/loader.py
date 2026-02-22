from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    ]
)
OPTIONAL_BODY_KEYS: frozenset[str] = frozenset(
    ["body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]
)
EPS = 1.0e-8


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read key from a mapping/object-like config."""
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_get_nested(cfg: Any, keys: Sequence[str], default: Any = None) -> Any:
    """Read nested keys from a mapping/object-like config."""
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
    """Convert value to list while treating strings/paths as scalars."""
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
    """Normalize optional list-like string config values."""
    if value is None:
        return None
    return [str(v) for v in _as_list(value)]


def _normalize_frame_range(value: Any) -> tuple[int, int] | None:
    """Normalize frame range to 1-indexed inclusive bounds."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        start = value.get("start")
        end = value.get("end")
    else:
        seq = list(value)
        if len(seq) != 2:
            raise ValueError("frame_range must have exactly two elements: [start, end].")
        start, end = seq
    start_i = int(start)
    end_i = int(end)
    if start_i < 1:
        raise ValueError("frame_range start must be >= 1 (1-indexed inclusive).")
    if end_i < start_i:
        raise ValueError("frame_range end must be >= start.")
    return start_i, end_i


def _sanitize_motion_name(name: str) -> str:
    """Make a stable, Zarr-safe-ish motion name."""
    cleaned = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(name)).strip("_")
    return cleaned or "motion"


@dataclass(frozen=True)
class MotionSource:
    """Resolved source entry for one motion trajectory."""

    motion_name: str
    path: Path
    input_fps: float
    frame_range: tuple[int, int] | None
    explicit_motion_name: bool = False


@dataclass(frozen=True)
class TrajectoryInfo:
    """Immutable container for trajectory slice information."""

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


class Lafan1CsvLoader(BaseLoader):
    """Load LAFAN1-style motions from prepared CSV/NPZ files.

    The CSV format follows the preprocessing assumptions used by
    ``unitree_rl_lab/scripts/mimic/csv_to_npz.py``:
    - columns [0:3]: root position (world frame),
    - columns [3:7]: root orientation quaternion in XYZW,
    - columns [7:]: joint positions.

    The loader resamples motions to ``control_freq`` and computes velocities
    so output can be consumed by imitation replay utilities expecting
    ``qpos``/``qvel``.
    """

    def __init__(
        self,
        cfg: Any,
        build_zarr_dataset: bool = True,
        zarr_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        self.dataset_name = str(
            _cfg_get(
                cfg,
                "dataset_name",
                _cfg_get_nested(cfg, ("dataset", "name"), "lafan1"),
            )
        )
        self.dataset_source = str(_cfg_get(cfg, "source_name", "lafan1_csv"))

        self.default_input_fps = float(_cfg_get(cfg, "input_fps", 60.0))
        if self.default_input_fps <= 0.0:
            raise ValueError("input_fps must be positive.")

        self.control_freq = self._resolve_control_freq()
        self.csv_root = self._resolve_csv_root()
        self.default_frame_range = _normalize_frame_range(
            _cfg_get(cfg, "frame_range", None)
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

        source_entries = self._collect_source_entries()
        self.motion_sources = self._resolve_motion_sources(source_entries)
        self._available_keys: set[str] = set(BASE_EXPORT_KEYS)
        num_motion_groups = len({s.motion_name for s in self.motion_sources})
        self.logger.info(
            "Initializing Lafan1CsvLoader with %d source trajectories across %d motion groups",
            len(self.motion_sources),
            num_motion_groups,
        )

        self._trajectory_info_list, self._motion_info_dict = self._get_trajectories(
            build_zarr_dataset=build_zarr_dataset,
            path=zarr_path or kwargs.pop("path", None),
            **kwargs,
        )
        self._metadata = self._discover_metadata()

    def _read_optional_name_list(self, *paths: Sequence[str]) -> list[str] | None:
        for path in paths:
            value = _cfg_get_nested(self.cfg, path, None)
            parsed = _maybe_list_of_str(value)
            if parsed is not None:
                return parsed
        return None

    def _resolve_csv_root(self) -> Path | None:
        root = _cfg_get(self.cfg, "csv_root", None)
        if root is None:
            root = _cfg_get_nested(self.cfg, ("dataset", "csv_root"), None)
        if root is None:
            return None
        return Path(str(root)).expanduser().resolve()

    def _resolve_control_freq(self) -> float:
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
            control_freq = 50.0

        control_freq = float(control_freq)
        if control_freq <= 0.0:
            raise ValueError("control_freq must be positive.")
        return control_freq

    def _collect_source_entries(self) -> list[Any]:
        """Collect source list from commonly used config locations."""
        candidates = [
            ("dataset", "trajectories", "lafan1_csv"),
            ("dataset", "trajectories", "lafan1"),
            ("dataset", "csv_files"),
            ("csv_files",),
            ("data_path",),
            ("input_path",),
        ]

        for candidate in candidates:
            value = _cfg_get_nested(self.cfg, candidate, None)
            if value is not None:
                return self._normalize_source_entries(value)

        raise ValueError(
            "Could not find LAFAN1 CSV sources in config. "
            "Expected one of: dataset.trajectories.lafan1_csv, "
            "dataset.trajectories.lafan1, dataset.csv_files, csv_files, data_path."
        )

    def _normalize_source_entries(self, value: Any) -> list[Any]:
        """Normalize source config into a list of entry specs.

        Supported forms:
        - list[str | dict], e.g. `["a.csv", {"name": "walk", "paths": [...]}]`
        - dict[name -> path_or_paths], e.g. `{"walk": ["a.csv", "b.csv"]}`
        - scalar path, e.g. `"/tmp/motion.csv"`
        """
        if isinstance(value, Mapping):
            if self._looks_like_motion_entry(value):
                return [value]
            # Mapping style: motion_name -> path(s)
            return [{"name": str(name), "path": path_spec} for name, path_spec in value.items()]
        return _as_list(value)

    def _looks_like_motion_entry(self, value: Mapping[str, Any]) -> bool:
        entry_keys = {
            "name",
            "path",
            "paths",
            "file",
            "files",
            "csv_path",
            "csv_files",
            "input_fps",
            "frame_range",
        }
        return any(k in value for k in entry_keys)

    def _resolve_motion_sources(self, entries: Sequence[Any]) -> list[MotionSource]:
        sources: list[MotionSource] = []
        for entry in entries:
            sources.extend(self._resolve_entry(entry))

        if not sources:
            raise ValueError("No motion files resolved from provided LAFAN1 sources.")

        # Keep explicitly named groups together. For implicit names, de-conflict.
        normalized_sources: list[MotionSource] = []
        implicit_name_counts: dict[str, int] = {}
        for source in sources:
            base_name = _sanitize_motion_name(source.motion_name)
            if source.explicit_motion_name:
                normalized_name = base_name
            else:
                count = implicit_name_counts.get(base_name, 0)
                normalized_name = base_name if count == 0 else f"{base_name}_{count}"
                implicit_name_counts[base_name] = count + 1
            normalized_sources.append(
                MotionSource(
                    motion_name=normalized_name,
                    path=source.path,
                    input_fps=source.input_fps,
                    frame_range=source.frame_range,
                    explicit_motion_name=source.explicit_motion_name,
                )
            )
        return normalized_sources

    def _resolve_entry(self, entry: Any) -> list[MotionSource]:
        if isinstance(entry, Mapping):
            raw_path = entry.get(
                "paths",
                entry.get(
                    "files",
                    entry.get(
                        "csv_files",
                        entry.get("path", entry.get("file", entry.get("csv_path", None))),
                    ),
                ),
            )
            if raw_path is None:
                raise ValueError(
                    "Motion entry dict must contain one of: path, paths, file, files, csv_path, csv_files."
                )
            explicit_motion_name = entry.get("name") is not None
            if explicit_motion_name:
                entry_name = str(entry["name"])
            else:
                first_path = _as_list(raw_path)[0]
                entry_name = Path(str(first_path)).stem
            input_fps = float(entry.get("input_fps", self.default_input_fps))
            frame_range = _normalize_frame_range(
                entry.get("frame_range", self.default_frame_range)
            )
        else:
            raw_path = entry
            entry_name = Path(str(raw_path)).stem
            input_fps = self.default_input_fps
            frame_range = self.default_frame_range
            explicit_motion_name = False

        resolved_paths = self._expand_paths(raw_path)
        output_sources: list[MotionSource] = []
        for path in resolved_paths:
            motion_name = (
                entry_name
                if explicit_motion_name or len(resolved_paths) == 1
                else path.stem
            )
            output_sources.append(
                MotionSource(
                    motion_name=motion_name,
                    path=path,
                    input_fps=input_fps,
                    frame_range=frame_range,
                    explicit_motion_name=explicit_motion_name,
                )
            )
        return output_sources

    def _expand_paths(self, raw_path: Any) -> list[Path]:
        out_paths: list[Path] = []

        for item in _as_list(raw_path):
            path = Path(str(item)).expanduser()
            if not path.is_absolute() and self.csv_root is not None:
                path = self.csv_root / path
            path = path.resolve()

            if path.is_dir():
                dir_paths = sorted(path.glob("*.csv")) + sorted(path.glob("*.npz"))
                if not dir_paths:
                    raise FileNotFoundError(f"No *.csv or *.npz files found in {path}")
                out_paths.extend(dir_paths)
                continue

            if path.is_file():
                out_paths.append(path)
                continue

            if path.suffix == "":
                guessed: list[Path] = []
                for ext in (".csv", ".npz"):
                    candidate = path.with_suffix(ext)
                    if candidate.is_file():
                        guessed.append(candidate)
                if guessed:
                    out_paths.extend(sorted(guessed))
                    continue

            raise FileNotFoundError(f"Could not resolve motion source: {path}")

        return out_paths

    @property
    def num_traj(self) -> int:
        return len(self._trajectory_info_list)

    @property
    def control_dt(self) -> float:
        return 1.0 / self.control_freq

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

        grouped_sources: dict[str, list[MotionSource]] = {}
        for source in self.motion_sources:
            grouped_sources.setdefault(source.motion_name, []).append(source)

        for motion_name, motion_sources in grouped_sources.items():
            motion_group: zarr.Group | None = None
            if build_zarr_dataset and dataset_group is not None:
                motion_group = dataset_group.create_group(motion_name)

            motion_entry = motion_info_dict.setdefault(self.dataset_name, {}).setdefault(
                motion_name,
                {
                    "motion_name": motion_name,
                    "trajectory_indices": [],
                    "trajectory_lengths": [],
                    "trajectory_local_start_indices": [],
                    "trajectory_local_end_indices": [],
                    "source_files": [],
                    "source_fps": [],
                    "output_fps": [],
                },
            )

            local_start_cursor = 0
            for local_idx, source in enumerate(motion_sources):
                traj_data, source_fps, output_fps = self._load_motion(source)
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
                    trajectory_in_motion=local_idx,
                    start=local_start,
                    end=local_end,
                )
                trajectory_info_list.append(traj_info.to_dict())

                motion_entry["trajectory_indices"].append(global_idx)
                motion_entry["trajectory_lengths"].append(traj_len)
                motion_entry["trajectory_local_start_indices"].append(local_start)
                motion_entry["trajectory_local_end_indices"].append(local_end)
                motion_entry["source_files"].append(str(source.path))
                motion_entry["source_fps"].append(float(source_fps))
                motion_entry["output_fps"].append(float(output_fps))

                if motion_group is not None:
                    traj_group = motion_group.create_group(f"trajectory_{local_idx}")
                    self._save_trajectory_data(
                        traj_group,
                        traj_data,
                        chunk_size=chunk_size,
                        shard_size=shard_size,
                    )

                global_idx += 1

            # Backward-compat convenience aliases for single/multi-source entries.
            if motion_entry["source_files"]:
                motion_entry["source_file"] = motion_entry["source_files"][0]
            if motion_entry["source_fps"]:
                motion_entry["source_fps_single"] = motion_entry["source_fps"][0]
            if motion_entry["output_fps"]:
                motion_entry["output_fps_single"] = motion_entry["output_fps"][0]

            if motion_group is not None:
                motion_group.attrs["num_trajectories"] = len(motion_sources)
                motion_group.attrs["trajectory_lengths"] = motion_entry["trajectory_lengths"]
                motion_group.attrs["source_files"] = motion_entry["source_files"]
                motion_group.attrs["source_fps"] = motion_entry["source_fps"]
                motion_group.attrs["output_fps"] = motion_entry["output_fps"]

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
            dataset_group.attrs["control_freq"] = self.control_freq
            dataset_group.attrs["trajectory_info_list"] = trajectory_info_list
            dataset_group.attrs["motion_info_dict"] = motion_info_dict
            self.logger.info("Saved trajectories to Zarr store at %s", path)

        self.logger.info(
            "Built trajectory manifest with %d entries across %d motions",
            len(trajectory_info_list),
            sum(len(motions) for motions in motion_info_dict.values()),
        )
        return trajectory_info_list, motion_info_dict

    def _discover_metadata(self) -> DatasetMeta:
        trajectory_lengths = [int(e["length"]) for e in self._trajectory_info_list]
        return DatasetMeta(
            name=self.dataset_name,
            source=self.dataset_source,
            version="1.0.0",
            citation=(
                "LAFAN1 motions prepared from CSV files and resampled with "
                "Unitree-style preprocessing."
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
                "control_freq": self.control_freq,
                "sources": [
                    {
                        "motion_name": source.motion_name,
                        "path": str(source.path),
                        "input_fps": source.input_fps,
                        "frame_range": source.frame_range,
                    }
                    for source in self.motion_sources
                ],
            },
        )

    def _infer_or_validate_names(self, traj_data: dict[str, np.ndarray]) -> None:
        joint_count = int(traj_data["joint_pos"].shape[-1])
        if self._joint_names is None:
            self._joint_names = [f"joint_{i}" for i in range(joint_count)]
        elif len(self._joint_names) != joint_count:
            raise ValueError(
                f"joint_names length mismatch: expected {joint_count}, got {len(self._joint_names)}."
            )

        if "body_pos_w" in traj_data:
            body_data = traj_data["body_pos_w"]
            body_count = int(body_data.shape[1]) if body_data.ndim >= 3 else 1
            if self._body_names is None:
                self._body_names = [f"body_{i}" for i in range(body_count)]
            elif len(self._body_names) != body_count:
                raise ValueError(
                    f"body_names length mismatch: expected {body_count}, got {len(self._body_names)}."
                )
        elif self._body_names is None:
            self._body_names = []

        if self._site_names is None:
            self._site_names = []

    def _load_motion(
        self, source: MotionSource
    ) -> tuple[dict[str, np.ndarray], float, float]:
        suffix = source.path.suffix.lower()
        if suffix == ".csv":
            return self._load_csv_motion(source)
        if suffix == ".npz":
            return self._load_npz_motion(source)
        raise ValueError(f"Unsupported motion file extension: {source.path}")

    def _load_csv_motion(
        self, source: MotionSource
    ) -> tuple[dict[str, np.ndarray], float, float]:
        motion = np.loadtxt(source.path, delimiter=",", dtype=np.float32)
        if motion.ndim == 1:
            motion = motion[None, :]
        if motion.shape[1] < 8:
            raise ValueError(
                f"CSV {source.path} must have >= 8 columns, got {motion.shape[1]}."
            )
        motion = self._apply_frame_range(motion, source.frame_range, source.path)
        root_pos = motion[:, :3]
        root_quat = motion[:, 3:7][:, [3, 0, 1, 2]]  # xyzw -> wxyz
        joint_pos = motion[:, 7:]

        root_pos, root_quat, joint_pos = self._resample_motion(
            root_pos=root_pos,
            root_quat=root_quat,
            joint_pos=joint_pos,
            input_fps=source.input_fps,
            output_fps=self.control_freq,
        )
        root_lin_vel, root_ang_vel, joint_vel = self._compute_velocities(
            root_pos=root_pos,
            root_quat=root_quat,
            joint_pos=joint_pos,
            dt=self.control_dt,
        )

        traj_data = self._build_trajectory_dict(
            root_pos=root_pos,
            root_quat=root_quat,
            joint_pos=joint_pos,
            root_lin_vel=root_lin_vel,
            root_ang_vel=root_ang_vel,
            joint_vel=joint_vel,
            extra_data=None,
        )
        return traj_data, source.input_fps, self.control_freq

    def _load_npz_motion(
        self, source: MotionSource
    ) -> tuple[dict[str, np.ndarray], float, float]:
        with np.load(source.path) as npz_data:
            arrays = {key: np.asarray(npz_data[key]) for key in npz_data.files}

        source_fps = float(np.asarray(arrays.get("fps", source.input_fps)).reshape(-1)[0])
        if source_fps <= 0.0:
            raise ValueError(f"Invalid source fps ({source_fps}) for {source.path}")

        if "qpos" in arrays:
            qpos = arrays["qpos"].astype(np.float32)
            qpos = self._apply_frame_range(qpos, source.frame_range, source.path)
            root_pos = qpos[:, :3]
            root_quat = qpos[:, 3:7]
            joint_pos = qpos[:, 7:]

            qvel = arrays.get("qvel")
            if qvel is not None:
                qvel = qvel.astype(np.float32)
                qvel = self._apply_frame_range(qvel, source.frame_range, source.path)
                root_lin_vel = qvel[:, :3]
                root_ang_vel = qvel[:, 3:6]
                joint_vel = qvel[:, 6:]
            else:
                root_lin_vel = None
                root_ang_vel = None
                joint_vel = None
            extra_data = {
                key: self._apply_frame_range(value, source.frame_range, source.path)
                for key, value in arrays.items()
                if key in OPTIONAL_BODY_KEYS
            }
        else:
            joint_pos = arrays.get("joint_pos")
            if joint_pos is None:
                raise ValueError(
                    f"NPZ {source.path} must contain 'qpos' or 'joint_pos'."
                )
            joint_pos = joint_pos.astype(np.float32)
            joint_pos = self._apply_frame_range(joint_pos, source.frame_range, source.path)

            root_pos, root_quat = self._extract_root_pose_from_npz(arrays, source.path)
            root_pos = self._apply_frame_range(root_pos, source.frame_range, source.path)
            root_quat = self._apply_frame_range(root_quat, source.frame_range, source.path)

            joint_vel_raw = arrays.get("joint_vel")
            root_lin_vel_raw, root_ang_vel_raw = self._extract_root_vel_from_npz(arrays)
            joint_vel = (
                self._apply_frame_range(joint_vel_raw, source.frame_range, source.path)
                if joint_vel_raw is not None
                else None
            )
            root_lin_vel = (
                self._apply_frame_range(root_lin_vel_raw, source.frame_range, source.path)
                if root_lin_vel_raw is not None
                else None
            )
            root_ang_vel = (
                self._apply_frame_range(root_ang_vel_raw, source.frame_range, source.path)
                if root_ang_vel_raw is not None
                else None
            )
            extra_data = {
                key: self._apply_frame_range(value, source.frame_range, source.path)
                for key, value in arrays.items()
                if key in OPTIONAL_BODY_KEYS
            }

        root_quat = self._normalize_quat(root_quat.astype(np.float32))
        needs_resample = not np.isclose(source_fps, self.control_freq)

        if needs_resample:
            root_pos, root_quat, joint_pos = self._resample_motion(
                root_pos=root_pos,
                root_quat=root_quat,
                joint_pos=joint_pos,
                input_fps=source_fps,
                output_fps=self.control_freq,
            )
            root_lin_vel, root_ang_vel, joint_vel = self._compute_velocities(
                root_pos=root_pos,
                root_quat=root_quat,
                joint_pos=joint_pos,
                dt=self.control_dt,
            )
            # Body states are not resampled here to avoid introducing FK assumptions.
            extra_data = None
            output_fps = self.control_freq
        else:
            output_fps = source_fps
            if root_lin_vel is None or root_ang_vel is None or joint_vel is None:
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
            extra_data=extra_data,
        )
        return traj_data, source_fps, output_fps

    def _extract_root_pose_from_npz(
        self, arrays: Mapping[str, np.ndarray], path: Path
    ) -> tuple[np.ndarray, np.ndarray]:
        if "root_pos" in arrays and "root_quat" in arrays:
            root_pos = arrays["root_pos"].astype(np.float32)
            root_quat = arrays["root_quat"].astype(np.float32)
            return root_pos, root_quat

        if "body_pos_w" in arrays and "body_quat_w" in arrays:
            body_pos = arrays["body_pos_w"].astype(np.float32)
            body_quat = arrays["body_quat_w"].astype(np.float32)
            if body_pos.ndim == 2:
                root_pos = body_pos
            else:
                root_pos = body_pos[:, 0]
            if body_quat.ndim == 2:
                root_quat = body_quat
            else:
                root_quat = body_quat[:, 0]
            return root_pos, root_quat

        raise ValueError(
            f"NPZ {path} must contain root pose data: "
            "'root_pos'/'root_quat' or 'body_pos_w'/'body_quat_w'."
        )

    def _extract_root_vel_from_npz(
        self, arrays: Mapping[str, np.ndarray]
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        root_lin_vel = arrays.get("root_lin_vel")
        root_ang_vel = arrays.get("root_ang_vel")
        if root_lin_vel is not None and root_ang_vel is not None:
            return root_lin_vel.astype(np.float32), root_ang_vel.astype(np.float32)

        body_lin_vel = arrays.get("body_lin_vel_w")
        body_ang_vel = arrays.get("body_ang_vel_w")
        if body_lin_vel is None or body_ang_vel is None:
            return None, None

        body_lin_vel = body_lin_vel.astype(np.float32)
        body_ang_vel = body_ang_vel.astype(np.float32)
        if body_lin_vel.ndim == 2:
            root_lin_vel = body_lin_vel
        else:
            root_lin_vel = body_lin_vel[:, 0]
        if body_ang_vel.ndim == 2:
            root_ang_vel = body_ang_vel
        else:
            root_ang_vel = body_ang_vel[:, 0]
        return root_lin_vel, root_ang_vel

    def _build_trajectory_dict(
        self,
        *,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
        joint_pos: np.ndarray,
        root_lin_vel: np.ndarray,
        root_ang_vel: np.ndarray,
        joint_vel: np.ndarray,
        extra_data: Mapping[str, np.ndarray] | None,
    ) -> dict[str, np.ndarray]:
        qpos = np.concatenate([root_pos, root_quat, joint_pos], axis=-1).astype(
            np.float32
        )
        qvel = np.concatenate(
            [root_lin_vel, root_ang_vel, joint_vel], axis=-1
        ).astype(np.float32)

        traj_data: dict[str, np.ndarray] = {
            "qpos": qpos,
            "qvel": qvel,
            "root_pos": root_pos.astype(np.float32),
            "root_quat": root_quat.astype(np.float32),
            "root_lin_vel": root_lin_vel.astype(np.float32),
            "root_ang_vel": root_ang_vel.astype(np.float32),
            "joint_pos": joint_pos.astype(np.float32),
            "joint_vel": joint_vel.astype(np.float32),
        }

        if extra_data is not None:
            for key in OPTIONAL_BODY_KEYS:
                if key not in extra_data:
                    continue
                value = np.asarray(extra_data[key], dtype=np.float32)
                if value.ndim == 0:
                    continue
                if value.shape[0] != qpos.shape[0]:
                    continue
                traj_data[key] = value

        return traj_data

    def _apply_frame_range(
        self,
        array: np.ndarray,
        frame_range: tuple[int, int] | None,
        source_path: Path,
    ) -> np.ndarray:
        if frame_range is None:
            return array
        if array.ndim == 0:
            return array
        start, end = frame_range
        if end > array.shape[0]:
            raise ValueError(
                f"frame_range {frame_range} exceeds length {array.shape[0]} for {source_path}"
            )
        return array[start - 1 : end]

    def _resample_motion(
        self,
        *,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
        joint_pos: np.ndarray,
        input_fps: float,
        output_fps: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if root_pos.shape[0] == 0:
            raise ValueError("Cannot resample empty motion.")
        if output_fps <= 0.0:
            raise ValueError("output_fps must be positive.")
        if root_pos.shape[0] == 1 or np.isclose(input_fps, output_fps):
            return (
                root_pos.astype(np.float32),
                self._normalize_quat(root_quat.astype(np.float32)),
                joint_pos.astype(np.float32),
            )

        input_dt = 1.0 / input_fps
        output_dt = 1.0 / output_fps
        duration = (root_pos.shape[0] - 1) * input_dt
        if duration <= 0.0:
            return (
                root_pos[:1].astype(np.float32),
                self._normalize_quat(root_quat[:1].astype(np.float32)),
                joint_pos[:1].astype(np.float32),
            )

        times = np.arange(0.0, duration, output_dt, dtype=np.float64)
        if times.size == 0:
            times = np.array([0.0], dtype=np.float64)

        index_0, index_1, blend = self._compute_frame_blend(
            times=times,
            duration=duration,
            input_frames=root_pos.shape[0],
        )

        root_pos_out = self._lerp(root_pos[index_0], root_pos[index_1], blend[:, None])
        root_quat_out = self._slerp(
            self._normalize_quat(root_quat[index_0]),
            self._normalize_quat(root_quat[index_1]),
            blend,
        )
        joint_pos_out = self._lerp(
            joint_pos[index_0], joint_pos[index_1], blend[:, None]
        )
        return (
            root_pos_out.astype(np.float32),
            root_quat_out.astype(np.float32),
            joint_pos_out.astype(np.float32),
        )

    def _compute_frame_blend(
        self,
        *,
        times: np.ndarray,
        duration: float,
        input_frames: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        phase = times / duration
        index_0 = np.floor(phase * (input_frames - 1)).astype(np.int64)
        index_1 = np.minimum(index_0 + 1, input_frames - 1)
        blend = phase * (input_frames - 1) - index_0
        return index_0, index_1, blend.astype(np.float32)

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
        root_ang_vel = self._so3_derivative(root_quat, dt).astype(np.float32)
        return root_lin_vel, root_ang_vel, joint_vel

    def _so3_derivative(self, rotations: np.ndarray, dt: float) -> np.ndarray:
        q = self._normalize_quat(rotations.astype(np.float32))
        n = q.shape[0]
        if n <= 1:
            return np.zeros((n, 3), dtype=np.float32)
        if n == 2:
            q_rel = self._quat_mul(q[1:2], self._quat_conjugate(q[0:1]))
            omega = self._quat_to_axis_angle(q_rel) / dt
            return np.repeat(omega, 2, axis=0).astype(np.float32)

        q_prev = q[:-2]
        q_next = q[2:]
        q_rel = self._quat_mul(q_next, self._quat_conjugate(q_prev))
        omega = self._quat_to_axis_angle(q_rel) / (2.0 * dt)
        omega = np.concatenate([omega[:1], omega, omega[-1:]], axis=0)
        return omega.astype(np.float32)

    def _lerp(self, a: np.ndarray, b: np.ndarray, blend: np.ndarray) -> np.ndarray:
        return a * (1.0 - blend) + b * blend

    def _slerp(self, a: np.ndarray, b: np.ndarray, blend: np.ndarray) -> np.ndarray:
        return self._quat_slerp_batch(a, b, blend)

    def _normalize_quat(self, quat: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(quat, axis=-1, keepdims=True)
        norm = np.where(norm < EPS, 1.0, norm)
        return quat / norm

    def _quat_conjugate(self, quat: np.ndarray) -> np.ndarray:
        out = quat.copy()
        out[..., 1:] *= -1.0
        return out

    def _quat_mul(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
        w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
        return np.stack(
            (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ),
            axis=-1,
        )

    def _quat_to_axis_angle(self, quat: np.ndarray) -> np.ndarray:
        q = self._normalize_quat(quat)
        w = np.clip(q[..., 0], -1.0, 1.0)
        xyz = q[..., 1:]
        xyz_norm = np.linalg.norm(xyz, axis=-1, keepdims=True)

        axis = np.divide(
            xyz,
            np.where(xyz_norm < EPS, 1.0, xyz_norm),
            out=np.zeros_like(xyz),
            where=xyz_norm >= EPS,
        )
        angle = 2.0 * np.arctan2(xyz_norm[..., 0], w)
        axis_angle = axis * angle[..., None]
        axis_angle[xyz_norm[..., 0] < EPS] = 0.0
        return axis_angle

    def _quat_slerp_batch(
        self, q0: np.ndarray, q1: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        qa = self._normalize_quat(q0.astype(np.float32))
        qb = self._normalize_quat(q1.astype(np.float32))
        t = np.asarray(t, dtype=np.float32)

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
            out[linear_mask] = self._normalize_quat(
                qa[linear_mask] * (1.0 - t_linear) + qb[linear_mask] * t_linear
            )

        if np.any(~linear_mask):
            theta_0 = np.arccos(dot[~linear_mask])
            sin_theta_0 = np.sin(theta_0)
            theta = theta_0 * t[~linear_mask]
            s0 = np.sin(theta_0 - theta) / np.maximum(sin_theta_0, EPS)
            s1 = np.sin(theta) / np.maximum(sin_theta_0, EPS)
            out[~linear_mask] = qa[~linear_mask] * s0[:, None] + qb[~linear_mask] * s1[
                :, None
            ]
            out[~linear_mask] = self._normalize_quat(out[~linear_mask])
        return out.astype(np.float32)

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
            if array.ndim == 0:
                continue
            if array.shape[0] == 0:
                continue

            chunks = [min(chunk_size, array.shape[0])] + list(array.shape[1:])
            shards = [min(shard_size, array.shape[0])] + list(array.shape[1:])
            ds = traj_group.create_dataset(
                key,
                shape=array.shape,
                dtype=array.dtype,
                chunks=chunks,
                shards=shards,
            )
            ds[:] = array
