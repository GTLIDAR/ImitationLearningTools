"""High-level offline dataset and memmap utilities.

This module ties together three layers that already exist in the repo:

- `BaseLoader` subclasses (e.g. `LocoMuJoCoLoader`) that know how to talk to
  raw sources and can export a Zarr trajectory store.
- `VectorizedTrajectoryDataset` that reads those per-trajectory Zarr layouts in
  a vectorized, multi-env friendly way.
- The replay stack (`replay_export.py` + `replay_manager.py` +
  `replay_memmap.py`) that turns Zarr trajectories into a TorchRL
  `TensorDictReplayBuffer` backed by `LazyMemmapStorage`.

The goal here is to expose a **clear, end-to-end API**:

1. Build an offline trajectory dataset on disk (Zarr) from Loco-MuJoCo.
2. Open an existing offline dataset and inspect its trajectories.
3. Build a memmap-backed replay buffer from the offline dataset and configure
   per-environment (task, trajectory, step) assignments.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import json
import zarr
from omegaconf import DictConfig

from iltools_core.metadata_schema import DatasetMeta

from .base_loader import BaseLoader
from .loco_mujoco.loader import LocoMuJoCoLoader
from .replay_export import build_replay_from_zarr
from .replay_manager import EnvAssignment, ExpertReplayManager


# ---------------------------------------------------------------------------
# Lightweight index over per-trajectory Zarr layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectoryRef:
    """Reference to a single trajectory within a Zarr store.

    The canonical layout we assume is:

        <zarr_root>/<dataset_source>/<motion>/<trajectory>/<key>

    where each `<key>` is a time-major array of shape `[T, ...]`.
    """

    dataset_source: str
    motion: str
    trajectory: str
    length: int

    @property
    def group_path(self) -> str:
        return f"{self.dataset_source}/{self.motion}/{self.trajectory}"


@dataclass
class OfflineDataset:
    """Represents an on-disk offline trajectory dataset.

    This is a thin, read-only façade over:

    - a root directory (for metadata and multiple stores),
    - a trajectories Zarr store (usually `trajectories.zarr`),
    - optional `DatasetMeta` describing the dataset,
    - a flat list of `TrajectoryRef` entries.
    """

    root_dir: Path
    zarr_path: Path
    metadata: Optional[DatasetMeta]
    trajectories: list[TrajectoryRef]

    # ---- Construction -----------------------------------------------------

    @classmethod
    def from_zarr_root(
        cls,
        root_dir: str | Path,
        *,
        zarr_name: str = "trajectories.zarr",
        metadata_filename: str = "meta.json",
        length_key_candidates: Sequence[str] = ("qpos",),
    ) -> "OfflineDataset":
        """Open an existing offline dataset rooted at `root_dir`.

        The method expects (and does **not** create) a Zarr store at
        `<root_dir>/<zarr_name>` with the nested trajectory layout used across
        this repo. If a `meta.json` file is present it is parsed into a
        `DatasetMeta` instance; otherwise `metadata` is left as `None`.
        """
        root_dir = Path(root_dir)
        zarr_path = root_dir / zarr_name
        if not zarr_path.exists():
            raise FileNotFoundError(f"Zarr store not found at {zarr_path}")

        # Optional JSON metadata alongside the store
        meta_path = root_dir / metadata_filename
        metadata: Optional[DatasetMeta] = None
        if meta_path.exists():
            with meta_path.open("r") as f:
                raw = json.load(f)
            metadata = DatasetMeta(**raw)

        zroot = zarr.open_group(str(zarr_path), mode="r")
        if not isinstance(zroot, zarr.Group):  # pragma: no cover - defensive
            raise TypeError(f"Expected zarr.Group at {zarr_path}, got {type(zroot)}")

        traj_refs: list[TrajectoryRef] = []

        for dataset_source in zroot:
            ds_group = zroot[dataset_source]
            if not isinstance(ds_group, zarr.Group):
                continue
            for motion in ds_group:
                motion_group = ds_group[motion]
                if not isinstance(motion_group, zarr.Group):
                    continue
                for traj in motion_group:
                    traj_group = motion_group[traj]
                    if not isinstance(traj_group, zarr.Group):
                        continue
                    # Determine length from the first available key candidate
                    length: Optional[int] = None
                    for key in length_key_candidates:
                        if key in traj_group:
                            length = int(traj_group[key].shape[0])
                            break
                    if length is None:
                        # Fallback: use first array-valued key
                        for key in traj_group.keys():
                            arr = traj_group[key]
                            if hasattr(arr, "shape") and len(arr.shape) >= 1:
                                length = int(arr.shape[0])
                                break
                    if length is None:
                        continue
                    traj_refs.append(
                        TrajectoryRef(
                            dataset_source=dataset_source,
                            motion=motion,
                            trajectory=traj,
                            length=length,
                        )
                    )

        return cls(
            root_dir=root_dir,
            zarr_path=zarr_path,
            metadata=metadata,
            trajectories=traj_refs,
        )

    # ---- Introspection helpers -------------------------------------------

    @property
    def num_trajectories(self) -> int:
        return len(self.trajectories)

    @property
    def total_steps(self) -> int:
        return sum(t.length for t in self.trajectories)

    def iter_trajectories(self) -> Iterable[TrajectoryRef]:
        return iter(self.trajectories)

    # ---- Replay utilities -------------------------------------------------

    def build_replay_manager(
        self,
        *,
        scratch_dir: str | Path,
        obs_keys: Sequence[str] | None = None,
        act_key: str | None = None,
        device: str = "cpu",
        **kwargs,
    ) -> ExpertReplayManager:
        """Construct an `ExpertReplayManager` from this offline dataset.

        This is a thin wrapper around `build_replay_from_zarr(...)` that
        forwards most keyword arguments. It always points the Zarr path at
        `self.zarr_path`.
        """
        mgr = build_replay_from_zarr(
            zarr_path=str(self.zarr_path),
            scratch_dir=str(scratch_dir),
            obs_keys=obs_keys,
            act_key=act_key,
            device=device,
            **kwargs,
        )
        return mgr

    def make_sequential_replay(
        self,
        *,
        scratch_dir: str | Path,
        num_envs: int,
        obs_keys: Sequence[str] | None = None,
        act_key: str | None = None,
        device: str = "cpu",
        start_step: int = 0,
        **kwargs,
    ) -> tuple[ExpertReplayManager, list[EnvAssignment]]:
        """Build a replay manager and attach a simple sequential env assignment.

        Envs are assigned to trajectories in a round-robin fashion over the
        flattened `(task_id, traj_id)` pairs as produced by
        `build_replay_from_zarr`. The returned `EnvAssignment` list can be
        further modified by the caller and passed back into
        `ExpertReplayManager.update_assignments`.
        """
        mgr = self.build_replay_manager(
            scratch_dir=scratch_dir,
            obs_keys=obs_keys,
            act_key=act_key,
            device=device,
            **kwargs,
        )

        # Each segment corresponds to one (task_id, traj_id) trajectory.
        segments = mgr.segments
        if not segments:
            raise ValueError("Replay manager has no segments – is the dataset empty?")

        assignments: list[EnvAssignment] = []
        for env_idx in range(int(num_envs)):
            seg = segments[env_idx % len(segments)]
            assignments.append(
                EnvAssignment(task_id=seg.task_id, traj_id=seg.traj_id, step=start_step)
            )

        mgr.set_assignment(assignments)
        return mgr, assignments


# ---------------------------------------------------------------------------
# Loco-MuJoCo specific convenience
# ---------------------------------------------------------------------------


def export_loco_mujoco_offline(
    *,
    output_dir: str | Path,
    env_name: str,
    cfg: DictConfig,
    export_transitions: bool = True,
    metadata_filename: str = "meta.json",
    zarr_name: str = "trajectories.zarr",
    loader_cls: type[BaseLoader] = LocoMuJoCoLoader,
    **loader_kwargs,
) -> OfflineDataset:
    """End-to-end helper: build an offline dataset from Loco-MuJoCo.

    High-level usage:

    - Call this once (offline) to write a Zarr store and metadata:

        offline = export_loco_mujoco_offline(
            output_dir=\"/path/to/dataset\",
            env_name=\"MyLocoEnv\",
            cfg=cfg,
        )

    - Later, open `OfflineDataset.from_zarr_root(...)` and build a replay
      manager via `offline.build_replay_manager(...)` or
      `offline.make_sequential_replay(...)`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories_path = output_dir / zarr_name

    # Instantiate and export via the loader
    loader = loader_cls(env_name=env_name, cfg=cfg, **loader_kwargs)
    loader.save(str(trajectories_path), export_transitions=export_transitions)

    # Persist DatasetMeta alongside the store for quick reopening
    meta_path = output_dir / metadata_filename
    try:
        meta = loader.metadata
        with meta_path.open("w") as f:
            json.dump(meta.model_dump(), f, indent=2)
    except Exception:
        # Metadata is best-effort; failures here should not break export.
        pass

    return OfflineDataset.from_zarr_root(
        output_dir,
        zarr_name=zarr_name,
        metadata_filename=metadata_filename,
    )
