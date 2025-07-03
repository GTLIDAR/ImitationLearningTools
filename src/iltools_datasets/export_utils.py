import os
import json
import logging
import numpy as np
import zarr
from typing import Optional
from iltools_core.metadata_schema import DatasetMeta
from iltools_datasets.base_loader import BaseTrajectoryLoader
from tqdm import tqdm

logger = logging.getLogger("iltools_datasets.export_utils")


def export_trajectories_to_disk(loader: BaseTrajectoryLoader, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    if not isinstance(loader.metadata, DatasetMeta):
        raise TypeError("Loader metadata must be an instance of DatasetMeta")
    # Collect per-trajectory dt
    dt_list = [float(loader[idx].dt) for idx in range(len(loader))]
    metadata = {
        "num_trajectories": len(loader),
        "trajectory_lengths": loader.metadata.trajectory_lengths,
        "observation_keys": loader.metadata.observation_keys,
        "action_keys": loader.metadata.action_keys,
        "dt": dt_list,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    for idx in range(len(loader)):
        traj = loader[idx]
        save_dict = {
            **{k: np.asarray(v) for k, v in traj.observations.items()},
            **{k: np.asarray(v) for k, v in (traj.actions or {}).items()},
        }
        np.savez(
            os.path.join(out_dir, f"traj_{idx}.npz"),
            **save_dict,  # type: ignore
        )


def export_trajectories_to_zarr_per_trajectory(
    loader: BaseTrajectoryLoader,
    out_dir: str,
    chunk_size: int = 1000,
):
    """Export trajectories to per-trajectory Zarr format.

    Each trajectory is stored as a separate Zarr group with structure:
    trajectories.zarr/
      traj_0000/
        observations/{key}  [T, ...]
        actions/{key}       [T, ...]
        metadata.json       (per-trajectory metadata including dt)
      traj_0001/
        ...

    Args:
        loader: The trajectory loader to export.
        out_dir: The directory to save the exported trajectories.
        chunk_size: Chunk size for Zarr arrays (time dimension).
    """
    os.makedirs(out_dir, exist_ok=True)
    zarr_path = os.path.join(out_dir, "trajectories.zarr")

    # Global metadata
    traj_lengths = loader.metadata.trajectory_lengths
    if isinstance(traj_lengths, int):
        traj_lengths = [traj_lengths] * loader.metadata.num_trajectories
    else:
        traj_lengths = list(traj_lengths)

    # Collect per-trajectory dt
    dt_list = [float(loader[idx].dt) for idx in range(len(loader))]

    if hasattr(loader.metadata, "model_dump"):
        global_metadata = loader.metadata.model_dump()
    elif hasattr(loader.metadata, "dict"):
        global_metadata = loader.metadata.dict()
    else:
        global_metadata = dict(loader.metadata)

    global_metadata["dt"] = dt_list
    global_metadata["export_format"] = "per_trajectory_zarr"

    # Write global metadata
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(global_metadata, f)

    # Create Zarr store
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.open_group(store=store, mode="w")

    # Get shapes and dtypes from first trajectory
    first_traj = loader[0]
    obs_shapes = {k: v.shape[1:] for k, v in first_traj.observations.items()}
    obs_dtypes = {k: v.dtype for k, v in first_traj.observations.items()}

    if first_traj.actions:
        act_shapes = {k: v.shape[1:] for k, v in first_traj.actions.items()}
        act_dtypes = {k: v.dtype for k, v in first_traj.actions.items()}
    else:
        act_shapes, act_dtypes = {}, {}

    # Export each trajectory
    for traj_idx in tqdm(range(len(loader)), desc="Exporting trajectories"):
        traj = loader[traj_idx]
        traj_length = traj_lengths[traj_idx]

        # Create trajectory group
        traj_group = root.create_group(f"traj_{traj_idx:04d}")

        # Create observation arrays
        obs_group = traj_group.create_group("observations")
        for key in loader.metadata.observation_keys:
            obs_data = traj.observations[key]
            obs_group.create_dataset(
                key,
                data=obs_data,
                chunks=(min(chunk_size, traj_length), *obs_shapes[key]),
                compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2),
            )

        # Create action arrays (if present)
        if loader.metadata.action_keys:
            act_group = traj_group.create_group("actions")
            for key in loader.metadata.action_keys:
                act_data = traj.actions[key]
                act_group.create_dataset(
                    key,
                    data=act_data,
                    chunks=(min(chunk_size, traj_length), *act_shapes[key]),
                    compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2),
                )

        # Store per-trajectory metadata as attributes
        traj_group.attrs["dt"] = float(traj.dt)
        traj_group.attrs["length"] = traj_length
        traj_group.attrs["trajectory_idx"] = traj_idx

    logger.info(f"Exported {len(loader)} trajectories to {zarr_path}")


def export_trajectories_to_zarr(
    loader: BaseTrajectoryLoader,
    out_dir: str,
    num_workers: int = 8,
    window_size: Optional[int] = None,
    control_freq: Optional[float] = None,
    desired_horizon_steps: Optional[int] = None,
    horizon_multiplier: float = 2.0,
):
    """Legacy export function - use export_trajectories_to_zarr_per_trajectory instead."""
    logger.warning(
        "export_trajectories_to_zarr is deprecated. Use export_trajectories_to_zarr_per_trajectory instead."
    )
    return export_trajectories_to_zarr_per_trajectory(loader, out_dir)
