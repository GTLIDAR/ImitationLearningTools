import os
import json
import logging
import numpy as np
import zarr
from typing import Optional
from iltools_core.metadata_schema import DatasetMeta
from iltools_datasets.base_loader import BaseTrajectoryLoader

logger = logging.getLogger("iltools_datasets.export_utils")


def export_trajectories_to_disk(loader: BaseTrajectoryLoader, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    if not isinstance(loader.metadata, DatasetMeta):
        raise TypeError("Loader metadata must be an instance of DatasetMeta")
    metadata = {
        "num_trajectories": len(loader),
        "trajectory_lengths": loader.metadata.trajectory_lengths,
        "observation_keys": loader.metadata.observation_keys,
        "action_keys": loader.metadata.action_keys,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    for idx in range(len(loader)):
        traj = loader[idx]
        save_dict = {
            **{k: np.asarray(v) for k, v in traj.observations.items()},
            **{k: np.asarray(v) for k, v in (traj.actions or {}).items()},
            "dt": np.array(traj.dt, dtype=np.float32),
        }
        np.savez(
            os.path.join(out_dir, f"traj_{idx}.npz"),
            **save_dict,  # type: ignore
        )


def export_trajectories_to_zarr(
    loader: BaseTrajectoryLoader,
    out_dir: str,
    num_workers: int = 8,
    window_size: Optional[int] = None,
    control_freq: Optional[float] = None,
):
    """Export trajectories to Zarr format.

    Args:
        loader: The trajectory loader to export.
        out_dir: The directory to save the exported trajectories.
        num_workers: The number of workers to use for parallel export.
        window_size: The size of the window to use for exporting the trajectories.
        control_freq: Control frequency to use for all trajectories. If None, uses loader's default.
    """
    os.makedirs(out_dir, exist_ok=True)
    zarr_path = os.path.join(out_dir, "trajectories.zarr")
    traj_lengths = loader.metadata.trajectory_lengths  # type: ignore
    if isinstance(traj_lengths, int):
        traj_lengths = [traj_lengths] * loader.metadata.num_trajectories
    else:
        traj_lengths = list(traj_lengths)
    if hasattr(loader.metadata, "model_dump"):
        metadata = loader.metadata.model_dump()
    elif hasattr(loader.metadata, "dict"):
        metadata = loader.metadata.dict()
    else:
        metadata = dict(loader.metadata)
    metadata["window_size"] = window_size
    if control_freq is not None:
        metadata["export_control_freq"] = control_freq
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.open_group(store=store, mode="w")
    try:
        # Get first trajectory with consistent frequency
        if hasattr(loader, "__getitem__") and control_freq is not None:
            first_traj = loader.__getitem__(0, control_freq=control_freq)
        else:
            first_traj = loader[0]
        first_traj_len = int(next(iter(first_traj.observations.values())).shape[0])
        obs_shapes = {
            k: first_traj.observations[k].shape[1:]
            for k in loader.metadata.observation_keys
        }
        obs_dtypes = {
            k: first_traj.observations[k].dtype
            for k in loader.metadata.observation_keys
        }
        if loader.metadata.action_keys:
            act_shapes = {
                k: first_traj.actions[k].shape[1:] for k in loader.metadata.action_keys
            }
            act_dtypes = {
                k: first_traj.actions[k].dtype for k in loader.metadata.action_keys
            }
        else:
            act_shapes, act_dtypes = {}, {}
    except Exception as e:
        logger.error(f"Failed to inspect first trajectory: {e}")
        raise
    if window_size is not None:
        win_size = int(window_size)
        total_windows = sum(max(1, (int(l) - win_size + 1)) for l in traj_lengths)
    else:
        win_size = int(first_traj_len)
        total_windows = len(loader)
    obs_arrays = {}
    for k in loader.metadata.observation_keys:
        obs_arrays[k] = root.create_dataset(
            f"observations/{k}",
            shape=(int(total_windows), win_size, *obs_shapes[k]),
            dtype=obs_dtypes[k],
            chunks=(min(1024, int(total_windows)), win_size, *obs_shapes[k]),
            compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2),
        )
    act_arrays = {}
    for k in loader.metadata.action_keys or []:
        act_arrays[k] = root.create_dataset(
            f"actions/{k}",
            shape=(int(total_windows), win_size, *act_shapes[k]),
            dtype=act_dtypes[k],
            chunks=(min(1024, int(total_windows)), win_size, *act_shapes[k]),
            compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2),
        )
    dt_array = root.create_dataset(
        "dt",
        shape=(int(total_windows),),
        dtype=np.float32,
        chunks=(min(1024, int(total_windows)),),
        compressor=zarr.Blosc(),
    )

    def get_windows(traj, window_size):
        length = int(next(iter(traj.observations.values())).shape[0])
        win_size = window_size or length
        for start in range(0, max(1, length - win_size + 1)):
            obs_win = {
                k: v[start : start + win_size] for k, v in traj.observations.items()
            }
            act_win = {
                k: v[start : start + win_size] for k, v in (traj.actions or {}).items()
            }
            yield obs_win, act_win, traj.dt

    import concurrent.futures

    results = []

    def process_traj(idx):
        try:
            # Use consistent frequency for all trajectories
            if hasattr(loader, "__getitem__") and control_freq is not None:
                traj = loader.__getitem__(int(idx), control_freq=control_freq)
            else:
                traj = loader[int(idx)]
            if window_size is not None:
                return list(get_windows(traj, window_size))
            else:
                return [(traj.observations, traj.actions or {}, traj.dt)]
        except Exception as e:
            logger.error(f"Failed to process trajectory {idx}: {e}")
            return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        fut_to_idx = {
            executor.submit(process_traj, int(idx)): int(idx)
            for idx in range(len(loader))
        }
        for fut in concurrent.futures.as_completed(fut_to_idx):
            idx = int(fut_to_idx[fut])
            try:
                res = fut.result()
                results.append((idx, res))
            except Exception as e:
                logger.error(f"Export failed for trajectory {idx}: {e}")
    flat_results = []
    for idx, winlist in sorted(results):
        flat_results.extend(winlist)
    for i, (obs, act, dt) in enumerate(flat_results):
        for k, arr in obs.items():
            obs_arrays[k][int(i), ...] = np.asarray(arr)
        for k, arr in act.items():
            act_arrays[k][int(i), ...] = np.asarray(arr)
        dt_array[int(i)] = np.float32(dt)
    logger.info(f"Exported {len(flat_results)} windows/trajectories to {zarr_path}")
