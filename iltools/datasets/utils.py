"""Utilities to build replay buffers from Zarr datasets (Zarr v3+).

Exports a Zarr trajectory dataset into a TorchRL TensorDictReplayBuffer backed by
LazyMemmapStorage on CPU or LazyTensorStorage on CUDA.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import zarr
from tensordict import TensorDict
from torchrl.data import LazyMemmapStorage, LazyTensorStorage, TensorStorage
from torchrl.data.replay_buffers import TensorDictReplayBuffer

logger = logging.getLogger(f"{__name__}.utils")

#: Sidecar written next to a persisted memmap buffer so it can be reopened
#: without walking the Zarr hierarchy again.
_PERSIST_MANIFEST_NAME = "iltools_rb_manifest.json"
_PERSIST_FORMAT_VERSION = 1


def _normalize_selection(x: str | Iterable[str] | None) -> list[str] | None:
    """Canonical form of a datasets/motions/trajectories/keys selection."""
    if x is None:
        return None
    if isinstance(x, str):
        return [x]
    return [str(item) for item in x]


def _persist_key(
    zarr_path: Path,
    datasets: str | Iterable[str] | None,
    motions: str | Iterable[str] | None,
    trajectories: str | Iterable[str] | None,
    keys: str | Iterable[str] | None,
    persist_id: str | None,
) -> dict:
    """Identity of a persisted buffer: which Zarr content, and which slice of it.

    With ``persist_id`` the identity is content-addressed and therefore
    relocatable: a buffer built on one machine can be copied to another and
    reopened there, even though the source Zarr lives at a different absolute
    path (or is not present at all). Without it the absolute Zarr path is used,
    which is safe for a build-and-train-in-place workflow but will spuriously
    invalidate a buffer that has been moved.
    """
    return {
        "source": (
            {"persist_id": str(persist_id)}
            if persist_id is not None
            else {"zarr_path": str(zarr_path.resolve())}
        ),
        "datasets": _normalize_selection(datasets),
        "motions": _normalize_selection(motions),
        "trajectories": _normalize_selection(trajectories),
        "keys": _normalize_selection(keys),
    }


def _load_persisted_rb(
    persist_dir: Path,
    expected_key: dict,
    *,
    device: torch.device,
    pin_memory: bool,
    prefetch: int,
    batch_size: int,
    compilable: bool,
) -> tuple[TensorDictReplayBuffer, dict] | None:
    """Reopen a previously persisted memmap buffer, or None if unusable.

    The persisted directory is always CPU memmap files -- that is the portable
    on-disk form. ``device`` selects where the *runtime* buffer lives: a CUDA
    device materializes the memmap into VRAM with one sequential read, which is
    what you want whenever the reference set fits, because sampling a memmap is
    several times slower than sampling GPU-resident tensors.

    Returns None (rather than raising) whenever the sidecar is missing, was
    written by another format version, or describes different content, so the
    caller can simply rebuild.
    """
    manifest_path = persist_dir / _PERSIST_MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        logger.warning("Unreadable buffer manifest at %s; rebuilding.", manifest_path)
        return None

    if manifest.get("format_version") != _PERSIST_FORMAT_VERSION:
        logger.info("Persisted buffer format mismatch at %s; rebuilding.", persist_dir)
        return None
    if manifest.get("key") != expected_key:
        logger.info(
            "Persisted buffer at %s was built for a different Zarr or selection; "
            "rebuilding.",
            persist_dir,
        )
        return None

    try:
        td = TensorDict.load_memmap(persist_dir)
    except Exception as err:  # noqa: BLE001 - any failure means "rebuild"
        logger.warning("Could not memory-map %s (%s); rebuilding.", persist_dir, err)
        return None

    written = int(manifest["traj_info"]["written"])
    if int(td.shape[0]) < written:
        logger.warning(
            "Persisted buffer at %s holds %s rows but its manifest claims %s; "
            "rebuilding.",
            persist_dir,
            int(td.shape[0]),
            written,
        )
        return None

    if device.type != "cpu":
        # One sequential read of the whole buffer, rather than the scattered
        # page faults a memmap pays on every sample.
        load_start = time.perf_counter()
        td = td.to(device)
        logger.info(
            "Materialized persisted buffer onto %s in %.1f s.",
            device,
            time.perf_counter() - load_start,
        )

    storage = TensorStorage(
        td,
        max_size=int(manifest["traj_info"]["capacity"]),
        device=device,
        compilable=compilable,
    )
    # TensorStorage infers no write cursor from a pre-built tensordict; the
    # buffer is read-only here, and `_len` is what bounds sampling.
    storage._len = written
    rb = TensorDictReplayBuffer(
        storage=storage,
        pin_memory=pin_memory,
        prefetch=prefetch,
        batch_size=batch_size,
    )
    traj_info = dict(manifest["traj_info"])
    traj_info["ordered_traj_list"] = [
        tuple(entry) for entry in traj_info["ordered_traj_list"]
    ]
    logger.info(
        "Reopened persisted replay buffer at %s (%s transitions, %s trajectories).",
        persist_dir,
        written,
        len(traj_info["ordered_traj_list"]),
    )
    return rb, traj_info


def _is_transition_aligned_key(key: str) -> bool:
    """Return True when a stored key is already aligned to transition length T-1."""
    return key in {"obs", "next_obs", "done", "absorbing"} or key.startswith("next_")


def _zarr_array_to_torch(
    arr: np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Convert a numpy array to torch, avoiding copies when possible."""
    t = torch.from_numpy(arr) if isinstance(arr, np.ndarray) else torch.as_tensor(arr)
    return t.to(device=device)


def _discover_or_make_list(
    parent: zarr.Group,
    x: str | Iterable[str] | None,
) -> list[str]:
    """Discover or make a list from input."""
    if x is None:
        return list(parent.keys())
    if isinstance(x, str):
        return [x]
    return x


def _compute_total_transitions(
    root: zarr.Group,
    datasets: Sequence[str],
) -> int:
    """Try to compute total transitions from attrs when possible; otherwise fallback to reading shapes."""
    total = 0

    for dataset in _discover_or_make_list(root, datasets):
        ds_grp = root[dataset]

        # If the dataset stores a trajectory_info_list in attrs, prefer that.
        # But note: your original code sums ALL lengths in trajectory_info_list, regardless of
        # motion/trajectory selection. Here we compute only for the selected ones if possible.
        info_list = ds_grp.attrs.get("trajectory_info_list", None)

        if info_list is not None:
            # If info_list is structured with names, you can filter here.
            # Since schemas differ, we do a conservative approach:
            # - if info entries have a "length" only, we can't map to chosen trajectories
            #   -> we assume info_list is for all chosen transitions and sum it.
            # You can customize filtering if your attrs include identifiers.
            # Note that since transitions are length-1, we subtract 1 from each length.
            try:
                total += int(sum(int(info["length"]) - 1 for info in info_list))
                continue
            except Exception:
                pass

    return total


def make_rb_from(
    zarr_path: str | Path,
    datasets: str | Iterable[str] | None = None,
    motions: str | Iterable[str] | None = None,
    trajectories: str | Iterable[str] | None = None,
    keys: str | Iterable[str] | None = None,
    scratch_dir: str | Path | None = None,
    device: str | torch.device = "cpu",
    existsok: bool = True,
    compilable: bool = True,
    verbose_tree: bool = True,
    pin_memory: bool = True,
    prefetch: int = 0,
    batch_size: int = 1,
    persist_dir: str | Path | None = None,
    persist_rebuild: bool = False,
    persist_id: str | None = None,
) -> tuple[TensorDictReplayBuffer, dict]:
    """Build a TorchRL replay buffer from a Zarr trajectory dataset.

    Args:
        zarr_path: Path to a Zarr root group.
        datasets/motions/trajectories: selections within the Zarr hierarchy.
        keys: if None, use all array keys in each trajectory group. If provided,
              only those keys are loaded.
        scratch_dir: directory for memmap files when using CPU storage.
        device: torch device for tensors in the RB.
        existsok/compilable: passed to the underlying TorchRL storage.
        verbose_tree: print zarr tree at start.
        persist_dir: CPU storage only. Directory holding a reusable memmap copy
            of the filled buffer. When it already contains a matching build, the
            buffer is memory-mapped straight from disk and the Zarr hierarchy is
            never walked; otherwise the buffer is filled there and a sidecar
            manifest is written for next time. This matters at scale: filling
            costs roughly 66 ms per trajectory plus 53 us per frame, so a
            129,785-clip reference set takes hours to fill and milliseconds to
            memory-map.
        persist_rebuild: ignore any existing persisted buffer and refill it.
        persist_id: content identity for the persisted buffer, making it
            relocatable. Set this to something that names the source *content*
            (a manifest sha256, a dataset release tag) and the buffer can be
            built on one machine, copied to another, and reopened there without
            the Zarr being present at all. Leave it None only when the buffer is
            built and consumed in place.

    Note:
        A persisted buffer is validated against ``persist_id`` (or, without one,
        the absolute Zarr path) plus the selection arguments. It is NOT
        invalidated by a Zarr rebuilt in place, nor by a ``persist_id`` you reuse
        for changed content. Pass ``persist_rebuild=True`` (or delete the
        directory) whenever the underlying content changes.

    Returns:
        TensorDictReplayBuffer filled with all selected transitions.
    """
    zarr_path = Path(zarr_path)
    requested_device = torch.device(device)
    persist_path = Path(persist_dir) if persist_dir is not None else None
    if persist_id is not None and persist_path is None:
        raise ValueError("persist_id is only meaningful together with persist_dir.")
    persist_identity = (
        _persist_key(zarr_path, datasets, motions, trajectories, keys, persist_id)
        if persist_path is not None
        else None
    )
    if persist_path is not None and not persist_rebuild:
        cached = _load_persisted_rb(
            persist_path,
            persist_identity,
            device=requested_device,
            pin_memory=pin_memory,
            prefetch=prefetch,
            batch_size=batch_size,
            compilable=compilable,
        )
        if cached is not None:
            return cached

    # The persisted form is always CPU memmap files: that is what is portable
    # across machines and what can exceed VRAM. When the caller wants the
    # runtime buffer on a GPU we still fill to disk first, then reopen through
    # the cached path so the build and the reuse path produce the same object.
    fill_device = requested_device
    if persist_path is not None:
        persist_path.mkdir(parents=True, exist_ok=True)
        scratch_dir = persist_path
        existsok = True
        fill_device = torch.device("cpu")

    root = zarr.open(zarr_path, mode="r")
    if not isinstance(root, zarr.Group):
        raise TypeError(f"Expected Zarr Group at root, got {type(root)}")

    if verbose_tree:
        logger.info("Zarr tree: %s", root.tree())

    device_t = fill_device

    # 1) Capacity
    capacity = _compute_total_transitions(
        root,
        datasets,
    )
    if capacity <= 0:
        raise ValueError("Computed non-positive capacity; check selections/structure.")

    # 2) Storage + RB
    if device_t.type == "cuda":
        storage = LazyTensorStorage(
            capacity,
            device=device_t,
            compilable=compilable,
        )
    else:
        storage = LazyMemmapStorage(
            capacity,
            scratch_dir=None if scratch_dir is None else str(Path(scratch_dir)),
            device=device_t,
            existsok=existsok,
            compilable=compilable,
        )
    rb = TensorDictReplayBuffer(
        storage=storage, pin_memory=pin_memory, prefetch=prefetch, batch_size=batch_size
    )

    # 3) Fill
    written = 0
    # Get a list of start index and end index for each trajectory
    start_indices = []
    end_indices = []
    # Get the ordered trajectory list as a list of tuples (dataset, motion, trajectory)
    trajectory_list = []

    if datasets is None:
        datasets = list(root.group_keys())
    if isinstance(datasets, str):
        datasets = [datasets]

    for dataset in _discover_or_make_list(root, datasets):
        ds_grp = root[dataset]
        for motion in _discover_or_make_list(ds_grp, motions):
            for trajectory in _discover_or_make_list(ds_grp[motion], trajectories):
                traj_grp = ds_grp[motion][trajectory]
                if not isinstance(traj_grp, zarr.Group):
                    raise TypeError(
                        f"Expected Zarr Group at {dataset}/{motion}/{trajectory}, "
                        f"got {type(traj_grp)}"
                    )

                # Determine keys for this trajectory *locally* (don’t mutate outer `keys`)
                if keys is None:
                    k_list = list(traj_grp.array_keys())
                else:
                    k_list = [keys] if isinstance(keys, str) else list(keys)

                if not k_list:
                    raise ValueError(
                        f"No keys selected for {dataset}/{motion}/{trajectory}."
                    )

                T = 0
                start_indices.append(written)
                trajectory_list.append((dataset, motion, trajectory))
                data_dict: dict[str, torch.Tensor] = {}
                for k in k_list:
                    logger.debug(
                        f"Loading key '{k}' from {dataset}/{motion}/{trajectory}..."
                    )
                    if k not in traj_grp:
                        raise KeyError(
                            f"Key '{k}' not found in {dataset}/{motion}/{trajectory}. "
                            f"Available: {list(traj_grp.array_keys())}"
                        )
                    if not _is_transition_aligned_key(k):
                        np_data = traj_grp[k][
                            :-1
                        ]  # discard the last step for non-transition data
                    else:
                        np_data = traj_grp[k][:]  # load all (T, ...) into memory
                    data_dict[k] = _zarr_array_to_torch(np_data, device=device_t)
                    T = data_dict[k].shape[0]

                traj_td = TensorDict(data_dict, batch_size=[T], device=device_t)

                # Append into memmap-backed RB
                rb.extend(traj_td)
                written += T
                end_indices.append(written)

    if written != capacity:
        # This can happen if attrs-based capacity computed differently than selection.
        # ReplayBuffer will still contain `written` transitions; capacity is just max size.
        logging.warning(
            f"[make_rb_from] Note: capacity={capacity} but written={written}. "
            "This is OK, but you may want to adjust capacity computation if you "
            "need them to match exactly."
        )

    traj_info = {
        "capacity": capacity,
        "written": written,
        "start_index": start_indices,
        "end_index": end_indices,
        "ordered_traj_list": trajectory_list,
    }

    if persist_path is not None:
        # Written last, so a manifest only ever exists next to a complete fill:
        # a job killed mid-fill leaves no manifest and the next run rebuilds.
        manifest = {
            "format_version": _PERSIST_FORMAT_VERSION,
            "key": persist_identity,
            "traj_info": {
                **traj_info,
                "ordered_traj_list": [list(entry) for entry in trajectory_list],
            },
        }
        tmp_path = persist_path / f"{_PERSIST_MANIFEST_NAME}.tmp"
        tmp_path.write_text(json.dumps(manifest))
        tmp_path.replace(persist_path / _PERSIST_MANIFEST_NAME)
        logger.info(
            "Persisted replay buffer to %s (%s transitions, %s trajectories).",
            persist_path,
            written,
            len(trajectory_list),
        )

        if requested_device != fill_device:
            # Caller asked for a GPU-resident buffer. Re-enter through the
            # cached reader so a fresh build and a later reuse return exactly
            # the same object, rather than two subtly different code paths.
            del rb
            reopened = _load_persisted_rb(
                persist_path,
                persist_identity,
                device=requested_device,
                pin_memory=pin_memory,
                prefetch=prefetch,
                batch_size=batch_size,
                compilable=compilable,
            )
            if reopened is None:
                raise RuntimeError(
                    f"Persisted buffer at {persist_path} could not be reopened "
                    "immediately after being written."
                )
            return reopened

    return rb, traj_info


def make_td_from(
    key: str,
    data_array: torch.Tensor | np.ndarray,
    *,
    device: str | torch.device = "cpu",
) -> TensorDict:
    """Convenience helper to build a TensorDict with a single entry.

    Args:
        key: tensordict key.
        data_array: array-like shaped [T, ...].
        device: where to put the tensor.

    Returns:
        TensorDict with batch_size=[T].
    """
    device_t = torch.device(device)
    t = (
        data_array
        if isinstance(data_array, torch.Tensor)
        else torch.from_numpy(data_array)
    ).to(device=device_t)
    return TensorDict({key: t}, batch_size=[int(t.shape[0])], device=device_t)


def get_traj_rank_from_info(
    dataset: str,
    motion: str,
    trajectory: str,
    ordered_traj_list: list[tuple[str, str, str]],
) -> int:
    """Compute the rank of a trajectory in the ordered list."""
    if (dataset, motion, trajectory) not in ordered_traj_list:
        raise ValueError(
            f"Trajectory {dataset}/{motion}/{trajectory} not found in the ordered list {ordered_traj_list}."
        )
    return ordered_traj_list.index((dataset, motion, trajectory))


def get_traj_rank_from_global_index(
    global_index: int,
    start_indices: list[int],
    end_indices: list[int],
) -> int:
    """Compute the rank of a trajectory in the ordered list from a global index.

    Args:
        global_index: the global index of the trajectory. Ranged from 0 to capacity - 1.
        start_indices: the start indices of the trajectories.
        end_indices: the end indices of the trajectories.
        ordered_traj_list: the ordered list of trajectories.

    Returns:
        the rank of the trajectory in the ordered list.
    """

    # Get the mask of the trajectory that the global index belongs to
    mask = (global_index >= start_indices) & (global_index < end_indices)
    return mask.index(True)


def get_ith_traj_info(
    traj_rank: int, ordered_traj_list: tuple[str, str, str]
) -> tuple[str, str, str]:
    """Get the info of the i-th trajectory in the ordered list."""
    return ordered_traj_list[traj_rank]


def _map_reference_to_target(
    reference_joint_names: Sequence[str],
    target_joint_names: Sequence[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Map the reference joint names to the target joint names; and return the target tensor and the mapping as a list of indices, so that the target tensor can be indexed by the indices, e.g., tensor[inv_map] = reference_tensor[map] produces a tensor with the same shape as the target tensor but the values are the same as the reference tensor re-ordered. Also, tensor[~inv_map] = NaN.

    Args:
        reference_joint_names: List of reference joint names
        target_joint_names: List of target joint names

    Returns:
        Tuple containing:
            - mapping: List of indices for mapping
            - inv_map: List of indices for inverse mapping
    """
    # Create mapping from reference to target joint positions
    mapping: list[int] = []
    inv_map: list[int] = []
    all_joint_names = list(set(target_joint_names + reference_joint_names))
    for joint_name in all_joint_names:
        if (
            joint_name not in target_joint_names
            or joint_name not in reference_joint_names
        ):
            continue
        map_idx = target_joint_names.index(joint_name)
        mapping.append(map_idx)
        inv_map_idx = reference_joint_names.index(joint_name)
        inv_map.append(inv_map_idx)

    return torch.tensor(mapping, device=device), torch.tensor(inv_map, device=device)
