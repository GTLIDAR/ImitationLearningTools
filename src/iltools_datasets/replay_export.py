"""Utilities to build replay buffers from Zarr datasets.

This module provides a simple path to export a Zarr trajectory dataset into a
TorchRL-based replay buffer using the memmap-backed storage defined in
`replay_memmap.py` and the high-level manager in `replay_manager.py`.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
import zarr
from tensordict import TensorDict

from .replay_manager import ExpertReplayManager, ExpertReplaySpec
from .replay_memmap import ExpertMemmapBuilder, build_trajectory_td


def _iter_trajectories(zroot: zarr.Group) -> Iterable[tuple[int, int, zarr.Group]]:
    """Yield (task_id, traj_id, traj_group) for all trajectories in nested layout.

    task_id increments per (dataset_source, motion) group, traj_id increments per
    trajectory within that motion.
    """
    task_id = 0
    for dataset_source in zroot:
        ds_group = zroot[dataset_source]
        if not isinstance(ds_group, zarr.Group):
            continue
        for motion in ds_group:
            motion_group = ds_group[motion]
            if not isinstance(motion_group, zarr.Group):
                continue
            traj_id = 0
            for traj in motion_group:
                traj_group = motion_group[traj]
                if isinstance(traj_group, zarr.Group):
                    yield task_id, traj_id, traj_group
                    traj_id += 1
            task_id += 1


def build_replay_from_zarr(
    *,
    zarr_path: str,
    scratch_dir: str,
    obs_keys: Sequence[str] | None = None,
    act_key: str | None = None,
    device: str = "cpu",
    strict: bool = True,
    concat_obs_to_key: str | None = "observation",
    original_obs_td_key: str = "lmj_observation",
    include_terminated: bool = True,
    include_truncated: bool = True,
    infer_terminated_if_missing: bool = True,
    infer_truncated_if_missing: bool = False,
    terminated_key_candidates: Sequence[str] = ("terminated", "done"),
    truncated_key_candidates: Sequence[str] = ("truncated", "absorbing"),
) -> ExpertReplayManager:
    """Construct an ExpertReplayManager from a Zarr dataset.

    - Validates that all `obs_keys` are present in each trajectory when `strict=True`.
    - Builds transitions by truncating to T-1: uses step `t` for obs and `t+1` for next_obs.
    - If `concat_obs_to_key` is provided, concatenates all `obs_keys` along the last
      dimension into that key (default 'observation'), in addition to storing each
      `obs_key` separately at the top level and under ('next', key).
    - If the Zarr contains the original observation under key 'obs', it is exported
      under `original_obs_td_key` (and ('next', original_obs_td_key)).
    - `act_key`: if provided, must be present; otherwise actions default to zeros.

    The resulting replay TensorDicts contain:
      - the concatenated observation (if `concat_obs_to_key`),
      - each component in `obs_keys`,
      - optional `original_obs_td_key` if found,
      - 'action', and ('next', ...) counterparts for observation keys.
    """
    root = zarr.open_group(zarr_path, mode="r")
    assert isinstance(root, zarr.Group)

    if obs_keys is None:
        obs_keys = ("qpos",)

    # First pass: compute total transitions (sum of (T-1) across all trajs)
    total_T = 0
    for _, _, traj_group in _iter_trajectories(root):
        # Ensure mandatory obs keys are present
        if strict:
            for k in obs_keys:
                if k not in traj_group:
                    raise KeyError(f"Trajectory group missing required obs key '{k}'")
        # Use the first available obs key to determine T
        first_key = next((k for k in obs_keys if k in traj_group), None)
        if first_key is None:
            continue
        T = int(np.asarray(traj_group[first_key]).shape[0])
        if T >= 2:
            total_T += T - 1

    builder = ExpertMemmapBuilder(
        scratch_dir=scratch_dir, max_size=total_T, device="cpu"
    )

    # Build per-task trajectories as TensorDicts and append to builder
    tasks: dict[int, list[TensorDict]] = {}
    for task_id, traj_id, traj_group in _iter_trajectories(root):
        # Determine T from the first present obs key
        first_key = next((k for k in obs_keys if k in traj_group), None)
        if first_key is None:
            if strict:
                missing = ", ".join(obs_keys)
                raise KeyError(
                    f"Trajectory group has none of the required obs keys: {missing}"
                )
            else:
                continue
        T = int(np.asarray(traj_group[first_key]).shape[0])
        if T < 2:
            continue
        n_this = T - 1

        # Prepare per-key observations
        obs_per_key = {}
        next_obs_per_key = {}
        for k in obs_keys:
            if k not in traj_group:
                if strict:
                    raise KeyError(f"Trajectory group missing required obs key '{k}'")
                else:
                    continue
            arr = np.asarray(traj_group[k])
            obs_per_key[k] = torch.from_numpy(arr[:-1]).to(torch.float32)
            next_obs_per_key[k] = torch.from_numpy(arr[1:]).to(torch.float32)

        # Concatenate observation if requested
        if concat_obs_to_key is not None:
            # Concatenate in the given obs_keys order
            obs_concat = torch.cat(
                [obs_per_key[k] for k in obs_keys if k in obs_per_key], dim=-1
            )
            next_obs_concat = torch.cat(
                [next_obs_per_key[k] for k in obs_keys if k in next_obs_per_key], dim=-1
            )
        else:
            # Use the first key as the main observation if not concatenating
            obs_concat = obs_per_key[first_key]
            next_obs_concat = next_obs_per_key[first_key]

        # Actions
        # Auto-detect action key if not provided
        if act_key is None:
            if "action" in traj_group:
                act_key_eff = "action"
            elif "actions" in traj_group:
                act_key_eff = "actions"
            else:
                act_key_eff = None
        else:
            act_key_eff = act_key

        if act_key_eff is not None:
            if act_key_eff not in traj_group:
                raise KeyError(
                    f"Trajectory group missing required action key '{act_key_eff}'"
                )
            act_np = np.asarray(traj_group[act_key_eff])
            act = torch.from_numpy(act_np[:-1]).to(torch.float32)
        else:
            act = torch.zeros((T - 1, 1), dtype=torch.float32)

        # Base TensorDict with main observation/action
        td = build_trajectory_td(
            observation=obs_concat, action=act, next_observation=next_obs_concat
        )

        # Attach each observation component and its next counterpart
        for k in obs_per_key:
            td.set(k, obs_per_key[k])
            td.set(("next", k), next_obs_per_key[k])

        # Optional original observation from dataset if present ('obs' from Loco-MuJoCo export)
        if "obs" in traj_group:
            orig_np = np.asarray(traj_group["obs"])
            orig = torch.from_numpy(orig_np[:-1]).to(torch.float32)
            next_orig = torch.from_numpy(orig_np[1:]).to(torch.float32)
            td.set(original_obs_td_key, orig)
            td.set(("next", original_obs_td_key), next_orig)

        # Optional terminated / truncated flags (TorchRL convention)
        if include_terminated or include_truncated:
            term_tensor = None
            trunc_tensor = None

            if include_terminated:
                # Prefer existing 'terminated', fallback to 'done'
                term_key = next(
                    (k for k in terminated_key_candidates if k in traj_group), None
                )
                if term_key is not None:
                    arr = np.asarray(traj_group[term_key])
                    term_tensor = torch.from_numpy(arr[:n_this].astype(np.bool_))
                elif infer_terminated_if_missing:
                    term_tensor = torch.zeros((n_this,), dtype=torch.bool)
                    if n_this > 0:
                        term_tensor[-1] = True

            if include_truncated:
                # Prefer existing 'truncated', fallback to 'absorbing' if present, else infer all False
                trunc_key = next(
                    (k for k in truncated_key_candidates if k in traj_group), None
                )
                if trunc_key is not None:
                    arr = np.asarray(traj_group[trunc_key])
                    trunc_tensor = torch.from_numpy(arr[:n_this].astype(np.bool_))
                elif infer_truncated_if_missing:
                    trunc_tensor = torch.zeros((n_this,), dtype=torch.bool)
                else:
                    trunc_tensor = torch.zeros((n_this,), dtype=torch.bool)

            if term_tensor is not None:
                td.set("terminated", term_tensor)
            if trunc_tensor is not None:
                td.set("truncated", trunc_tensor)

        # ------------------------------------------------------------------
        # Attach all remaining per-state arrays as (k, ('next', k)) pairs.
        #
        # Any Zarr dataset under this trajectory group with leading dimension
        # equal to `T` and not already handled above is treated as a per-state
        # signal. We export:
        #
        #   td[k]           = value[:-1]
        #   td[('next', k)] = value[1:]
        #
        # Arrays with different leading lengths (e.g. per-transition flags
        # that are already handled via `terminated`/`truncated`) are skipped.
        # ------------------------------------------------------------------
        special_keys = set(obs_keys) | {
            act_key_eff,
            "terminated",
            "done",
            "truncated",
            "absorbing",
        }

        for k in traj_group.keys():
            if k in special_keys:
                continue
            node = traj_group[k]
            # Skip non-array leaves (e.g. groups or scalars)
            if not hasattr(node, "shape"):
                continue
            arr = np.asarray(node)
            # Only treat arrays with leading dimension T as per-state
            if arr.ndim == 0 or arr.shape[0] != T:
                continue
            curr_np = arr[:-1]
            next_np = arr[1:]

            if arr.dtype == np.bool_ or np.issubdtype(arr.dtype, np.bool_):
                curr = torch.from_numpy(curr_np.astype(np.bool_))
                nxt = torch.from_numpy(next_np.astype(np.bool_))
            else:
                curr = torch.from_numpy(curr_np).to(torch.float32)
                nxt = torch.from_numpy(next_np).to(torch.float32)

            # Avoid accidentally overwriting anything we have already set
            if k not in td.keys():
                td.set(k, curr)
            if ("next", k) not in td.keys(True):
                td.set(("next", k), nxt)
        print(f"[build_replay_from_zarr] td: {td}")
        # Append to builder and record in tasks
        builder.add_trajectory(task_id=task_id, traj_id=traj_id, transitions=td)
        tasks.setdefault(task_id, []).append(td)

    storage, segments = builder.finalize()
    spec = ExpertReplaySpec(
        tasks=tasks, scratch_dir=scratch_dir, device=device, sample_batch_size=256
    )
    return ExpertReplayManager(spec)
