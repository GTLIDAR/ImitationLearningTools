# iltools_datasets

Practical utilities for loading, caching, and sampling expert trajectories for imitation learning. It focuses on fast, memory-efficient access patterns that play nicely with vectorized RL training loops.

What you get:
- Offline export from raw motion libraries (e.g. Loco-MuJoCo) to a structured Zarr trajectory store.
- Disk-backed replay via TorchRL memmap (`replay_memmap.py` + `replay_export.py`).
- A high-level replay manager with sequential, windowed, and uniform sampling modes (`replay_manager.py`).

## Quick Start

Build a replay from a Zarr dataset and draw batches:

```python
from iltools_datasets.replay_export import build_replay_from_zarr

mgr = build_replay_from_zarr(
    zarr_path=".../trajectories.zarr",
    scratch_dir="/tmp/iltools_memmap",
    obs_keys=["qpos", "qvel"],
    act_key="action",
    concat_obs_to_key="observation",
)

# Global uniform minibatch (without replacement)
mgr.set_uniform_sampler(batch_size=256, without_replacement=True)
batch = mgr.buffer.sample()
```

Sequential per-env sampling (DeepMimic-style marching through clips):

```python
from iltools_datasets.replay_manager import EnvAssignment

assign = [EnvAssignment(task_id=0, traj_id=0, step=i) for i in range(num_envs)]
mgr.set_assignment(assign)
step_batch = mgr.buffer.sample()  # shape: [num_envs, ...]
```

Uniform minibatch restricted to the currently assigned (task,traj) segments:

```python
mgr.set_uniform_sampler(batch_size=512, without_replacement=True, respect_assignment=True)
batch = mgr.buffer.sample()
```

Update assignments on reset (no full list rebuild):

```python
# Single env
mgr.update_env_assignment(env_index=3, task_id=2, traj_id=5, step=0)

# Bulk update: env_index -> (task_id, traj_id[, step])
mgr.update_assignments({0: (1, 2), 7: (2, 0, 10)})
```

Move sampled batches to device:

```python
import torch
batch = mgr.buffer.sample()
batch = batch.to(torch.device("cuda"))  # Buffer storage handles device transforms automatically
```

## Components

– Replay Memmap (`replay_memmap.py`)
- `ExpertMemmapBuilder`: appends trajectories into a single memmap (`LazyMemmapStorage`).
- `Segment(task_id, traj_id, start, length)`: O(1) map from local step to global index.
- `build_trajectory_td(...)`: packs `(observation, action, next_observation)` into a `TensorDict[T]`.

Example:
```python
import torch
from iltools_datasets.replay_memmap import ExpertMemmapBuilder, build_trajectory_td

obs, act, nxt = torch.randn(100, 64), torch.randn(100, 12), torch.randn(100, 64)
td = build_trajectory_td(observation=obs, action=act, next_observation=nxt)
b = ExpertMemmapBuilder("/tmp/iltools_memmap", max_size=100)
seg = b.add_trajectory(task_id=0, traj_id=0, transitions=td)
storage, segments = b.finalize()
```

– Expert Replay Manager (`replay_manager.py`)
- Wraps memmap storage in a `TensorDictReplayBuffer`.
- Sampling modes:
  - Sequential per-env: `set_assignment([...])` → returns one index per env and advances steps.
  - Global uniform minibatch: `set_uniform_sampler(batch_size, without_replacement=True)`.
  - Assigned-uniform minibatch: `set_uniform_sampler(..., respect_assignment=True)` limits sampling to the assigned segments.
- Convenient updates:
  - `update_env_assignment(i, task_id, traj_id, step=0)`
  - `update_assignments({env: (task, traj[, step]), ...})`
- Device handling: Buffer storage automatically manages device transforms

## Zarr → Replay in One Call

`replay_export.py` provides `build_replay_from_zarr(...)` to go from Zarr to a ready-to-sample manager. It exports all per-state arrays in each trajectory group (e.g. `qpos`, `qvel`, kinematic features, etc.) along with concatenated observations, actions, and optional termination flags when present.

Parameters (high-level):
- `obs_keys`: observation components to export (required).
- `act_key`: action key (defaults to zeros if missing).
- `concat_obs_to_key`: name for concatenated observation view.
- `include_terminated` / `include_truncated`: attach TorchRL flags when available.

## Tips
- Storage lives on CPU; buffer storage automatically handles device transforms when sampling.
- Switching samplers reuses the same memmap storage; transforms are preserved.
- For assigned-uniform without replacement, the sampler reshuffles once its internal epoch is exhausted or when assignments change.

## Minimal API Reference

- `Segment`: `(task_id, traj_id, start, length)`, `index_at(t, wrap=True)`
- `ExpertMemmapBuilder`: `add_trajectory(...)`, `finalize()`
- `build_trajectory_td(...)`, `build_trajectory_td_from_components(...)`
- `ExpertReplayManager`:
  - `set_assignment(assign: list[EnvAssignment])`
  - `update_env_assignment(env_index, task_id, traj_id, step=0)`
  - `update_assignments({env: (task, traj[, step])})`
  - `set_uniform_sampler(batch_size, without_replacement=True, respect_assignment=True)`
  - Device transforms are handled automatically by buffer storage
  - `clear_assignment()`

That’s it — compact building blocks to get expert data streaming reliably into your training loop.
