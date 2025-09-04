# iltools_datasets

Utilities for loading, caching and sampling imitation-learning trajectories.

This module provides:

- Vectorized, windowed access to per-trajectory Zarr datasets (`storage.py`).
- Disk-backed replay storage using TorchRL memmap (`replay_memmap.py`).
- A high-level replay manager with per-env sequential sampling (`replay_manager.py`).

## VectorizedTrajectoryDataset (`storage.py`)

`VectorizedTrajectoryDataset` reads a hierarchical Zarr dataset organized as:

```
<zarr_root>/
  <dataset_source>/
    <motion>/
      <trajectory>/
        qpos: [T, qpos_dim]
        qvel: [T, qvel_dim]  (optional)
        ... other keys
```

Key features:

- Vectorized fetch for multiple envs with per-env trajectory/step pointers.
- Small sliding-window buffers per env to reduce random I/O (`buffer_size`).
- Introspects available dataset sources, motions, and trajectories.

Basic usage:

```python
from omegaconf import OmegaConf
from iltools_datasets.storage import VectorizedTrajectoryDataset

cfg = OmegaConf.create({
    "window_size": 32,   # consumer concept (not used internally)
    "buffer_size": 128,  # per-env sliding I/O window
    "allow_wrap": False, # wrap steps modulo traj length instead of raising
})
ds = VectorizedTrajectoryDataset(zarr_path=".../trajectories.zarr", num_envs=16, cfg=cfg)

# Assign envs to trajectories and set current steps
ds.update_references(env_to_traj={i: i % len(ds.available_trajectories) for i in range(16)},
                     env_to_step={i: 0 for i in range(16)})

# Fetch a key for selected envs at their current steps
qpos = ds.fetch([0, 5, 7], key="qpos")  # shape: [3, qpos_dim]
```

Notes:
- `window_size` is intentionally not used by the dataset for single-step fetches;
  use it in downstream logic that constructs temporal windows.
- Set `allow_wrap=True` to prevent IndexError when a step falls outside the
  trajectory length; the step will wrap modulo the length.

## Replay Memmap (`replay_memmap.py`)

These helpers build a disk-backed storage of transitions with O(1) indexing.

- `Segment(task_id, traj_id, start, length)`: maps local time `t` to global index `start + t`.
- `ExpertMemmapBuilder(scratch_dir, max_size)`: append trajectories and obtain:
  - `LazyMemmapStorage` (TorchRL) with all transitions
  - `list[Segment]` describing contiguous ranges per trajectory
- `build_trajectory_td(...)`: pack `(observation, action, next_observation)` into a `TensorDict` with batch shape `[T]`.
- `build_trajectory_td_from_components(...)`: concat observation parts before packing.

Example:

```python
from tensordict import TensorDict
import torch
from iltools_datasets.replay_memmap import ExpertMemmapBuilder, build_trajectory_td

obs = torch.randn(100, 64)
act = torch.randn(100, 12)
next_obs = torch.randn(100, 64)
td = build_trajectory_td(observation=obs, action=act, next_observation=next_obs)

builder = ExpertMemmapBuilder("/tmp/iltools_memmap", max_size=100)
seg = builder.add_trajectory(task_id=0, traj_id=0, transitions=td)
storage, segments = builder.finalize()
```

## Expert Replay Manager (`replay_manager.py`)

High-level manager that constructs a TorchRL `TensorDictReplayBuffer` on top of memmap
storage and exposes flexible sampling strategies.

- `ExpertReplaySpec`: describes tasks `{task_id: [trajectory_td, ...]}`, scratch dir,
  and default batch size.
- `ExpertReplayManager(spec)`: builds memmap storage and replay buffer.
- `EnvAssignment(task_id, traj_id, step)`: assigns each env to a trajectory and maintains its local time.
- `SequentialPerEnvSampler`: returns one index per env each sample call, advancing per-env pointers.

Common flows:

1) Uniform minibatch sampling (IPMD training):

```python
mgr.set_uniform_sampler(batch_size=256, without_replacement=True)
batch = mgr.buffer.sample()  # shape: [256, ...]
```

2) Sequential per-env sampling (DeepMimic-style marching through clips):

```python
from iltools_datasets.replay_manager import EnvAssignment

assign = [EnvAssignment(task_id=0, traj_id=0, step=0) for _ in range(num_envs)]
mgr.set_assignment(assign)

step_batch = mgr.buffer.sample()  # shape: [num_envs, ...]
```

3) Device transforms for overlap of H2D copies with compute:

```python
mgr.set_device_transform(device=torch.device("cuda"))
```

Notes:
- Storage lives on CPU via memmap; transforms move sampled batches as needed.
- Re-creating the buffer when switching samplers reuses the same underlying storage.

## Zarr → Replay Pipeline (`replay_export.py`)

Quickly build a replay buffer from a Zarr dataset:

```python
from iltools_datasets.replay_export import build_replay_from_zarr

mgr = build_replay_from_zarr(
    zarr_path=".../trajectories.zarr",
    scratch_dir="/tmp/iltools_memmap",
    obs_keys=["qpos", "qvel"],       # required observation keys
    act_key="action",                # required for consistent transitions
    concat_obs_to_key="observation", # concatenated view; optional
)

batch = mgr.buffer.sample()
# Access multiple views
batch["observation"]          # concat([qpos, qvel])
batch["qpos"], batch["qvel"]  # individual components
batch["lmj_observation"]      # original env observation if present in Zarr (obs/observation)
batch["terminated"], batch["truncated"]  # TorchRL termination flags (bool)
```

Parameters:
- obs_keys: required observation keys to export (raises if missing when strict=True).
- act_key: action key; if omitted, actions default to zeros.
- concat_obs_to_key: name for concatenated observation view (set to None to disable).
- Original observation: if the Zarr export includes 'obs' (as produced by Loco‑MuJoCo
  transition export), it is included as `lmj_observation` with its next counterpart.
- include_terminated/include_truncated: attach `terminated` and `truncated` flags (TorchRL convention).
  If `terminated` is not present, it falls back to `done` if available or infers a simple last-step True flag.
  If `truncated` is not present, it falls back to `absorbing` if available or defaults to all False.
