# Trajopt data loader

Memory efficient memmap dataset management from trajectory optimizaiton generated data

## Design idea

Goal: stream fixed-shape windows of trajectory data to jitted training loops without loading whole episodes into device memory.

Key pieces
- Trajopt → Zarr export: `TrajoptLoader.save(...)` writes a vectorized layout that `VectorizedTrajectoryDataset` can read lazily.
- Vectorized window fetch: `VectorizedTrajectoryDataset.fetch_window(...)` returns `[num_envs, window_size, ...]` slices using small sliding host buffers.
- Double buffering to device: `DoubleBufferStreamer` overlaps host prefetch + device_put with device compute.
- Digit references: `digit_refs.build_refs_from_window(...)` derives Digit-specific ref arrays (joint targets, root pose/vel, EE) from raw `qpos/qvel` windows (no per-episode padding).

Why this minimizes memory
- Host keeps only small per-env sliding windows (buffer_size).
- Device holds only the current double-buffered window for compute.
- No Python/Zarr I/O inside jit; shapes are fixed for XLA.

Data layout (Zarr)
```
<zarr_root>/<dataset_source>/<motion>/traj{i}/{qpos,qvel,action[,ee_pos]}
```
This mirrors the tests (see `tests/datasets/test_storage_vectorized.py`).

## Typical working pipeline

1) Export trajopt `.npz` → Zarr
```python
from iltools_datasets.trajopt.loader import TrajoptLoader
zarr_path = TrajoptLoader("/path/to/npz").save(
    out_dir="/tmp/zarr", dataset_source="trajopt", motion="digit_v3"
)
```

2) Create a vectorized dataset with small window/buffer
```python
from omegaconf import OmegaConf
from iltools_datasets.storage import VectorizedTrajectoryDataset

num_envs = 64
window  = 32
cfg = OmegaConf.create({"window_size": window, "buffer_size": 128, "allow_wrap": True})
ds = VectorizedTrajectoryDataset(zarr_path=zarr_path, num_envs=num_envs, cfg=cfg)
ds.update_references(
    env_to_traj={e: e % 8 for e in range(num_envs)},
    env_to_step={e: 0     for e in range(num_envs)},
)
```

3) Double-buffer windows to device (JAX)
```python
from iltools_datasets.jax_stream import DoubleBufferStreamer

def host_fetch_window():
    batch = {
        "qpos": ds.fetch_window(range(num_envs), key="qpos", window_size=window),
        "qvel": ds.fetch_window(range(num_envs), key="qvel", window_size=window),
    }
    # advance steps by window for next chunk
    ds.update_references(env_to_step={e: ds.env_steps[e] + window for e in range(num_envs)})
    return batch

streamer = DoubleBufferStreamer(window_fn=host_fetch_window)
streamer.start()
# device batch with fixed shape
batch = streamer.next_window()
```

4) Build Digit references from the window (optional, when you need ref targets)
```python
from iltools_datasets.trajopt.digit_refs import DigitRefSpec, build_refs_from_window
import numpy as np

spec = DigitRefSpec(
    a_pos_index=np.array([7,8,9,14,18,23,30,31,32,33,34,35,36,41,45,50,57,58,59,60,61,62,63,64], dtype=np.int32),
    a_vel_index=np.array([6,7,8,12,16,20,26,27,28,29,30,31,32,36,40,44,50,51,52,53,54,55,56,57], dtype=np.int32),
    subsample=1,
)

refs = [build_refs_from_window(batch["qpos"][e], batch["qvel"][e], spec) for e in range(num_envs)]
# Or stack per key to shape [E, T, ...] if preferred
```

5) Jitted training over the window
```python
import jax, jax.numpy as jp

def body(carry, inputs):
    qpos_t, qvel_t = inputs
    return carry + jp.sum(qpos_t * 0.0 + qvel_t * 0.0), 0.0

@jax.jit
def train_step(batch):
    qpos = jp.moveaxis(batch["qpos"], 1, 0)  # [T,E,D]
    qvel = jp.moveaxis(batch["qvel"], 1, 0)
    _, _ = jax.lax.scan(body, 0.0, (qpos, qvel))
    return {}
```

Integration with the Digit environment (mujoco_playground)
- In `triarm_eetrack.py`, the env reads `cfg.zarr_path`. If provided, it attaches the vectorized dataset and uses a minimal window on `reset` to build per-episode references.
- You can point both `cfg.ref_path` (npz root for fallback) and `cfg.zarr_path` (preferred) in tests or training.

See also
- Streaming tests: `tests/datasets/test_digit_stream_jit.py`
- End-to-end training-like test: `tests/training/test_digit_training_streaming.py`
- Vectorized dataset API: `src/iltools_datasets/storage.py`
