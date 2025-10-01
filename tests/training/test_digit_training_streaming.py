import os
from typing import Tuple

import jax
import jax.numpy as jp
import numpy as np
from omegaconf import OmegaConf

from iltools_datasets.trajopt.loader import TrajoptLoader
from iltools_datasets.storage import VectorizedTrajectoryDataset
from iltools_datasets.jax_stream import DoubleBufferStreamer


def _write_trajopt_npz(root_dir: str) -> Tuple[str, list[int]]:
    os.makedirs(root_dir, exist_ok=True)
    lengths = [128, 128]
    qpos_dim, qvel_dim = 8, 6
    for i, L in enumerate(lengths):
        qpos = np.random.randn(L, qpos_dim).astype(np.float32)
        qvel = np.random.randn(L, qvel_dim).astype(np.float32)
        np.savez(os.path.join(root_dir, f"traj_{i}.npz"), qpos=qpos, qvel=qvel)
    return root_dir, lengths


def _init_mlp(rng, in_dim: int, hidden: int, out_dim: int):
    k1, k2 = jax.random.split(rng, 2)
    W1 = jax.random.normal(k1, (in_dim, hidden)) * (1.0 / np.sqrt(in_dim))
    b1 = jp.zeros((hidden,))
    W2 = jax.random.normal(k2, (hidden, out_dim)) * (1.0 / np.sqrt(hidden))
    b2 = jp.zeros((out_dim,))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def _mlp_forward(params, x):
    h = jp.maximum(0.0, x @ params["W1"] + params["b1"])  # ReLU
    y = h @ params["W2"] + params["b2"]
    return y


def test_digit_like_training_streaming(tmp_path):
    # 1) Build Zarr from trajopt npz
    npz_dir, _ = _write_trajopt_npz(os.path.join(tmp_path, "npz"))
    zarr_path = TrajoptLoader(npz_dir).save(
        out_dir=str(tmp_path), dataset_source="trajopt", motion="digit_v3"
    )

    # 2) Vectorized dataset setup (Digit-like, vectorized envs)
    num_envs = 8
    window = 32
    cfg = OmegaConf.create(
        {"window_size": window, "buffer_size": 64, "allow_wrap": True}
    )
    ds = VectorizedTrajectoryDataset(zarr_path=zarr_path, num_envs=num_envs, cfg=cfg)
    ds.update_references(
        env_to_traj={e: e % 2 for e in range(num_envs)},
        env_to_step={e: 0 for e in range(num_envs)},
    )

    print(ds.env_traj_ids)
    print(ds.env_steps)
    print(ds.window_starts)
    print(ds.buffers)
    print(ds.handlers)
    print(ds.traj_lengths)
    print(ds.all_keys)
    print(ds.available_dataset_sources)
    print(ds.available_motions)
    print(ds.available_trajectories)
    print(ds.available_trajectories_in("trajopt", "digit_v3"))
    print(ds.available_motions_in("trajopt"))
    print(ds.available_dataset_sources)
    print(ds.zarr_dataset)
    print(ds.zarr_dataset["trajopt/digit_v3/traj0/qpos"])
    print(ds.zarr_dataset["trajopt/digit_v3/traj0/qvel"])

    # 3) Host window provider + double buffering
    def host_fetch_window():
        qpos = ds.fetch_window(
            idx=list(range(num_envs)), key="qpos", window_size=window
        )
        qvel = ds.fetch_window(
            idx=list(range(num_envs)), key="qvel", window_size=window
        )
        # Advance by a chunk (scan over fixed number of steps)
        ds.update_references(
            env_to_step={e: ds.env_steps[e] + window for e in range(num_envs)}
        )
        # Device transfer happens in streamer
        return {"qpos": qpos, "qvel": qvel}

    streamer = DoubleBufferStreamer(window_fn=host_fetch_window)
    streamer.start()

    # 4) Define a tiny policy/value model and a jitted train step over the window
    rng = jax.random.key(0)
    params = _init_mlp(rng, in_dim=14, hidden=32, out_dim=6)
    lr = 1e-2

    def loss_fn(params, batch):
        # batch: {qpos:[E,T,8], qvel:[E,T,6]} → flatten over env/time
        qpos = batch["qpos"]
        qvel = batch["qvel"]
        E, T, Dp = qpos.shape
        Dv = qvel.shape[-1]
        x = jp.concatenate([qpos.reshape(E * T, Dp), qvel.reshape(E * T, Dv)], axis=-1)
        y = _mlp_forward(params, x)  # [E*T, 6]
        # Dummy target zero: encourage stable outputs
        return jp.mean(y * y)

    @jax.jit
    def train_step(params, batch):
        val, grads = jax.value_and_grad(loss_fn)(params, batch)
        params = {k: params[k] - lr * grads[k] for k in params}
        return params, val

    # 5) Consume several windows and update parameters
    losses = []
    for _ in range(4):
        batch = streamer.next_window()
        params, l = train_step(params, batch)
        losses.append(float(l))

    streamer.stop()

    # 6) Sanity: loss should not explode and ideally decreases
    assert np.isfinite(losses[0]) and np.isfinite(losses[-1])
    assert (
        losses[-1] <= losses[0] * 1.05
    )  # allow small noise; should not worsen significantly
