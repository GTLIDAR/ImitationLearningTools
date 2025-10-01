import os
import time

import jax
import jax.numpy as jp
import numpy as np
from omegaconf import OmegaConf

from iltools_datasets.trajopt.loader import TrajoptLoader
from iltools_datasets.storage import VectorizedTrajectoryDataset
from iltools_datasets.jax_stream import DoubleBufferStreamer


def _write_trajopt_npz(root_dir: str):
    os.makedirs(root_dir, exist_ok=True)
    lengths = [64, 80]
    qpos_dim, qvel_dim = 8, 6
    for i, L in enumerate(lengths):
        qpos = np.random.randn(L, qpos_dim).astype(np.float32)
        qvel = np.random.randn(L, qvel_dim).astype(np.float32)
        np.savez(os.path.join(root_dir, f"traj_{i}.npz"), qpos=qpos, qvel=qvel)


def test_streamed_windows_with_jit(tmp_path):
    # 1) Build Zarr from trajopt npz
    npz_dir = os.path.join(tmp_path, "npz")
    _write_trajopt_npz(npz_dir)
    loader = TrajoptLoader(npz_dir)
    zarr_path = loader.save(
        out_dir=str(tmp_path), dataset_source="trajopt", motion="digit_v3"
    )

    # 2) Vectorized dataset setup
    num_envs = 4
    window = 16
    cfg = OmegaConf.create(
        {"window_size": window, "buffer_size": 32, "allow_wrap": True}
    )
    ds = VectorizedTrajectoryDataset(zarr_path=zarr_path, num_envs=num_envs, cfg=cfg)
    # Assign envs to trajectories and starting steps
    ds.update_references(
        env_to_traj={e: e % 2 for e in range(num_envs)},
        env_to_step={e: 0 for e in range(num_envs)},
    )

    # 3) Window provider: host fetches small fixed windows only
    def host_fetch_window():
        qpos = ds.fetch_window(
            idx=list(range(num_envs)), key="qpos", window_size=window
        )
        qvel = ds.fetch_window(
            idx=list(range(num_envs)), key="qvel", window_size=window
        )
        # Advance env steps by window (simulate scan chunk)
        ds.update_references(
            env_to_step={e: ds.env_steps[e] + window for e in range(num_envs)}
        )
        return {"qpos": qpos, "qvel": qvel}

    streamer = DoubleBufferStreamer(window_fn=host_fetch_window)
    streamer.start()

    # 4) Jitted scan over the window: pure compute
    def scan_body(carry, inputs):
        qpos_t, qvel_t = inputs
        # dummy compute; ensure shapes are static
        out = jp.sum(qpos_t * 0.0 + qvel_t * 0.0)
        return carry + out, out

    @jax.jit
    def consume_window(batch):
        # batch["qpos"]: [E, T, D], move time to leading axis for scan
        qpos = jp.moveaxis(batch["qpos"], 1, 0)
        qvel = jp.moveaxis(batch["qvel"], 1, 0)
        _, outs = jax.lax.scan(scan_body, 0.0, (qpos, qvel))
        return outs

    # consume two windows
    for _ in range(2):
        batch = streamer.next_window()
        outs = consume_window(batch)
        assert outs.shape[0] == window

    streamer.stop()
