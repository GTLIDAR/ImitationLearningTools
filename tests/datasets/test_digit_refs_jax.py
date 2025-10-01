import os
from typing import Tuple

import jax
import jax.numpy as jp
import numpy as np

from iltools_datasets.trajopt.digit_refs import DigitTrajoptRefBuilder, DigitRefSpec


def _write_trajopt_npz(root_dir: str) -> Tuple[str, list[int]]:
    os.makedirs(root_dir, exist_ok=True)
    lengths = [9, 7]
    qpos_dim, qvel_dim = 8, 6

    for i, L in enumerate(lengths):
        qpos = np.zeros((L, qpos_dim), dtype=np.float32)
        qvel = np.zeros((L, qvel_dim), dtype=np.float32)
        ee = np.zeros((L, 5, 3), dtype=np.float32)
        for t in range(L):
            qpos[t, :] = 10 * i + t
            qvel[t, :] = 20 * i + t
            ee[t, :, :] = 30 * i + t
        np.savez(
            os.path.join(root_dir, f"traj_{i}.npz"), qpos=qpos, qvel=qvel, ee_pos=ee
        )

    return root_dir, lengths


def test_digit_refs_jit_vmap(tmp_path):
    npz_dir, lengths = _write_trajopt_npz(os.path.join(tmp_path, "npz"))

    spec = DigitRefSpec(
        a_pos_index=np.arange(3, dtype=np.int32),
        a_vel_index=np.arange(3, dtype=np.int32),
        subsample=1,
    )
    builder = DigitTrajoptRefBuilder(npz_dir, spec)

    # build two different refs (no JIT), confirm padding to max_len
    refs0 = builder.build_by_index(0)
    refs1 = builder.build_by_index(1)
    assert refs0["ref_len"].shape == (1,)
    assert refs1["ref_len"].shape == (1,)
    assert refs0["ref_motor_joint_pos"].shape[0] == builder.max_len
    assert refs1["ref_motor_joint_pos"].shape[0] == builder.max_len

    # JIT a simple step that indexes current step
    def get_step_refs(refs_dict, t):
        # refs_dict contains numpy; convert to jax arrays outside jit in prod
        refs = {k: jp.asarray(v) for k, v in refs_dict.items() if k != "ref_len"}
        return (
            refs["ref_motor_joint_pos"][t],
            refs["ref_base_local_pos"][t],
            refs["ref_local_ee_pos"][t],
        )

    jitted = jax.jit(get_step_refs)
    out0 = jitted(refs0, 0)
    out1 = jitted(refs1, 1)
    assert all(x.shape[0] > 0 for x in out0)
    assert all(x.shape[0] > 0 for x in out1)

    # vmap across envs over prebuilt refs
    batched_refs = [refs0, refs1]

    def per_env(refs_dict, t):
        refs = {k: jp.asarray(v) for k, v in refs_dict.items() if k != "ref_len"}
        return refs["ref_motor_joint_pos"][t]

    vmapped = jax.vmap(per_env, in_axes=(0, None))
    stacked = vmapped(batched_refs, 0)
    assert stacked.shape[0] == 2
