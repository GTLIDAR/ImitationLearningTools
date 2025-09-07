import os
import sys

import pytest
import torch

try:
    from iltools_datasets.replay_manager import (
        EnvAssignment,
        ExpertReplayManager,
        ExpertReplaySpec,
        SequentialPerEnvSampler,
    )
    from iltools_datasets.replay_memmap import Segment, build_trajectory_td

    TORCHRL_AVAILABLE = True
except Exception as e:  # pragma: no cover - environment dependent
    TORCHRL_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not TORCHRL_AVAILABLE, reason="torchrl/tensordict not available"
)


def _mk_traj(task_id: int, traj_id: int, T: int, obs_dim: int = 3, act_dim: int = 1):
    # observation encodes [task_id, traj_id, t]
    t = torch.arange(T, dtype=torch.float32).unsqueeze(-1)
    obs = torch.cat(
        [torch.full_like(t, float(task_id)), torch.full_like(t, float(traj_id)), t],
        dim=1,
    )
    nxt = obs + 0.5
    act = torch.zeros(T, act_dim)
    return build_trajectory_td(observation=obs, action=act, next_observation=nxt)


def test_sequential_per_env_sampler_indices():
    # Two segments: [0..2] and [3..4]
    segs = [
        Segment(task_id=0, traj_id=0, start=0, length=3),
        Segment(task_id=0, traj_id=1, start=3, length=2),
    ]
    asg = [
        EnvAssignment(task_id=0, traj_id=0, step=0),
        EnvAssignment(task_id=0, traj_id=1, step=1),
    ]
    sampler = SequentialPerEnvSampler(segments=segs, assignment=asg)

    idx, info = sampler.sample(storage=None, batch_size=None)  # type: ignore[arg-type]
    assert idx.shape == (2,)
    # Env0 -> seg0 @ step0 => index 0; Env1 -> seg1 @ step1 => start 3 + 1 = 4
    assert int(idx[0]) == 0
    assert int(idx[1]) == 4

    # Next call advances pointers
    idx2, info2 = sampler.sample(storage=None)  # type: ignore[arg-type]
    # Env0 -> step1 => 1; Env1 -> step2 wraps length=2 => start 3 + 0 = 3
    assert int(idx2[0]) == 1
    assert int(idx2[1]) == 3


def test_expert_replay_manager_end_to_end(tmp_path):
    # Build tasks with 2 trajectories total
    tasks = {
        0: [_mk_traj(0, 0, T=3)],
        1: [_mk_traj(1, 0, T=2)],
    }
    spec = ExpertReplaySpec(
        tasks=tasks, scratch_dir=str(tmp_path), device="cpu", sample_batch_size=4
    )
    mgr = ExpertReplayManager(spec)

    # Initially uniform sampler with default batch size
    batch = mgr.buffer.sample()
    assert isinstance(
        batch, type(mgr.buffer.sample())
    )  # sample twice to ensure no errors

    # Switch to sequential assignment for 3 envs
    asg = [EnvAssignment(0, 0, 0), EnvAssignment(1, 0, 0), EnvAssignment(0, 0, 2)]
    mgr.set_assignment(asg)
    out = mgr.buffer.sample()
    assert out.batch_size[0] == 3
    # Check that observations encode (task, traj, t)
    obs = out["observation"]
    # env0 -> task0 traj0 step0 -> [0,0,0]
    assert torch.allclose(obs[0], torch.tensor([0.0, 0.0, 0.0]))
    # env1 -> task1 traj0 step0 -> start index=3 => obs should reflect t=0 with task 1
    assert torch.allclose(obs[1], torch.tensor([1.0, 0.0, 0.0]))
    # env2 -> task0 traj0 step2 -> last element of first segment (t=2)
    assert torch.allclose(obs[2], torch.tensor([0.0, 0.0, 2.0]))

    # Next call should advance steps by 1 in sequence for each env
    out2 = mgr.buffer.sample()
    obs2 = out2["observation"]
    # env0 now t=1
    assert torch.allclose(obs2[0], torch.tensor([0.0, 0.0, 1.0]))
    # env1 now t=1 (second segment length=2)
    assert torch.allclose(obs2[1], torch.tensor([1.0, 0.0, 1.0]))
    # env2 wraps to t=0
    assert torch.allclose(obs2[2], torch.tensor([0.0, 0.0, 0.0]))

    # Switch to uniform sampler with custom batch size
    mgr.set_uniform_sampler(batch_size=5, without_replacement=True)
    uni = mgr.buffer.sample()
    assert uni.batch_size[0] == 5
