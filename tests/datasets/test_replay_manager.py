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


def test_uniform_sampler_respects_assignment(tmp_path):
    # Build 3 segments across two tasks
    # tasks: 0 has traj 0 (T=3) and traj 1 (T=4); task 1 has traj 0 (T=2)
    tasks = {
        0: [_mk_traj(0, 0, T=3), _mk_traj(0, 1, T=4)],
        1: [_mk_traj(1, 0, T=2)],
    }
    spec = ExpertReplaySpec(
        tasks=tasks, scratch_dir=str(tmp_path), device="cpu", sample_batch_size=16
    )
    mgr = ExpertReplayManager(spec)

    # Set an assignment that excludes (task=0, traj=1)
    asg = [EnvAssignment(0, 0, 0), EnvAssignment(1, 0, 0)]
    mgr.set_assignment(asg)

    # Switch to uniform minibatching but restrict to assigned segments
    mgr.set_uniform_sampler(
        batch_size=32, without_replacement=False, respect_assignment=True
    )
    batch = mgr.buffer.sample()
    # All samples should come from either (0,0) or (1,0); never (0,1)
    obs = batch["observation"]
    # First column encodes task_id, second encodes traj_id in our synthetic builder
    tasks_ids = obs[:, 0]
    traj_ids = obs[:, 1]
    # Ensure there is no (task=0, traj=1)
    invalid = (tasks_ids == 0.0) & (traj_ids == 1.0)
    assert torch.count_nonzero(invalid) == 0


def test_update_env_assignment_sequential(tmp_path):
    # Two tasks, one traj each
    tasks = {0: [_mk_traj(0, 0, T=3)], 1: [_mk_traj(1, 0, T=3)]}
    spec = ExpertReplaySpec(tasks=tasks, scratch_dir=str(tmp_path), device="cpu")
    mgr = ExpertReplayManager(spec)
    # Start with env0->(0,0)@0, env1->(1,0)@0
    mgr.set_assignment([EnvAssignment(0, 0, 0), EnvAssignment(1, 0, 0)])
    b1 = mgr.buffer.sample()
    o1 = b1["observation"]
    assert torch.allclose(o1[0], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.allclose(o1[1], torch.tensor([1.0, 0.0, 0.0]))
    # Reassign env1 to (0,0) with step=2
    mgr.update_env_assignment(1, task_id=0, traj_id=0, step=2)
    b2 = mgr.buffer.sample()
    o2 = b2["observation"]
    # env0 advanced to t=1, env1 now reads (0,0)@2
    assert torch.allclose(o2[0], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(o2[1], torch.tensor([0.0, 0.0, 2.0]))


def test_update_env_assignment_affects_assigned_uniform(tmp_path):
    # Three segments: (0,0), (0,1), (1,0) each of length 2
    tasks = {0: [_mk_traj(0, 0, T=2), _mk_traj(0, 1, T=2)], 1: [_mk_traj(1, 0, T=2)]}
    spec = ExpertReplaySpec(tasks=tasks, scratch_dir=str(tmp_path), device="cpu")
    mgr = ExpertReplayManager(spec)
    # Initial assignment excludes (0,1)
    mgr.set_assignment([EnvAssignment(0, 0, 0), EnvAssignment(1, 0, 0)])
    mgr.set_uniform_sampler(
        batch_size=4, without_replacement=True, respect_assignment=True
    )
    batch = mgr.buffer.sample()
    obs = batch["observation"]
    assert torch.count_nonzero((obs[:, 0] == 0.0) & (obs[:, 1] == 1.0)) == 0
    # Now reassign env0 to (0,1) and ensure minibatch draws only from (0,1) and (1,0)
    mgr.update_env_assignment(0, task_id=0, traj_id=1, step=0)
    # New allowed set has (0,1) and (1,0) totaling 4 indices; sample full epoch
    batch2 = mgr.buffer.sample()
    obs2 = batch2["observation"]
    assert torch.count_nonzero((obs2[:, 0] == 0.0) & (obs2[:, 1] == 0.0)) == 0
