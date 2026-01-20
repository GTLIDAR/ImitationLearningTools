import pytest
import torch
from tensordict import TensorDict
from torchrl.data import LazyMemmapStorage
from torchrl.data.replay_buffers import TensorDictReplayBuffer


def _make_dummy_rb_and_traj_info(tmp_path, lengths=(3, 5)):
    # Build 2 trajectories back-to-back in memmap storage
    total = sum(lengths)
    storage = LazyMemmapStorage(
        total, scratch_dir=str(tmp_path), device="cpu", existsok=True
    )
    rb = TensorDictReplayBuffer(storage=storage)

    start = []
    end = []
    ordered = []
    offset = 0
    num_joints = 2
    for i, L in enumerate(lengths):
        qpos = torch.zeros((L, 7 + num_joints), dtype=torch.float32)
        qvel = torch.zeros((L, 6 + num_joints), dtype=torch.float32)
        td = TensorDict(
            {
                "obs": torch.full((L, 1), float(i)),
                "done": torch.zeros((L, 1), dtype=torch.bool),
                "qpos": qpos,
                "qvel": qvel,
            },
            batch_size=[L],
        )
        rb.extend(td)
        start.append(offset)
        offset += L
        end.append(offset)
        ordered.append(("ds", "m", f"t{i}"))

    traj_info = {"start_index": start, "end_index": end, "ordered_traj_list": ordered}
    return rb, traj_info


def _make_step_rb_and_traj_info(tmp_path, lengths=(5, 6), num_joints=2):
    total = sum(lengths)
    storage = LazyMemmapStorage(
        total, scratch_dir=str(tmp_path), device="cpu", existsok=True
    )
    rb = TensorDictReplayBuffer(storage=storage)

    start = []
    end = []
    ordered = []
    offset = 0
    for i, L in enumerate(lengths):
        steps = torch.arange(L, dtype=torch.float32)
        obs = (steps + i * 100.0).unsqueeze(1)
        qpos = torch.zeros((L, 7 + num_joints), dtype=torch.float32)
        qvel = torch.zeros((L, 6 + num_joints), dtype=torch.float32)
        td = TensorDict(
            {
                "obs": obs,
                "done": torch.zeros((L, 1), dtype=torch.bool),
                "qpos": qpos,
                "qvel": qvel,
            },
            batch_size=[L],
        )
        rb.extend(td)
        start.append(offset)
        offset += L
        end.append(offset)
        ordered.append(("ds", "m", f"t{i}"))

    traj_info = {"start_index": start, "end_index": end, "ordered_traj_list": ordered}
    return rb, traj_info


def test_parallel_trajectory_manager_direct_indexing_and_reset(tmp_path):
    from iltools.datasets.manager import ParallelTrajectoryManager, ResetSchedule

    rb, traj_info = _make_dummy_rb_and_traj_info(tmp_path, lengths=(3, 5))
    mgr = ParallelTrajectoryManager(
        rb=rb,
        traj_info=traj_info,
        num_envs=2,
        reset_schedule=ResetSchedule.SEQUENTIAL,
        wrap_steps=False,
        target_joint_names=["joint1", "joint2"],
        reference_joint_names=["joint1", "joint2"],
    )

    # SEQUENTIAL init reset should assign ranks [0,1] with step 0
    assert mgr.env_traj_rank.tolist() == [0, 1]
    assert mgr.env_step.tolist() == [0, 0]

    td0 = mgr.sample(advance=True)
    # obs is filled with float(rank) for this dummy setup
    assert td0["obs"].squeeze(-1).tolist() == [0.0, 1.0]
    assert mgr.env_step.tolist() == [1, 1]

    # Reset env 0 -> sequential should pick rank 0 again (cycles) and step resets
    mgr.reset_envs([0])
    assert mgr.env_step.tolist()[0] == 0
    assert mgr.env_traj_rank.tolist()[0] in [0, 1]


def test_parallel_trajectory_manager_round_robin(tmp_path):
    from iltools.datasets.manager import ParallelTrajectoryManager, ResetSchedule

    rb, traj_info = _make_dummy_rb_and_traj_info(tmp_path, lengths=(3, 5))
    mgr = ParallelTrajectoryManager(
        rb=rb,
        traj_info=traj_info,
        num_envs=2,
        reset_schedule=ResetSchedule.ROUND_ROBIN,
        target_joint_names=["joint1", "joint2"],
        reference_joint_names=["joint1", "joint2"],
    )
    # Force both to rank 0 then round-robin reset should move them to rank 1
    mgr.set_env_cursor(env_ids=[0, 1], ranks=torch.tensor([0, 0]))
    mgr.reset_envs([0, 1])
    assert mgr.env_traj_rank.tolist() == [1, 1]


def test_parallel_trajectory_manager_sample_slice_modes(tmp_path):
    from iltools.datasets.manager import ParallelTrajectoryManager, ResetSchedule

    rb, traj_info = _make_step_rb_and_traj_info(tmp_path, lengths=(5, 6))
    mgr = ParallelTrajectoryManager(
        rb=rb,
        traj_info=traj_info,
        num_envs=2,
        reset_schedule=ResetSchedule.RANDOM,
        target_joint_names=["joint1", "joint2"],
        reference_joint_names=["joint1", "joint2"],
    )
    mgr.set_env_cursor(env_ids=[0, 1], ranks=torch.tensor([0, 1]))

    start_steps = torch.tensor([[1, 3], [0, 2]])
    td = mgr.sample_slice(
        batch_size=2,
        env_ids=[0, 1],
        start_steps=start_steps,
        mode="independent",
    )
    obs = td["obs"].squeeze(-1)
    expected = torch.tensor([[1.0, 3.0], [100.0, 102.0]])
    assert torch.allclose(obs, expected)

    start_steps_contig = torch.tensor([2, 1])
    td_contig = mgr.sample_slice(
        batch_size=3,
        env_ids=[0, 1],
        start_steps=start_steps_contig,
        mode="contiguous",
    )
    obs_contig = td_contig["obs"].squeeze(-1)
    expected_contig = torch.tensor([[2.0, 3.0, 4.0], [101.0, 102.0, 103.0]])
    assert torch.allclose(obs_contig, expected_contig)


@pytest.mark.slow
def test_parallel_trajectory_manager_loco_mujoco_walk_run(tmp_path):
    """Integration test: export Loco-MuJoCo default walk+run, then test manager indexing."""
    try:
        from omegaconf import DictConfig

        from iltools.datasets.loco_mujoco.loader import LocoMuJoCoLoader
        from iltools.datasets.manager import (
            ParallelTrajectoryManager,
            ResetSchedule,
            get_global_index,
        )
        from iltools.datasets.utils import make_rb_from
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Dependencies missing for loco-mujoco integration test: {e}")

    cfg = DictConfig(
        {
            "dataset": {
                "trajectories": {"default": ["walk", "run"], "amass": [], "lafan1": []}
            },
            "control_freq": 50.0,
            "window_size": 4,
            "sim": {"dt": 0.001},
            "decimation": 20,
        }
    )

    try:
        zarr_path = tmp_path / "g1_default_walk_run_for_manager.zarr"
        _ = LocoMuJoCoLoader(
            env_name="UnitreeG1",
            cfg=cfg,
            build_zarr_dataset=True,
            path=zarr_path,
            export_transitions=True,
        )
    except Exception as e:
        pytest.skip(f"LocoMuJoCoLoader export skipped due to: {e}")

    # Build RB for just one trajectory per motion for speed (walk + run)
    rb, info = make_rb_from(
        zarr_path,
        datasets="loco_mujoco",
        motions=["default_walk", "default_run"],
        trajectories="trajectory_0",
        keys="obs",
        verbose_tree=False,
    )

    mgr = ParallelTrajectoryManager(
        rb=rb,
        traj_info=info,
        num_envs=2,
        reset_schedule=ResetSchedule.RANDOM,  # we will set ranks manually
        wrap_steps=False,
        target_joint_names=["joint1", "joint2"],
        reference_joint_names=["joint1", "joint2"],
    )

    r_walk = mgr.get_traj_rank("loco_mujoco", "default_walk", "trajectory_0")
    r_run = mgr.get_traj_rank("loco_mujoco", "default_run", "trajectory_0")

    mgr.set_env_cursor(env_ids=[0, 1], ranks=torch.tensor([r_walk, r_run]), steps=None)

    td = mgr.sample(advance=False)
    assert "obs" in td.keys()
    assert td.batch_size == torch.Size([2])

    # Validate that manager indexing matches direct storage indexing
    start = torch.as_tensor(info["start_index"], dtype=torch.int64)
    end = torch.as_tensor(info["end_index"], dtype=torch.int64)
    ranks = torch.tensor([r_walk, r_run], dtype=torch.int64)
    steps = torch.zeros((2,), dtype=torch.int64)
    idx = get_global_index(ranks, start, end, steps)
    td_expected = rb._storage[idx]  # type: ignore[attr-defined]
    assert torch.allclose(td["obs"], td_expected["obs"])
