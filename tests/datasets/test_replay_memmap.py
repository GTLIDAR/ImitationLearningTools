import os
import tempfile

import pytest
try:
    import torch
    from tensordict import TensorDict
    from iltools_datasets.replay_memmap import (
        Segment,
        ExpertMemmapBuilder,
        build_trajectory_td,
        build_trajectory_td_from_components,
        concat_components,
    )
    TORCH_STACK_AVAILABLE = True
except Exception:
    TORCH_STACK_AVAILABLE = False
    pytest.skip("PyTorch/TensorDict not available", allow_module_level=True)


def _make_td(T: int, obs_dim: int = 3, act_dim: int = 2) -> TensorDict:
    obs = torch.arange(T * obs_dim, dtype=torch.float32).reshape(T, obs_dim)
    nxt = obs + 1
    act = torch.arange(T * act_dim, dtype=torch.float32).reshape(T, act_dim)
    return build_trajectory_td(observation=obs, action=act, next_observation=nxt)


def test_segment_indexing_wrap_and_bounds():
    seg = Segment(task_id=1, traj_id=2, start=10, length=5)
    # In range
    assert seg.index_at(0) == 10
    assert seg.index_at(4) == 14
    # Wrap
    assert seg.index_at(5) == 10
    assert seg.index_at(7) == 12
    assert seg.index_at(12) == 12  # 12 % 5 = 2 => 10 + 2

    # No wrap: out of range raises
    with pytest.raises(IndexError):
        seg.index_at(5, wrap=False)
    with pytest.raises(IndexError):
        seg.index_at(-1, wrap=False)


def test_concat_and_build_from_components():
    T = 4
    a = torch.ones(T, 2)
    b = 2 * torch.ones(T, 3)
    c = 3 * torch.ones(T, 1)

    cat = concat_components([a, b, c])
    assert cat.shape == (T, 6)
    assert torch.allclose(cat[:, :2], a)
    assert torch.allclose(cat[:, 2:5], b)
    assert torch.allclose(cat[:, 5:], c)

    td = build_trajectory_td_from_components(
        obs_parts=[a, b], action=torch.zeros(T, 1), next_obs_parts=[b, a]
    )
    assert td.batch_size == (T,)
    assert td["observation"].shape == (T, 5)
    assert td[("next", "observation")].shape == (T, 5)


def test_memmap_builder_add_and_finalize(tmp_path):
    T1, T2 = 3, 5
    td1 = _make_td(T1)
    td2 = _make_td(T2)

    builder = ExpertMemmapBuilder(scratch_dir=str(tmp_path), max_size=T1 + T2, device="cpu")
    seg1 = builder.add_trajectory(task_id=0, traj_id=0, transitions=td1)
    seg2 = builder.add_trajectory(task_id=1, traj_id=0, transitions=td2)

    assert builder.size == T1 + T2
    assert seg1.start == 0 and seg1.length == T1
    assert seg2.start == T1 and seg2.length == T2

    storage, segments = builder.finalize()
    assert len(segments) == 2
    # Storage length equals total transitions
    assert len(storage) == T1 + T2
