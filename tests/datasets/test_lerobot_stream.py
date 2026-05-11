from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from iltools.datasets.lerobot_stream import (
    LeRobotStreamingCacheConfig,
    StreamingTensorDictReplayCache,
    UnitreeG1WBT29DofMapper,
    UnitreeG1WBT29DofMapperConfig,
)


def _make_mapper() -> UnitreeG1WBT29DofMapper:
    default_joint_pos = torch.linspace(-0.2, 0.2, 29)
    action_scale = torch.linspace(0.5, 1.5, 29)
    return UnitreeG1WBT29DofMapper(
        UnitreeG1WBT29DofMapperConfig(
            default_joint_pos=default_joint_pos.tolist(),
            action_scale=action_scale.tolist(),
        )
    )


def _make_fake_wbt_rows(
    *,
    episode_index: int = 7,
    length: int = 5,
) -> tuple[list[dict[str, object]], torch.Tensor, torch.Tensor]:
    mapper = _make_mapper()
    default_joint_pos = mapper.default_joint_pos
    action_scale = mapper.action_scale

    q_current = torch.zeros(length, 36)
    q_current[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    joint_offsets = torch.linspace(-0.3, 0.3, 29)
    frame_offsets = torch.arange(length, dtype=torch.float32).unsqueeze(-1) * 0.1
    q_current[:, 7:] = default_joint_pos + joint_offsets + frame_offsets

    expert_action = torch.stack(
        [torch.linspace(-0.5, 0.5, 29) + 0.05 * float(frame) for frame in range(length)]
    )
    q_desired = q_current.clone()
    q_desired[:, 7:] = default_joint_pos + expert_action * action_scale

    rows = [
        {
            "episode_index": episode_index,
            "observation.state.robot_q_current": q_current[index],
            "action.robot_q_desired": q_desired[index],
        }
        for index in range(length)
    ]
    return rows, q_current, expert_action


def test_unitree_g1_wbt_mapper_builds_aligned_training_transitions() -> None:
    mapper = _make_mapper()
    rows, q_current, expert_action = _make_fake_wbt_rows(length=5)

    transitions = mapper.map_episode(rows)

    assert transitions.batch_size == torch.Size([4])
    torch.testing.assert_close(transitions["action"], expert_action[:-1])
    torch.testing.assert_close(transitions["expert_action"], expert_action[:-1])
    torch.testing.assert_close(
        transitions.get(("policy", "root_quat")),
        torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(4, 4),
    )
    torch.testing.assert_close(
        transitions.get(("next", "policy", "joint_pos")),
        q_current[1:, 7:],
    )
    torch.testing.assert_close(
        transitions.get(("policy", "last_action"))[0],
        torch.zeros(29),
    )
    torch.testing.assert_close(
        transitions.get(("policy", "last_action"))[1:],
        expert_action[:-2],
    )
    torch.testing.assert_close(
        transitions.get(("next", "policy", "last_action")),
        expert_action[:-1],
    )
    torch.testing.assert_close(
        transitions.get(("policy", "joint_vel_rel")),
        torch.full((4, 29), 3.0),
    )
    torch.testing.assert_close(
        transitions.get(("policy", "base_ang_vel")),
        torch.zeros(4, 3),
    )
    torch.testing.assert_close(
        transitions.get(("policy", "expert_motion")),
        torch.cat([q_current[:-1, 7:], torch.full((4, 29), 3.0)], dim=-1),
    )
    torch.testing.assert_close(
        transitions.get(("policy", "expert_anchor_pos_b")),
        torch.zeros(4, 3),
    )
    expected_rot6d = torch.zeros(4, 6)
    expected_rot6d[:, 0] = 1.0
    expected_rot6d[:, 4] = 1.0
    torch.testing.assert_close(
        transitions.get(("policy", "expert_anchor_ori_b")),
        expected_rot6d,
    )
    torch.testing.assert_close(
        transitions.get(("reward_input", "expert_motion")),
        transitions.get(("policy", "expert_motion")),
    )
    assert transitions["done"].tolist() == [False, False, False, True]
    assert transitions.get(("next", "done")).tolist() == [False, False, False, True]


def test_unitree_g1_wbt_mapper_selects_episode_default_from_pool() -> None:
    default_a = torch.zeros(29)
    default_b = torch.linspace(-0.1, 0.1, 29)
    action_scale = torch.ones(29)
    mapper = UnitreeG1WBT29DofMapper(
        UnitreeG1WBT29DofMapperConfig(
            default_joint_pos=[default_a.tolist(), default_b.tolist()],
            action_scale=action_scale.tolist(),
        )
    )
    length = 4
    q_current = torch.zeros(length, 36)
    q_current[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    q_current[:, 7:] = default_b + 0.25
    q_desired = q_current.clone()
    q_desired[:, 7:] = default_b + 0.5
    rows = [
        {
            "episode_index": 1,
            "observation.state.robot_q_current": q_current[index],
            "action.robot_q_desired": q_desired[index],
        }
        for index in range(length)
    ]

    transitions = mapper.map_episode(rows)

    torch.testing.assert_close(
        transitions.get(("policy", "joint_pos_rel")),
        torch.full((length - 1, 29), 0.25),
    )
    torch.testing.assert_close(
        transitions.get("expert_action"),
        torch.full((length - 1, 29), 0.5),
    )


def test_unitree_g1_wbt_mapper_fails_fast_on_bad_robot_width() -> None:
    mapper = _make_mapper()
    episode = TensorDict(
        {
            "observation.state.robot_q_current": torch.zeros(3, 35),
            "action.robot_q_desired": torch.zeros(3, 36),
        },
        batch_size=[3],
    )

    with pytest.raises(ValueError, match=r"shape \[T, 36\]"):
        mapper.map_episode(episode)


def test_streaming_cache_fills_memmap_before_sampling(tmp_path) -> None:
    mapper = _make_mapper()
    rows, _, _ = _make_fake_wbt_rows(length=6)
    cache = StreamingTensorDictReplayCache(
        LeRobotStreamingCacheConfig(
            cache_dir=tmp_path,
            max_cache_transitions=16,
            min_ready_transitions=5,
            low_watermark=4,
            batch_size=2,
            max_episodes=1,
            mapper=mapper.config,
        ),
        mapper=mapper,
        source=rows,
    )

    cache.start()
    cache.wait_until_ready(timeout_s=5.0)
    sample = cache.sample(2)
    cache.stop()

    assert sample.numel() == 2
    assert ("policy", "base_ang_vel") in sample.keys(True)
    assert ("next", "policy", "joint_pos_rel") in sample.keys(True)
    assert "expert_action" in sample.keys(True)
