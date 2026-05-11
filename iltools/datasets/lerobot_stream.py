"""LeRobot streaming ingestion utilities for offline imitation pretraining.

This module keeps LeRobot on the input side and TorchRL TensorDict replay
buffers on the training side. Heavy conversion happens before samples enter
the replay buffer, so algorithm code can consume validated TensorDict batches.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict
from tensordict.base import TensorDictBase
from torch import Tensor
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.writers import TensorDictRoundRobinWriter

logger = logging.getLogger(__name__)


UNITREE_G1_WBT_DEFAULT_REPO_ID = "unitreerobotics/G1_WBT_Brainco_Pickup_Pillow"


@dataclass(frozen=True)
class UnitreeG1WBT29DofMapperConfig:
    """Schema and robot constants for Unitree G1 WBT low-dimensional data."""

    robot_q_current_key: str = "observation.state.robot_q_current"
    robot_q_desired_key: str = "action.robot_q_desired"
    episode_key: str = "episode_index"
    dt: float = 1.0 / 30.0
    default_joint_pos: Sequence[Any] = ()
    action_scale: Sequence[float] = ()
    quat_order: str = "wxyz"


@dataclass(frozen=True)
class LeRobotStreamingCacheConfig:
    """Runtime options for streaming LeRobot data into a TorchRL cache."""

    repo_id: str = UNITREE_G1_WBT_DEFAULT_REPO_ID
    split: str = "train"
    cache_dir: str | Path = "/tmp/iltools_lerobot_torchrl_cache"
    max_cache_transitions: int = 5_000_000
    min_ready_transitions: int = 100_000
    low_watermark: int = 1_000_000
    starvation_timeout_s: float = 300.0
    local_sample_prefetch: int = 0
    batch_size: int | None = None
    max_episodes: int | None = None
    mapper: UnitreeG1WBT29DofMapperConfig = UnitreeG1WBT29DofMapperConfig()


def _to_tensor(value: Any) -> Tensor:
    if torch.is_tensor(value):
        return value.detach().to(dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


def _normalize_quat_wxyz(quat: Tensor, quat_order: str) -> Tensor:
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternion width 4, got {tuple(quat.shape)}.")
    if quat_order == "wxyz":
        quat_wxyz = quat
    elif quat_order == "xyzw":
        quat_wxyz = quat[..., [3, 0, 1, 2]]
    else:
        raise ValueError(f"Unsupported quat_order={quat_order!r}.")
    return quat_wxyz / quat_wxyz.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)


def _quat_conjugate_wxyz(quat: Tensor) -> Tensor:
    return torch.cat([quat[..., :1], -quat[..., 1:]], dim=-1)


def _quat_mul_wxyz(lhs: Tensor, rhs: Tensor) -> Tensor:
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _axis_angle_from_quat_wxyz(quat: Tensor) -> Tensor:
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    vector = quat[..., 1:]
    vector_norm = vector.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, quat[..., :1].clamp(-1.0, 1.0))
    axis = vector / vector_norm.clamp_min(1.0e-8)
    return axis * angle


def _finite_difference(values: Tensor, dt: float) -> Tensor:
    if values.shape[0] < 2:
        raise ValueError("Need at least two frames to finite-difference an episode.")
    return torch.gradient(values, spacing=(float(dt),), dim=0)[0]


def _so3_derivative_wxyz(rotations: Tensor, dt: float) -> Tensor:
    if rotations.shape[0] < 3:
        return torch.zeros(
            (rotations.shape[0], 3), dtype=rotations.dtype, device=rotations.device
        )
    q_prev = rotations[:-2]
    q_next = rotations[2:]
    q_rel = _quat_mul_wxyz(_quat_conjugate_wxyz(q_prev), q_next)
    omega = _axis_angle_from_quat_wxyz(q_rel) / (2.0 * float(dt))
    return torch.cat([omega[:1], omega, omega[-1:]], dim=0)


def _get_required(mapping: Mapping[Any, Any] | TensorDictBase, key: str) -> Any:
    if isinstance(mapping, TensorDictBase):
        value = mapping.get(key)
    elif key in mapping:
        value = mapping[key]
    else:
        raise KeyError(f"Missing required LeRobot field {key!r}.")
    if value is None:
        raise KeyError(f"Missing required LeRobot field {key!r}.")
    return value


def _stack_rows(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> TensorDict:
    if len(rows) == 0:
        raise ValueError("Cannot stack an empty episode.")
    data = {}
    for key in keys:
        data[key] = torch.stack([_to_tensor(_get_required(row, key)) for row in rows])
    return TensorDict(data, batch_size=[len(rows)])


def _identity_rot6d(*, length: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    rot6d = torch.zeros((length, 6), device=device, dtype=dtype)
    rot6d[:, 0] = 1.0
    rot6d[:, 4] = 1.0
    return rot6d


class UnitreeG1WBT29DofMapper:
    """Map Unitree G1 WBT LeRobot episodes into canonical training transitions."""

    robot_q_width = 36
    joint_width = 29

    def __init__(self, config: UnitreeG1WBT29DofMapperConfig) -> None:
        self.config = config
        if float(config.dt) <= 0.0:
            raise ValueError("mapper.dt must be positive.")
        default_joint_pos = _to_tensor(config.default_joint_pos)
        if default_joint_pos.ndim == 1:
            self.default_joint_pos_pool = default_joint_pos.unsqueeze(0)
        elif default_joint_pos.ndim == 2:
            self.default_joint_pos_pool = default_joint_pos
        else:
            raise ValueError(
                "default_joint_pos must have shape [29] or [N, 29], got "
                f"{tuple(default_joint_pos.shape)}."
            )
        self.default_joint_pos = self.default_joint_pos_pool[0]
        self.action_scale = _to_tensor(config.action_scale).flatten()
        if self.default_joint_pos_pool.shape[-1] != self.joint_width:
            raise ValueError(
                "default_joint_pos must contain 29 G1 joint values per row, got "
                f"{tuple(self.default_joint_pos_pool.shape)}."
            )
        if self.default_joint_pos_pool.shape[0] <= 0:
            raise ValueError("default_joint_pos must contain at least one row.")
        if tuple(self.action_scale.shape) != (self.joint_width,):
            raise ValueError(
                "action_scale must contain 29 G1 joint values, got "
                f"{tuple(self.action_scale.shape)}."
            )
        if torch.any(self.action_scale.abs() <= 1.0e-8):
            raise ValueError("action_scale must not contain zeros.")

    def map_episode(
        self, episode: TensorDictBase | Mapping[str, Any] | Sequence[Mapping[str, Any]]
    ) -> TensorDict:
        if isinstance(episode, Sequence) and not isinstance(episode, TensorDictBase):
            episode_td = _stack_rows(
                episode,
                (
                    self.config.episode_key,
                    self.config.robot_q_current_key,
                    self.config.robot_q_desired_key,
                ),
            )
        else:
            data = {
                self.config.robot_q_current_key: _to_tensor(
                    _get_required(episode, self.config.robot_q_current_key)  # type: ignore[arg-type]
                ),
                self.config.robot_q_desired_key: _to_tensor(
                    _get_required(episode, self.config.robot_q_desired_key)  # type: ignore[arg-type]
                ),
            }
            if self.config.episode_key in episode:  # type: ignore[operator]
                data[self.config.episode_key] = _to_tensor(  # type: ignore[index]
                    episode[self.config.episode_key]  # type: ignore[index]
                )
            episode_td = TensorDict(
                data,
                batch_size=[
                    int(
                        _to_tensor(
                            _get_required(episode, self.config.robot_q_current_key)  # type: ignore[arg-type]
                        ).shape[0]
                    )
                ],
            )
        return self._map_batched_episode(episode_td)

    def _episode_default_joint_pos(self, episode: TensorDictBase, like: Tensor) -> Tensor:
        pool = self.default_joint_pos_pool.to(device=like.device, dtype=like.dtype)
        if pool.shape[0] == 1:
            return pool[0]
        episode_index = episode.get(self.config.episode_key)
        if episode_index is None:
            raise KeyError(
                "Unitree G1 WBT mapper requires episode_index to select from a "
                "default_joint_pos pool."
            )
        pool_index = int(_to_tensor(episode_index).flatten()[0].item()) % int(
            pool.shape[0]
        )
        return pool[pool_index]

    def _map_batched_episode(self, episode: TensorDictBase) -> TensorDict:
        robot_q_current = _to_tensor(episode.get(self.config.robot_q_current_key))
        robot_q_desired = _to_tensor(episode.get(self.config.robot_q_desired_key))
        if robot_q_current.ndim != 2 or robot_q_current.shape[-1] != self.robot_q_width:
            raise ValueError(
                "robot_q_current must have shape [T, 36], got "
                f"{tuple(robot_q_current.shape)}."
            )
        if robot_q_desired.ndim != 2 or robot_q_desired.shape[-1] != self.robot_q_width:
            raise ValueError(
                "robot_q_desired must have shape [T, 36], got "
                f"{tuple(robot_q_desired.shape)}."
            )
        if robot_q_current.shape[0] != robot_q_desired.shape[0]:
            raise ValueError("robot_q_current and robot_q_desired lengths differ.")
        if int(robot_q_current.shape[0]) < 2:
            raise ValueError("A WBT episode must contain at least two frames.")

        default_joint_pos = self._episode_default_joint_pos(episode, robot_q_current)
        action_scale = self.action_scale.to(
            device=robot_q_current.device, dtype=robot_q_current.dtype
        )
        root_quat = _normalize_quat_wxyz(
            robot_q_current[:, 3:7], self.config.quat_order
        )
        root_pos = robot_q_current[:, :3]
        joint_pos = robot_q_current[:, 7:]
        joint_vel = _finite_difference(joint_pos, self.config.dt)
        base_ang_vel = _so3_derivative_wxyz(root_quat, self.config.dt)
        expert_motion = torch.cat([joint_pos, joint_vel], dim=-1)
        expert_anchor_pos_b = torch.zeros(
            (robot_q_current.shape[0], 3),
            device=robot_q_current.device,
            dtype=robot_q_current.dtype,
        )
        expert_anchor_ori_b = _identity_rot6d(
            length=int(robot_q_current.shape[0]),
            device=robot_q_current.device,
            dtype=robot_q_current.dtype,
        )
        expert_action = (robot_q_desired[:, 7:] - default_joint_pos) / action_scale
        last_action = torch.cat(
            [torch.zeros_like(expert_action[:1]), expert_action[:-1]], dim=0
        )

        n = int(robot_q_current.shape[0]) - 1
        done = torch.zeros(n, dtype=torch.bool, device=robot_q_current.device)
        done[-1] = True

        return TensorDict(
            {
                ("policy", "root_pos"): root_pos[:-1],
                ("policy", "root_quat"): root_quat[:-1],
                ("policy", "joint_pos"): joint_pos[:-1],
                ("policy", "base_ang_vel"): base_ang_vel[:-1],
                ("policy", "joint_pos_rel"): joint_pos[:-1] - default_joint_pos,
                ("policy", "joint_vel_rel"): joint_vel[:-1],
                ("policy", "last_action"): last_action[:-1],
                ("policy", "expert_motion"): expert_motion[:-1],
                ("policy", "expert_anchor_pos_b"): expert_anchor_pos_b[:-1],
                ("policy", "expert_anchor_ori_b"): expert_anchor_ori_b[:-1],
                ("critic", "expert_motion"): expert_motion[:-1],
                ("critic", "expert_anchor_pos_b"): expert_anchor_pos_b[:-1],
                ("critic", "expert_anchor_ori_b"): expert_anchor_ori_b[:-1],
                ("reward_input", "expert_motion"): expert_motion[:-1],
                ("reward_input", "expert_anchor_pos_b"): expert_anchor_pos_b[:-1],
                ("reward_input", "expert_anchor_ori_b"): expert_anchor_ori_b[:-1],
                ("next", "policy", "root_pos"): root_pos[1:],
                ("next", "policy", "root_quat"): root_quat[1:],
                ("next", "policy", "joint_pos"): joint_pos[1:],
                ("next", "policy", "base_ang_vel"): base_ang_vel[1:],
                ("next", "policy", "joint_pos_rel"): joint_pos[1:] - default_joint_pos,
                ("next", "policy", "joint_vel_rel"): joint_vel[1:],
                ("next", "policy", "last_action"): last_action[1:],
                ("next", "policy", "expert_motion"): expert_motion[1:],
                ("next", "policy", "expert_anchor_pos_b"): expert_anchor_pos_b[1:],
                ("next", "policy", "expert_anchor_ori_b"): expert_anchor_ori_b[1:],
                "action": expert_action[:-1],
                "expert_action": expert_action[:-1],
                "done": done,
                ("next", "done"): done,
                ("next", "truncated"): torch.zeros_like(done),
            },
            batch_size=[n],
        )


class StreamingTensorDictReplayCache:
    """Bounded local TensorDict replay cache populated by a background producer."""

    def __init__(
        self,
        config: LeRobotStreamingCacheConfig,
        *,
        mapper: UnitreeG1WBT29DofMapper,
        source: Iterable[Mapping[str, Any]] | None = None,
    ) -> None:
        if int(config.max_cache_transitions) <= 0:
            raise ValueError("max_cache_transitions must be positive.")
        if int(config.min_ready_transitions) < 0:
            raise ValueError("min_ready_transitions must be >= 0.")
        if int(config.low_watermark) < 0:
            raise ValueError("low_watermark must be >= 0.")
        if int(config.min_ready_transitions) > int(config.max_cache_transitions):
            raise ValueError(
                "min_ready_transitions cannot exceed max_cache_transitions."
            )
        if int(config.low_watermark) > int(config.max_cache_transitions):
            raise ValueError("low_watermark cannot exceed max_cache_transitions.")
        self.config = config
        self.mapper = mapper
        self.source = source
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        storage = LazyMemmapStorage(
            int(config.max_cache_transitions),
            scratch_dir=str(self.cache_dir),
            device="cpu",
            existsok=True,
        )
        self.replay_buffer = TensorDictReplayBuffer(
            storage=storage,
            writer=TensorDictRoundRobinWriter(),
            batch_size=config.batch_size,
            prefetch=int(config.local_sample_prefetch)
            if int(config.local_sample_prefetch) > 0
            else None,
        )
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._error: BaseException | None = None
        self._episodes_written = 0

    @property
    def ready_transitions(self) -> int:
        return len(self.replay_buffer)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Streaming cache producer has already been started.")
        self._thread = threading.Thread(
            target=self._producer_loop,
            name="lerobot-streaming-cache",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)

    def wait_until_ready(self, timeout_s: float | None = None) -> None:
        min_ready = int(self.config.min_ready_transitions)
        timeout_s = (
            float(self.config.starvation_timeout_s) if timeout_s is None else timeout_s
        )
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self.ready_transitions >= min_ready
                or self._error is not None
                or (self._thread is not None and not self._thread.is_alive()),
                timeout=float(timeout_s),
            )
            if self._error is not None:
                raise RuntimeError(
                    "LeRobot streaming producer failed."
                ) from self._error
            if self.ready_transitions >= min_ready:
                return
            if not ready:
                raise TimeoutError(
                    "Timed out waiting for LeRobot cache readiness: "
                    f"ready={self.ready_transitions}, min_ready={min_ready}."
                )
            raise RuntimeError(
                "LeRobot streaming producer finished before cache reached "
                f"min_ready_transitions={min_ready}; ready={self.ready_transitions}."
            )

    def sample(self, batch_size: int | None = None) -> TensorDict:
        with self._condition:
            if self._error is not None:
                raise RuntimeError(
                    "LeRobot streaming producer failed."
                ) from self._error
            if self.ready_transitions <= 0:
                raise RuntimeError("Cannot sample from an empty LeRobot cache.")
            return self.replay_buffer.sample(batch_size)

    def _source_iter(self) -> Iterator[Mapping[str, Any]]:
        if self.source is not None:
            yield from self.source
            return
        try:
            from lerobot.datasets import StreamingLeRobotDataset
        except ImportError:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise ImportError(
                    "lerobot_stream requires either lerobot or datasets. "
                    "Install iltools[lerobot], install lerobot directly, or install "
                    "huggingface datasets."
                ) from exc
            yield from load_dataset(
                self.config.repo_id,
                split=self.config.split,
                streaming=True,
            )
            return
        yield from StreamingLeRobotDataset(self.config.repo_id)

    def _producer_loop(self) -> None:
        try:
            current_episode_id: int | None = None
            current_rows: list[Mapping[str, Any]] = []
            for row in self._source_iter():
                if self._stop_event.is_set():
                    break
                episode_id = int(_get_required(row, self.config.mapper.episode_key))
                if current_episode_id is None:
                    current_episode_id = episode_id
                if episode_id != current_episode_id:
                    self._write_episode(current_rows)
                    current_rows = []
                    current_episode_id = episode_id
                    if (
                        self.config.max_episodes is not None
                        and self._episodes_written >= int(self.config.max_episodes)
                    ):
                        break
                current_rows.append(row)
            if current_rows and not self._stop_event.is_set():
                self._write_episode(current_rows)
        except BaseException as exc:  # noqa: BLE001
            with self._condition:
                self._error = exc
                self._condition.notify_all()

    def _write_episode(self, rows: Sequence[Mapping[str, Any]]) -> None:
        transitions = self.mapper.map_episode(rows)
        with self._condition:
            self.replay_buffer.extend(transitions)
            self._episodes_written += 1
            ready_transitions = self.ready_transitions
            self._condition.notify_all()
        logger.debug(
            "cached Unitree WBT episode %d | rows=%d | transitions=%d | ready=%d",
            self._episodes_written,
            len(rows),
            transitions.numel(),
            ready_transitions,
        )
