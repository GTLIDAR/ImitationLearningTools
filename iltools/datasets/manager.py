from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import TensorDictReplayBuffer

from .utils import _map_reference_to_target, get_ith_traj_info

logger = logging.getLogger(__name__)


def _compile_if_available(fn: Callable):
    if hasattr(torch, "compile"):
        return torch.compile(fn)
    return fn


@_compile_if_available
def _get_global_index_impl(
    start_rank: Tensor, end_rank: Tensor, local_step: Tensor
) -> Tensor:
    return (start_rank + local_step).clamp(min=start_rank, max=end_rank - 1)


def get_global_index(
    rank: Tensor, start: Tensor, end: Tensor, local_step: Tensor
) -> Tensor:
    """Vectorized map (traj_rank, local_step) -> global index in replay storage."""
    start_rank = start.index_select(0, rank)
    end_rank = end.index_select(0, rank)
    if local_step.ndim > 1:
        view_shape = (start_rank.shape[0],) + (1,) * (local_step.ndim - 1)
        start_rank = start_rank.view(view_shape)
        end_rank = end_rank.view(view_shape)
    return _get_global_index_impl(start_rank, end_rank, local_step)


@_compile_if_available
def _clamp_env_steps_impl(steps_t: Tensor, max_step: Tensor) -> Tensor:
    return torch.minimum(steps_t.clamp(min=0), max_step)


@_compile_if_available
def _advance_steps_wrap_impl(step: Tensor, length: Tensor) -> Tensor:
    return (step + 1) % length


@_compile_if_available
def _advance_steps_clamp_impl(step: Tensor, length: Tensor) -> Tensor:
    return torch.minimum(step + 1, length - 1)


@_compile_if_available
def _make_contiguous_local_steps_impl(
    start_steps: Tensor, step_offsets: Tensor, max_step: Tensor
) -> Tensor:
    return torch.minimum(
        start_steps.unsqueeze(1) + step_offsets.unsqueeze(0),
        max_step.unsqueeze(1),
    )


@_compile_if_available
def _clamp_independent_local_steps_impl(
    local_steps: Tensor, max_step: Tensor
) -> Tensor:
    return torch.minimum(local_steps.clamp(min=0), max_step.unsqueeze(1))


@_compile_if_available
def _extract_reference_fields_impl(
    qpos: Tensor,
    qvel: Tensor,
    nan_joint_template: Tensor,
    ref_to_target_map: Tensor,
    target_to_ref_map: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    root_pos = qpos[..., 0:3]
    root_quat = qpos[..., 3:7]
    root_lin_vel = qvel[..., 0:3]
    root_ang_vel = qvel[..., 3:6]
    joint_pos_src = qpos[..., 7:].index_select(-1, target_to_ref_map)
    joint_vel_src = qvel[..., 6:].index_select(-1, target_to_ref_map)
    out_shape = qpos.shape[:-1] + (nan_joint_template.shape[-1],)
    joint_pos_out = nan_joint_template.expand(out_shape).clone()
    joint_vel_out = nan_joint_template.expand(out_shape).clone()
    joint_pos_out[..., ref_to_target_map] = joint_pos_src
    joint_vel_out[..., ref_to_target_map] = joint_vel_src
    return (
        root_pos,
        root_quat,
        root_lin_vel,
        root_ang_vel,
        joint_pos_out,
        joint_vel_out,
    )


@dataclass(frozen=True)
class ResetSchedule:
    """How to pick the next trajectory rank when an env resets."""

    RANDOM: str = "random"
    SEQUENTIAL: str = "sequential"
    ROUND_ROBIN: str = "round_robin"
    CUSTOM: str = "custom"


class ParallelTrajectoryManager:
    """Manage `num_envs` logical environments, each bound to one expert trajectory.

    - Each env tracks a `(traj_rank, local_step)` cursor.
    - Sampling is done by direct indexing into the replay buffer storage with the
      computed global indices (one step per env).
    - On reset, an env is reassigned to a new trajectory according to
      `reset_schedule`, where trajectories are referenced by their **rank**
      in `traj_info["ordered_traj_list"]`.
    """

    def __init__(
        self,
        *,
        rb: TensorDictReplayBuffer,
        traj_info: dict,
        num_envs: int,
        reset_schedule: str = ResetSchedule.RANDOM,
        custom_reset_fn: Optional[Callable[[Tensor, int], Tensor]] = None,
        reset_start_step: int = 0,
        wrap_steps: bool = False,
        device: torch.device | str | None = None,
        target_joint_names: Optional[Sequence[str]] = None,
        reference_joint_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.rb = rb
        self.traj_info = traj_info
        self.reset_schedule = reset_schedule
        self.custom_reset_fn = custom_reset_fn
        self.reference_joint_names = list(reference_joint_names or [])
        self.target_joint_names = list(target_joint_names or [])
        if int(reset_start_step) < 0:
            raise ValueError("reset_start_step must be >= 0")
        self.reset_start_step = int(reset_start_step)
        self.wrap_steps = bool(wrap_steps)

        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)

        self._device = torch.device(device) if device is not None else None
        storage = getattr(self.rb, "_storage", None)
        storage_device = getattr(storage, "device", None)
        self._storage_device = (
            torch.device(storage_device)
            if storage_device is not None
            else torch.device("cpu")
        )
        self._state_device = self._device or self._storage_device

        try:
            start = torch.as_tensor(
                traj_info["start_index"],
                dtype=torch.int64,
                device=self._state_device,
            )
            end = torch.as_tensor(
                traj_info["end_index"],
                dtype=torch.int64,
                device=self._state_device,
            )
            ordered = traj_info["ordered_traj_list"]
        except KeyError as e:
            raise KeyError(
                "traj_info must contain keys: 'start_index', 'end_index', 'ordered_traj_list'"
            ) from e

        if start.ndim != 1 or end.ndim != 1 or start.numel() != end.numel():
            raise ValueError("traj_info start/end must be 1D and have the same length")
        if int(start.numel()) == 0:
            raise ValueError("No trajectories found in traj_info")

        self._start = start
        self._end = end
        self._length = (end - start).clamp(min=0)
        self._ordered_traj_list = list(ordered)
        self._traj_rank_lookup = {
            traj: idx for idx, traj in enumerate(self._ordered_traj_list)
        }
        sample_keys = {str(key) for key in self.rb[0].keys()}
        self.has_next_reference = (
            "next_qpos" in sample_keys and "next_qvel" in sample_keys
        )

        self.env_traj_rank = torch.zeros(
            (self.num_envs,), dtype=torch.int64, device=self._state_device
        )
        self.env_step = torch.zeros(
            (self.num_envs,), dtype=torch.int64, device=self._state_device
        )
        self._all_env_ids = torch.arange(
            self.num_envs, dtype=torch.int64, device=self._state_device
        )
        self._step_offsets_cache: dict[int, Tensor] = {}

        self._next_rank = 0
        self._reconstructed_action_targets: Tensor | None = None

        if self.reset_schedule == ResetSchedule.ROUND_ROBIN:
            # Prime per-env cursors so the first reset fans out across trajectories
            # while keeping later resets as "advance this env to its next rank".
            self.env_traj_rank.copy_(
                torch.remainder(self._all_env_ids - 1, self.num_trajectories)
            )
        self.reset_envs(self._all_env_ids)
        logger.info(
            "Initialized ParallelTrajectoryManager(num_envs=%s, num_trajectories=%s, reset_schedule=%s, reset_start_step=%s)",
            self.num_envs,
            self.num_trajectories,
            self.reset_schedule,
            self.reset_start_step,
        )

        self.ref_to_target_map, self.target_to_ref_map = _map_reference_to_target(
            self.reference_joint_names,
            self.target_joint_names,
            self._state_device,
        )
        self._num_target_joints = len(self.target_joint_names)
        self.target_mask = torch.zeros(
            self._num_target_joints, dtype=torch.bool, device=self._state_device
        )
        self.target_mask[self.ref_to_target_map] = True
        self._nan_joint_template = torch.full(
            (1, self._num_target_joints),
            torch.nan,
            dtype=torch.float32,
            device=self._state_device,
        )

        self.joint_pos = torch.empty(
            self.num_envs,
            self._num_target_joints,
            device=self._state_device,
            dtype=torch.float32,
        )
        self.joint_vel = torch.empty(
            self.num_envs,
            self._num_target_joints,
            device=self._state_device,
            dtype=torch.float32,
        )
        self.root_pos = torch.empty(
            self.num_envs, 3, device=self._state_device, dtype=torch.float32
        )
        self.root_quat = torch.empty(
            self.num_envs, 4, device=self._state_device, dtype=torch.float32
        )
        self.root_lin_vel = torch.empty(
            self.num_envs, 3, device=self._state_device, dtype=torch.float32
        )
        self.root_ang_vel = torch.empty(
            self.num_envs, 3, device=self._state_device, dtype=torch.float32
        )

    @property
    def num_trajectories(self) -> int:
        return int(self._start.numel())

    @property
    def storage_device(self) -> torch.device:
        return self._storage_device

    def get_env_traj_info(self, env_id: int) -> tuple[str, str, str]:
        """Return (dataset, motion, trajectory) tuple for the env's current rank."""
        r = int(self.env_traj_rank[int(env_id)])
        return get_ith_traj_info(r, self._ordered_traj_list)  # type: ignore[arg-type]

    def get_traj_rank(self, dataset: str, motion: str, trajectory: str) -> int:
        """Map (dataset,motion,trajectory) to trajectory rank."""
        return self._traj_rank_lookup[(dataset, motion, trajectory)]

    def _normalize_env_ids(self, env_ids: Sequence[int] | Tensor) -> Tensor:
        env_ids_t = (
            env_ids
            if isinstance(env_ids, torch.Tensor)
            else torch.as_tensor(list(env_ids), dtype=torch.int64)
        )
        return env_ids_t.to(device=self._state_device, dtype=torch.int64)

    def _get_step_offsets(self, batch_size: int) -> Tensor:
        step_offsets = self._step_offsets_cache.get(batch_size)
        if step_offsets is None:
            step_offsets = torch.arange(
                batch_size, dtype=torch.int64, device=self._state_device
            )
            self._step_offsets_cache[batch_size] = step_offsets
        return step_offsets

    def _choose_new_ranks(self, env_ids: Tensor) -> Tensor:
        env_ids = self._normalize_env_ids(env_ids)
        n = int(env_ids.numel())
        if n == 0:
            return torch.empty((0,), dtype=torch.int64, device=env_ids.device)

        if self.reset_schedule == ResetSchedule.RANDOM:
            return torch.randint(
                low=0,
                high=self.num_trajectories,
                size=(n,),
                dtype=torch.int64,
                device=env_ids.device,
            )

        if self.reset_schedule == ResetSchedule.SEQUENTIAL:
            base = self._next_rank
            ranks = (
                torch.arange(n, dtype=torch.int64, device=env_ids.device) + base
            ) % self.num_trajectories
            self._next_rank = int((base + n) % self.num_trajectories)
            return ranks

        if self.reset_schedule == ResetSchedule.ROUND_ROBIN:
            return (self.env_traj_rank[env_ids] + 1) % self.num_trajectories

        if self.reset_schedule == ResetSchedule.CUSTOM:
            if self.custom_reset_fn is None:
                raise ValueError(
                    "reset_schedule=CUSTOM but custom_reset_fn is None. "
                    "Provide custom_reset_fn(env_ids, num_trajectories)->ranks."
                )
            ranks = self.custom_reset_fn(env_ids, self.num_trajectories)
            ranks = torch.as_tensor(ranks, dtype=torch.int64, device=env_ids.device)
            if ranks.shape != (n,):
                raise ValueError(
                    "custom_reset_fn must return a 1D tensor of shape [len(env_ids)]"
                )
            return ranks % self.num_trajectories

        raise ValueError(f"Unknown reset_schedule: {self.reset_schedule}")

    def reset_envs(
        self,
        env_ids: Sequence[int] | Tensor,
        *,
        ranks: Optional[Tensor] = None,
        steps: Optional[Tensor | int] = None,
    ) -> None:
        """Reset selected envs: set step and (optionally) reassign traj_rank."""
        env_ids_t = self._normalize_env_ids(env_ids)
        if env_ids_t.numel() == 0:
            return

        if ranks is None:
            new_ranks = self._choose_new_ranks(env_ids_t)
        else:
            new_ranks = torch.as_tensor(
                ranks, dtype=torch.int64, device=env_ids_t.device
            )
            if new_ranks.shape != env_ids_t.shape:
                raise ValueError("ranks must have the same shape as env_ids")
            new_ranks = new_ranks % self.num_trajectories

        self.env_traj_rank[env_ids_t] = new_ranks
        if steps is None:
            self._set_env_steps(env_ids_t, self.reset_start_step)
        else:
            self._set_env_steps(env_ids_t, steps)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Reset envs %s -> ranks %s, start_steps %s",
                env_ids_t.tolist(),
                new_ranks.tolist(),
                self.env_step[env_ids_t].tolist(),
            )

    def set_env_cursor(
        self,
        *,
        env_ids: Sequence[int] | Tensor,
        ranks: Tensor,
        steps: Optional[Tensor | int] = None,
    ) -> None:
        """Set (rank, step) for a set of envs."""
        env_ids_t = self._normalize_env_ids(env_ids)
        ranks_t = torch.as_tensor(ranks, dtype=torch.int64, device=env_ids_t.device)
        if ranks_t.shape != env_ids_t.shape:
            raise ValueError("ranks must have the same shape as env_ids")
        self.env_traj_rank[env_ids_t] = ranks_t % self.num_trajectories

        if steps is None:
            self.env_step[env_ids_t] = 0
        else:
            self._set_env_steps(env_ids_t, steps)

    def _set_env_steps(self, env_ids_t: Tensor, steps: Tensor | int) -> None:
        env_ids_t = self._normalize_env_ids(env_ids_t)
        if isinstance(steps, int):
            steps_t = torch.full_like(env_ids_t, int(steps), dtype=torch.int64)
        else:
            steps_t = torch.as_tensor(steps, dtype=torch.int64, device=env_ids_t.device)
            if steps_t.shape != env_ids_t.shape:
                raise ValueError("steps must have the same shape as env_ids")
        max_step = (self._length[self.env_traj_rank[env_ids_t]] - 1).clamp(min=0)
        self.env_step[env_ids_t] = _clamp_env_steps_impl(steps_t, max_step)

    def _set_reference_fields(
        self,
        td: TensorDict,
        *,
        qpos_key: str,
        qvel_key: str,
        output_prefix: tuple[str, ...],
        use_buffers: bool,
    ) -> None:
        qpos = td.get(qpos_key)
        qvel = td.get(qvel_key)
        if qpos is None or qvel is None:
            return

        (
            root_pos,
            root_quat,
            root_lin_vel,
            root_ang_vel,
            joint_pos_out,
            joint_vel_out,
        ) = _extract_reference_fields_impl(
            qpos,
            qvel,
            self._nan_joint_template.to(device=qpos.device, dtype=qpos.dtype),
            self.ref_to_target_map.to(device=qpos.device),
            self.target_to_ref_map.to(device=qpos.device),
        )
        make_key = (
            (lambda name: name)
            if len(output_prefix) == 0
            else (lambda name: (*output_prefix, name))
        )

        if (
            not output_prefix
            and use_buffers
            and qpos.ndim == 2
            and qpos.shape[0] == self.num_envs
        ):
            self.root_pos.copy_(root_pos)
            self.root_quat.copy_(root_quat)
            self.root_lin_vel.copy_(root_lin_vel)
            self.root_ang_vel.copy_(root_ang_vel)
            self.joint_pos.copy_(joint_pos_out)
            self.joint_vel.copy_(joint_vel_out)

            td.set("root_pos", self.root_pos)
            td.set("root_quat", self.root_quat)
            td.set("root_lin_vel", self.root_lin_vel)
            td.set("root_ang_vel", self.root_ang_vel)
            td.set("joint_pos", self.joint_pos)
            td.set("joint_vel", self.joint_vel)
            return

        td.set(make_key("root_pos"), root_pos)
        td.set(make_key("root_quat"), root_quat)
        td.set(make_key("root_lin_vel"), root_lin_vel)
        td.set(make_key("root_ang_vel"), root_ang_vel)
        td.set(make_key("joint_pos"), joint_pos_out)
        td.set(make_key("joint_vel"), joint_vel_out)

    def _attach_reference_fields(
        self, td: TensorDict, *, use_buffers: bool
    ) -> TensorDict:
        if td.get("qpos") is None or td.get("qvel") is None:
            raise AssertionError("qpos and qvel must be present in the reference data")

        self._set_reference_fields(
            td,
            qpos_key="qpos",
            qvel_key="qvel",
            output_prefix=(),
            use_buffers=use_buffers,
        )
        if self.has_next_reference:
            self._set_reference_fields(
                td,
                qpos_key="next_qpos",
                qvel_key="next_qvel",
                output_prefix=("next",),
                use_buffers=False,
            )
        return td

    def set_reconstructed_action_targets(self, action_targets: Tensor | None) -> None:
        """Register cached reconstructed action targets in target-joint order."""
        self._reconstructed_action_targets = action_targets

    def get_reconstructed_action_targets(self, global_indices: Tensor) -> Tensor | None:
        """Fetch cached reconstructed action targets for replay-buffer indices."""
        if self._reconstructed_action_targets is None:
            return None
        index = torch.as_tensor(
            global_indices,
            dtype=torch.int64,
            device=self._reconstructed_action_targets.device,
        )
        return self._reconstructed_action_targets.index_select(
            0, index.reshape(-1)
        ).reshape(index.shape + (self._reconstructed_action_targets.shape[-1],))

    def sample_random_transitions(
        self,
        batch_size: int,
    ) -> tuple[TensorDict, Tensor, Tensor]:
        """Sample random transitions without advancing any environment cursor."""
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")

        env_ids_tm = torch.randint(
            low=0,
            high=self.num_envs,
            size=(batch_size,),
            device=self._state_device,
            dtype=torch.int64,
        )
        traj_ranks = self.env_traj_rank[env_ids_tm]
        lengths = self._length[traj_ranks].clamp(min=1)
        random_steps = torch.floor(
            torch.rand(batch_size, device=self._state_device)
            * lengths.to(dtype=torch.float32)
        ).to(dtype=torch.int64)
        global_indices = get_global_index(
            traj_ranks, self._start, self._end, random_steps
        )

        reference = self.rb[global_indices.to(device=self._storage_device)]
        if self._device is not None:
            reference = reference.to(self._device)
        reference = self._attach_reference_fields(reference, use_buffers=False)
        return reference, env_ids_tm, global_indices

    def sample(
        self,
        env_ids: Sequence[int] | Tensor | None = None,
        *,
        advance: bool = True,
        use_buffers: bool = False,
    ) -> TensorDict:
        """Sample one transition per env (or subset) via direct storage indexing."""
        env_ids_t = (
            self._all_env_ids if env_ids is None else self._normalize_env_ids(env_ids)
        )

        r = self.env_traj_rank[env_ids_t]
        step = self.env_step[env_ids_t]
        idx = get_global_index(r, self._start, self._end, step)

        td = self.rb[idx.to(device=self._storage_device)]
        if self._device is not None:
            td = td.to(self._device)

        if advance:
            self._advance_steps(env_ids_t)

        logger.debug(
            "Sampled %s envs (advance=%s)",
            int(env_ids_t.numel()),
            advance,
        )
        return self._attach_reference_fields(td, use_buffers=use_buffers)

    def sample_slice(
        self,
        batch_size: int,
        env_ids: Sequence[int] | Tensor | None = None,
        *,
        start_steps: Optional[Tensor] = None,
        mode: str = "contiguous",
        advance: bool = False,
    ) -> TensorDict:
        """Sample a batch from each env's assigned trajectory.

        If mode="contiguous" and start_steps is None, start positions are sampled
        uniformly per env. If mode="independent", each step is sampled uniformly.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if mode not in {"contiguous", "independent"}:
            raise ValueError("mode must be 'contiguous' or 'independent'")

        env_ids_t = (
            self._all_env_ids if env_ids is None else self._normalize_env_ids(env_ids)
        )

        r = self.env_traj_rank[env_ids_t]
        length = self._length[r].clamp(min=1)
        max_step = length - 1

        if mode == "contiguous":
            if torch.any(length < batch_size):
                logger.warning(
                    "sample_slice batch_size=%s exceeds some trajectory lengths; clamping to final steps.",
                    batch_size,
                )
            if start_steps is None:
                max_start = (length - batch_size).clamp(min=0)
                rand = torch.rand_like(max_start, dtype=torch.float32)
                start_steps_t = torch.floor(
                    rand * (max_start.to(dtype=rand.dtype) + 1)
                ).to(dtype=torch.int64)
            else:
                start_steps_t = torch.as_tensor(
                    start_steps, dtype=torch.int64, device=env_ids_t.device
                )
                if start_steps_t.shape != env_ids_t.shape:
                    raise ValueError("start_steps must have the same shape as env_ids")
                start_steps_t = start_steps_t.clamp(min=0)
            start_steps_t = torch.minimum(start_steps_t, max_step)
            local_steps = _make_contiguous_local_steps_impl(
                start_steps_t,
                self._get_step_offsets(batch_size),
                max_step,
            )
        else:
            if start_steps is None:
                rand = torch.rand(
                    (int(env_ids_t.numel()), batch_size),
                    device=length.device,
                    dtype=torch.float32,
                )
                local_steps = torch.floor(
                    rand * length.to(dtype=rand.dtype).unsqueeze(1)
                ).to(dtype=torch.int64)
            else:
                start_steps_t = torch.as_tensor(
                    start_steps, dtype=torch.int64, device=env_ids_t.device
                )
                if start_steps_t.shape == env_ids_t.shape:
                    local_steps = start_steps_t.unsqueeze(1).expand(-1, batch_size)
                elif start_steps_t.shape == (int(env_ids_t.numel()), batch_size):
                    local_steps = start_steps_t
                else:
                    raise ValueError(
                        "start_steps must have shape [num_envs] or [num_envs, batch_size] for mode='independent'"
                    )
            local_steps = _clamp_independent_local_steps_impl(local_steps, max_step)

        idx = get_global_index(r, self._start, self._end, local_steps)
        td = self.rb[idx.to(device=self._storage_device)]
        if self._device is not None:
            td = td.to(self._device)

        if advance:
            logger.warning("advance=True is not supported for sample_slice; ignoring.")
        logger.debug(
            "Slice-sampled batch_size=%s for %s envs (mode=%s, advance=%s)",
            batch_size,
            int(env_ids_t.numel()),
            mode,
            advance,
        )
        return self._attach_reference_fields(td, use_buffers=False)

    def _advance_steps(self, env_ids: Tensor) -> None:
        env_ids = self._normalize_env_ids(env_ids)
        length = self._length[self.env_traj_rank[env_ids]].clamp(min=1)
        step = self.env_step[env_ids]

        if self.wrap_steps:
            self.env_step[env_ids] = _advance_steps_wrap_impl(step, length)
        else:
            self.env_step[env_ids] = _advance_steps_clamp_impl(step, length)
