"""Reset starting-frame and trajectory sampling for imitation environments.

Two pieces live here:

- :class:`StartFrameSampler` -- the generic way to pick a starting *frame*
  inside a trajectory when an environment resets. Trajectory *selection* is
  the trajectory manager's job (``ResetSchedule`` in
  ``iltools.datasets.manager``); this sampler only answers "given that an env
  will now follow trajectory rank ``r``, which local frame does it start at?".
  Three modes are supported:

  - ``"fixed"``: always start at ``fixed_step``.
  - ``"random"``: start uniformly in ``[random_step_min, random_step_max]``
    (inclusive).
  - ``"adaptive"``: start proportionally to a caller-supplied weight
    function ``weight_fn(ranks, steps) -> weights``. The weight function is a
    plain callable, so iltools users can plug in any policy -- SONIC's
    failure-rate bin weights (:class:`SonicAdaptiveResetSampler`), a learned
    saliency map, a fixed per-frame distribution, etc.

- :class:`SonicAdaptiveResetSampler` -- the failure-aware motion/frame
  sampler matching SONIC's public motion library, moved here from the Isaac
  Lab extension. It tracks per-bin failure/visit statistics and can be used
  either standalone (``sample(count)`` returns ranks *and* frames jointly) or
  as the ``weight_fn`` for :class:`StartFrameSampler` (its ``weights`` /
  ``__call__`` returns the equivalent per-frame weights).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

# A weight function maps a batch of (trajectory_rank, frame_step) pairs to a
# non-negative weight of the same shape. Larger weight => more likely to be
# sampled as a reset starting frame.
WeightFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class StartFrameSampler:
    """Sample reset starting frames for a batch of trajectory ranks.

    Trajectory selection stays with the trajectory manager's reset schedule;
    this class only decides the local starting frame inside each selected
    trajectory. Modes:

    - ``"fixed"``: every reset starts at ``fixed_step``.
    - ``"random"``: every reset starts uniformly in
      ``[random_step_min, random_step_max]`` (inclusive).
    - ``"adaptive"``: every reset starts proportionally to the caller's
      ``weight_fn``, a callable ``weight_fn(ranks, steps) -> weights``. The
      weights may depend on anything (observed failure rates, saliency,
      per-frame priors), are normalized per trajectory, and sampling falls
      back to uniform within a trajectory whose weights are all zero so it can
      never dead-end.

    All returned steps are clamped to ``[0, trajectory_length - 1]``.
    """

    FIXED = "fixed"
    RANDOM = "random"
    ADAPTIVE = "adaptive"
    MODES = (FIXED, RANDOM, ADAPTIVE)

    def __init__(
        self,
        trajectory_lengths: torch.Tensor,
        *,
        mode: str = FIXED,
        fixed_step: int = 0,
        random_step_min: int = 0,
        random_step_max: int = 0,
        weight_fn: WeightFunction | None = None,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        lengths = torch.as_tensor(
            trajectory_lengths, dtype=torch.long, device=trajectory_lengths.device
        ).reshape(-1)
        if lengths.numel() == 0:
            raise ValueError("trajectory_lengths must contain at least one trajectory.")
        if torch.any(lengths <= 0):
            raise ValueError("trajectory_lengths must all be positive.")

        mode = str(mode).strip().lower()
        if mode not in self.MODES:
            raise ValueError(
                f"Unsupported starting-frame mode {mode!r}; "
                f"expected one of {self.MODES}."
            )
        if int(fixed_step) < 0:
            raise ValueError("fixed_step must be >= 0.")
        if int(random_step_min) < 0:
            raise ValueError("random_step_min must be >= 0.")
        if int(random_step_max) < int(random_step_min):
            raise ValueError("random_step_max must be >= random_step_min.")
        if mode == self.ADAPTIVE and weight_fn is None:
            raise ValueError(
                "adaptive starting-frame mode requires a weight_fn callable."
            )
        if mode != self.ADAPTIVE and weight_fn is not None:
            raise ValueError("weight_fn is only used in adaptive starting-frame mode.")

        self._device = torch.device(device) if device is not None else lengths.device
        self.trajectory_lengths = lengths.to(self._device)
        self.mode = mode
        self.fixed_step = int(fixed_step)
        self.random_step_min = int(random_step_min)
        self.random_step_max = int(random_step_max)
        self.weight_fn = weight_fn
        if generator is not None and torch.device(generator.device) != self._device:
            raise ValueError(
                "generator device must match the sampler device; got "
                f"{torch.device(generator.device)} and {self._device}."
            )
        self.generator = generator

    def _clamp_steps(self, ranks: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        max_steps = self.trajectory_lengths.index_select(0, ranks) - 1
        return torch.minimum(torch.maximum(steps, torch.zeros_like(steps)), max_steps)

    def sample_steps(self, trajectory_ranks: torch.Tensor) -> torch.Tensor:
        """Sample one local starting frame per requested trajectory rank.

        Args:
            trajectory_ranks: 1D tensor of trajectory ranks.

        Returns:
            A 1D ``torch.long`` tensor of local starting frames, one per rank.
        """
        ranks = torch.as_tensor(
            trajectory_ranks, dtype=torch.long, device=self._device
        ).reshape(-1)
        n = int(ranks.numel())
        if n == 0:
            return torch.empty(0, dtype=torch.long, device=self._device)
        if torch.any((ranks < 0) | (ranks >= self.trajectory_lengths.numel())):
            raise ValueError("trajectory_ranks contains an out-of-range value.")

        if self.mode == self.FIXED:
            steps = torch.full(
                (n,), self.fixed_step, dtype=torch.long, device=self._device
            )
        elif self.mode == self.RANDOM:
            if self.random_step_max > self.random_step_min:
                steps = torch.randint(
                    self.random_step_min,
                    self.random_step_max + 1,
                    (n,),
                    device=self._device,
                    dtype=torch.long,
                    generator=self.generator,
                )
            else:
                steps = torch.full(
                    (n,),
                    self.random_step_min,
                    dtype=torch.long,
                    device=self._device,
                )
        else:
            steps = self._sample_adaptive(ranks)
        return self._clamp_steps(ranks, steps)

    def _sample_adaptive(self, ranks: torch.Tensor) -> torch.Tensor:
        """Sample one starting frame per rank from the weight-function grid."""
        n = int(ranks.numel())
        lengths = self.trajectory_lengths.index_select(0, ranks)  # (n,)
        max_len = int(lengths.max().item())
        frame_grid = torch.arange(max_len, device=self._device)  # (max_len,)
        valid = frame_grid[None, :] < lengths[:, None]  # (n, max_len)

        rank_grid = ranks[:, None].expand(n, max_len).reshape(-1)
        frame_grid_exp = frame_grid[None, :].expand(n, max_len).reshape(-1)
        assert self.weight_fn is not None
        weights = self.weight_fn(rank_grid, frame_grid_exp)
        weights = torch.as_tensor(weights, dtype=torch.float32, device=self._device)
        if tuple(weights.shape) != (n * max_len,):
            raise ValueError(
                "weight_fn must return one weight per (rank, step) pair; "
                f"expected shape {(n * max_len,)}, got {tuple(weights.shape)}."
            )
        weights = weights.reshape(n, max_len)
        # Never let non-finite user weights poison the multinomial.
        weights = torch.where(
            torch.isfinite(weights), weights, torch.zeros_like(weights)
        )
        weights = weights * valid.to(torch.float32)
        # A trajectory whose weights are all zero falls back to uniform over
        # its valid frames so sampling never dead-ends.
        zero_rows = weights.sum(dim=-1) <= 0.0
        if bool(zero_rows.any()):
            fallback = torch.zeros_like(weights)
            fallback[zero_rows] = valid[zero_rows].to(torch.float32)
            weights = torch.where(zero_rows[:, None], fallback, weights)
        probs = weights / weights.sum(dim=-1, keepdim=True)
        return torch.multinomial(probs, 1, generator=self.generator).squeeze(-1)


class SonicAdaptiveResetSampler:
    """Failure-aware motion/frame sampler matching SONIC's public motion library.

    Each trajectory is split into fixed-size frame bins. Sampling probabilities
    are based on the observed failure rate in each bin, blended with a uniform
    component and weighted so long sequences do not dominate solely because
    they contain more bins. A sampled frame is shifted backwards by a random
    lead-in so the policy can act before the difficult segment.

    The sampler can be used two ways:

    - Standalone: :meth:`sample` returns trajectory ranks and local starting
      frames jointly from the bin distribution (with the SONIC lead-in).
    - As a weight function: :meth:`weights` (also ``__call__``) returns, for
      any ``(rank, step)`` pairs, the per-frame weight
      ``P(bin) / len(bin)``. Passing it to
      ``StartFrameSampler(mode="adaptive", weight_fn=sampler)`` reproduces the
      exact SONIC frame distribution (minus the lead-in shift, which the
      standalone :meth:`sample` applies).
    """

    def __init__(
        self,
        trajectory_lengths: torch.Tensor,
        *,
        bin_size: int = 50,
        sequence_length_agnostic: bool = True,
        init_num_failures: float = 1.0,
        uniform_sampling_rate: float = 0.1,
        pre_failure_sample_window: int = 200,
        failure_rate_max_over_mean: float = 200.0,
        generator: torch.Generator | None = None,
    ) -> None:
        lengths = torch.as_tensor(
            trajectory_lengths,
            dtype=torch.long,
            device=trajectory_lengths.device,
        ).reshape(-1)
        if lengths.numel() == 0:
            raise ValueError("trajectory_lengths must contain at least one trajectory.")
        if torch.any(lengths <= 0):
            raise ValueError("trajectory_lengths must all be positive.")
        if int(bin_size) <= 0:
            raise ValueError("bin_size must be positive.")
        if float(init_num_failures) <= 0.0:
            raise ValueError("init_num_failures must be positive.")
        if not 0.0 <= float(uniform_sampling_rate) <= 1.0:
            raise ValueError("uniform_sampling_rate must be in [0, 1].")
        if int(pre_failure_sample_window) < 0:
            raise ValueError("pre_failure_sample_window must be >= 0.")
        if float(failure_rate_max_over_mean) <= 0.0:
            raise ValueError("failure_rate_max_over_mean must be positive.")

        self.device = lengths.device
        self.trajectory_lengths = lengths
        self.bin_size = int(bin_size)
        self.sequence_length_agnostic = bool(sequence_length_agnostic)
        self.uniform_sampling_rate = float(uniform_sampling_rate)
        self.pre_failure_sample_window = int(pre_failure_sample_window)
        self.failure_rate_max_over_mean = float(failure_rate_max_over_mean)
        if generator is not None and torch.device(generator.device) != self.device:
            raise ValueError(
                "generator device must match the sampler device; got "
                f"{torch.device(generator.device)} and {self.device}."
            )
        self.generator = generator

        bins: list[torch.Tensor] = []
        trajectory_bin_ids: list[torch.Tensor] = []
        next_bin_id = 0
        for trajectory_rank, trajectory_length in enumerate(lengths.tolist()):
            starts = torch.arange(
                0,
                trajectory_length,
                self.bin_size,
                device=self.device,
                dtype=torch.long,
            )
            ends = torch.minimum(
                starts + self.bin_size,
                torch.full_like(starts, trajectory_length),
            )
            ranks = torch.full_like(starts, trajectory_rank)
            bins.append(torch.stack((ranks, starts, ends), dim=-1))
            trajectory_bin_ids.append(
                torch.arange(
                    next_bin_id,
                    next_bin_id + starts.numel(),
                    device=self.device,
                    dtype=torch.long,
                )
            )
            next_bin_id += starts.numel()

        self.bins = torch.cat(bins, dim=0)
        self.num_bins = int(self.bins.shape[0])
        self.trajectory_bin_ids = trajectory_bin_ids
        self.first_bin_ids = torch.stack([ids[0] for ids in trajectory_bin_ids])
        self.bin_lengths = self.bins[:, 2] - self.bins[:, 1]
        peer_bin_counts = torch.empty(
            self.num_bins, device=self.device, dtype=torch.float32
        )
        for ids in trajectory_bin_ids:
            peer_bin_counts.index_fill_(0, ids, float(ids.numel()))

        self.bin_weights = self.bin_lengths.to(dtype=torch.float32)
        self.bin_weights /= self.bin_weights.mean()
        if self.sequence_length_agnostic:
            self.bin_weights /= peer_bin_counts

        initial_count = float(init_num_failures)
        self.num_failures = torch.full(
            (self.num_bins,), initial_count, device=self.device, dtype=torch.float32
        )
        self.num_visits = torch.full_like(self.num_failures, initial_count)

    def _bin_ids(
        self, trajectory_ranks: torch.Tensor, frame_steps: torch.Tensor
    ) -> torch.Tensor:
        ranks = torch.as_tensor(
            trajectory_ranks, device=self.device, dtype=torch.long
        ).reshape(-1)
        steps = torch.as_tensor(
            frame_steps, device=self.device, dtype=torch.long
        ).reshape(-1)
        if ranks.shape != steps.shape:
            raise ValueError(
                "trajectory_ranks and frame_steps must have matching shapes."
            )
        if torch.any((ranks < 0) | (ranks >= self.trajectory_lengths.numel())):
            raise ValueError("trajectory_ranks contains an out-of-range value.")
        max_steps = self.trajectory_lengths.index_select(0, ranks) - 1
        steps = torch.minimum(torch.maximum(steps, torch.zeros_like(steps)), max_steps)
        local_bins = torch.div(steps, self.bin_size, rounding_mode="floor")
        return self.first_bin_ids.index_select(0, ranks) + local_bins

    def record_visits(
        self, trajectory_ranks: torch.Tensor, frame_steps: torch.Tensor
    ) -> None:
        """Record one control-step visit per environment, normalized by bin length."""
        bin_ids = self._bin_ids(trajectory_ranks, frame_steps)
        counts = torch.bincount(bin_ids, minlength=self.num_bins).to(torch.float32)
        self.num_visits.add_(counts / self.bin_lengths.to(torch.float32))

    def record_failures(
        self, trajectory_ranks: torch.Tensor, frame_steps: torch.Tensor
    ) -> None:
        """Record terminal tracking failures at their motion-local frame bins."""
        if torch.as_tensor(trajectory_ranks).numel() == 0:
            return
        bin_ids = self._bin_ids(trajectory_ranks, frame_steps)
        self.num_failures.add_(
            torch.bincount(bin_ids, minlength=self.num_bins).to(torch.float32)
        )

    def sampling_probabilities(self) -> torch.Tensor:
        """Return the current SONIC-style global trajectory-bin distribution."""
        failure_rate = self.num_failures / self.num_visits
        upper_bound = failure_rate.mean() * self.failure_rate_max_over_mean
        clipped = torch.clamp(failure_rate, min=0.0, max=upper_bound)
        clipped_sum = clipped.sum()
        if not torch.isfinite(clipped_sum) or clipped_sum <= 0.0:
            failure_prob = torch.full_like(clipped, 1.0 / self.num_bins)
        else:
            failure_prob = clipped / clipped_sum
        uniform_prob = torch.full_like(failure_prob, 1.0 / self.num_bins)
        probabilities = (
            failure_prob * (1.0 - self.uniform_sampling_rate)
            + uniform_prob * self.uniform_sampling_rate
        )
        probabilities *= self.bin_weights
        return probabilities / probabilities.sum()

    def weights(
        self, trajectory_ranks: torch.Tensor, frame_steps: torch.Tensor
    ) -> torch.Tensor:
        """Per-frame sampling weights for ``StartFrameSampler`` adaptive mode.

        For a frame in bin ``b`` the weight is ``P(bin b) / len(bin b)`` --
        the SONIC bin distribution spread uniformly over the bin's frames -- so
        ``StartFrameSampler(mode="adaptive", weight_fn=sampler)`` reproduces
        the exact SONIC frame distribution (without the lead-in shift).
        """
        bin_ids = self._bin_ids(trajectory_ranks, frame_steps)
        bin_probs = self.sampling_probabilities()
        bin_lengths = self.bin_lengths
        return bin_probs.index_select(0, bin_ids) / bin_lengths.index_select(
            0, bin_ids
        ).to(dtype=torch.float32)

    def __call__(
        self, trajectory_ranks: torch.Tensor, frame_steps: torch.Tensor
    ) -> torch.Tensor:
        """Callable alias of :meth:`weights` for use as a ``weight_fn``."""
        return self.weights(trajectory_ranks, frame_steps)

    def sample(
        self,
        count: int,
        *,
        probabilities: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample trajectory ranks and local starts with SONIC's lead-in.

        ``probabilities`` may hold a caller-owned snapshot of the bin
        distribution. This makes the sampling-time contract explicit for
        asynchronous consumers instead of racing later failure updates.
        """
        count = int(count)
        if count < 0:
            raise ValueError("count must be >= 0.")
        if count == 0:
            empty = torch.empty(0, device=self.device, dtype=torch.long)
            return empty, empty
        if probabilities is None:
            probabilities = self.sampling_probabilities()
        else:
            probabilities = torch.as_tensor(
                probabilities, device=self.device, dtype=torch.float32
            )
            if probabilities.shape != (self.num_bins,):
                raise ValueError(
                    "probabilities must have one entry per SONIC bin; expected "
                    f"{(self.num_bins,)}, got {tuple(probabilities.shape)}."
                )
            if not torch.all(torch.isfinite(probabilities)):
                raise ValueError("probabilities must be finite")
            if torch.any(probabilities < 0) or probabilities.sum() <= 0:
                raise ValueError("probabilities must be non-negative with positive sum")
            probabilities = probabilities / probabilities.sum()
        sampled_bin_ids = torch.multinomial(
            probabilities,
            count,
            replacement=True,
            generator=self.generator,
        )
        sampled_bins = self.bins.index_select(0, sampled_bin_ids)
        trajectory_ranks = sampled_bins[:, 0]
        bin_starts = sampled_bins[:, 1]
        bin_ends = sampled_bins[:, 2]
        frame_steps = (
            torch.rand(count, device=self.device, generator=self.generator)
            * (bin_ends - bin_starts)
        ).floor().to(torch.long) + bin_starts
        if self.pre_failure_sample_window > 0:
            lead_in = torch.randint(
                self.pre_failure_sample_window,
                (count,),
                device=self.device,
                dtype=torch.long,
                generator=self.generator,
            )
            frame_steps = (frame_steps - lead_in).clamp_min(0)
        return trajectory_ranks, frame_steps
