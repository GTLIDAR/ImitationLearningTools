from __future__ import annotations

import pytest
import torch

from iltools.datasets.reset_sampling import (
    SonicAdaptiveResetSampler,
    StartFrameSampler,
)


# ---------------------------------------------------------------------------
# SonicAdaptiveResetSampler (ported from the Isaac Lab extension).
# ---------------------------------------------------------------------------


def test_fixed_motion_local_bins_and_sequence_length_weights() -> None:
    sampler = SonicAdaptiveResetSampler(
        torch.tensor([120, 55]),
        bin_size=50,
        pre_failure_sample_window=0,
    )

    assert sampler.bins.tolist() == [
        [0, 0, 50],
        [0, 50, 100],
        [0, 100, 120],
        [1, 0, 50],
        [1, 50, 55],
    ]
    expected_weights = torch.tensor([50 / 3, 50 / 3, 20 / 3, 50 / 2, 5 / 2])
    expected_weights /= expected_weights.sum()
    torch.testing.assert_close(sampler.sampling_probabilities(), expected_weights)


def test_visit_and_failure_statistics_match_sonic_updates() -> None:
    sampler = SonicAdaptiveResetSampler(
        torch.tensor([120, 55]),
        bin_size=50,
        pre_failure_sample_window=0,
    )
    sampler.record_visits(
        torch.tensor([0, 0, 1]),
        torch.tensor([10, 110, 54]),
    )
    sampler.record_failures(
        torch.tensor([0, 1]),
        torch.tensor([110, 54]),
    )

    expected_visits = torch.tensor([1.0 + 1 / 50, 1.0, 1.0 + 1 / 20, 1.0, 1.0 + 1 / 5])
    expected_failures = torch.tensor([1.0, 1.0, 2.0, 1.0, 2.0])
    torch.testing.assert_close(sampler.num_visits, expected_visits)
    torch.testing.assert_close(sampler.num_failures, expected_failures)


def test_failure_rates_change_motion_and_bin_sampling_jointly() -> None:
    sampler = SonicAdaptiveResetSampler(
        torch.tensor([100, 100]),
        bin_size=50,
        uniform_sampling_rate=0.1,
        pre_failure_sample_window=0,
    )
    sampler.num_visits.fill_(100.0)
    sampler.num_failures.fill_(1.0)
    sampler.num_failures[3] = 90.0

    probabilities = sampler.sampling_probabilities()
    assert probabilities[3] > probabilities[0] * 20
    assert probabilities[2:].sum() > probabilities[:2].sum()


def test_random_full_trajectory_starts_apply_sonic_lead_in() -> None:
    lengths = torch.tensor([500, 260])
    raw_sampler = SonicAdaptiveResetSampler(
        lengths,
        bin_size=50,
        pre_failure_sample_window=0,
    )
    lead_in_sampler = SonicAdaptiveResetSampler(
        lengths,
        bin_size=50,
        pre_failure_sample_window=200,
    )

    torch.manual_seed(1234)
    raw_ranks, raw_steps = raw_sampler.sample(4096)
    torch.manual_seed(1234)
    lead_in_ranks, lead_in_steps = lead_in_sampler.sample(4096)

    torch.testing.assert_close(lead_in_ranks, raw_ranks)
    assert torch.all(lead_in_steps <= raw_steps)
    assert torch.all(raw_steps - lead_in_steps <= 199)
    assert torch.all(lead_in_steps >= 0)
    assert torch.all(lead_in_steps < lengths.index_select(0, lead_in_ranks))
    assert torch.unique(lead_in_steps).numel() > 100
    assert torch.any(lead_in_steps > 200)
    assert torch.any(lead_in_steps == 0)


def test_sonic_dedicated_generator_is_independent_of_global_rng() -> None:
    lengths = torch.tensor([500, 260])

    def _sample(global_seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(global_seed)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(77)
        sampler = SonicAdaptiveResetSampler(lengths, generator=generator)
        return sampler.sample(512)

    first_ranks, first_steps = _sample(1)
    second_ranks, second_steps = _sample(9999)
    torch.testing.assert_close(first_ranks, second_ranks)
    torch.testing.assert_close(first_steps, second_steps)


def test_sonic_probability_snapshot_is_not_changed_by_later_failures() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(19)
    sampler = SonicAdaptiveResetSampler(
        torch.tensor([100, 100]),
        pre_failure_sample_window=0,
        generator=generator,
    )
    snapshot = sampler.sampling_probabilities().clone()
    sampler.num_visits.fill_(100.0)
    sampler.num_failures.fill_(1.0)
    sampler.num_failures[-1] = 100.0

    ranks, steps = sampler.sample(4096, probabilities=snapshot)
    # The frozen initial distribution is balanced across the equal-length
    # motions even though the live distribution now overwhelmingly favors 1.
    first_fraction = (ranks == 0).float().mean()
    assert 0.45 < first_fraction < 0.55
    assert torch.all(steps >= 0)
    assert torch.all(steps < 100)

    with pytest.raises(ValueError, match="one entry per SONIC bin"):
        sampler.sample(1, probabilities=torch.ones(3))


# ---------------------------------------------------------------------------
# StartFrameSampler: fixed / random modes.
# ---------------------------------------------------------------------------


def test_fixed_mode_returns_fixed_step_clamped_to_lengths() -> None:
    sampler = StartFrameSampler(
        torch.tensor([100, 10]),
        mode="fixed",
        fixed_step=50,
    )
    steps = sampler.sample_steps(torch.tensor([0, 1]))
    assert steps.tolist() == [50, 9]  # second trajectory is only 10 frames long


def test_fixed_mode_defaults_to_zero() -> None:
    sampler = StartFrameSampler(torch.tensor([100, 100]))
    assert sampler.mode == "fixed"
    assert sampler.sample_steps(torch.tensor([0, 1])).tolist() == [0, 0]


def test_random_mode_stays_within_inclusive_bounds() -> None:
    sampler = StartFrameSampler(
        torch.tensor([100, 100]),
        mode="random",
        random_step_min=10,
        random_step_max=20,
    )
    torch.manual_seed(0)
    steps = sampler.sample_steps(torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]))
    assert torch.all(steps >= 10)
    assert torch.all(steps <= 20)
    assert torch.unique(steps).numel() > 1


def test_random_mode_with_single_value_is_fixed() -> None:
    sampler = StartFrameSampler(
        torch.tensor([100]),
        mode="random",
        random_step_min=7,
        random_step_max=7,
    )
    assert sampler.sample_steps(torch.tensor([0])).tolist() == [7]


def test_start_frame_dedicated_generator_is_independent_of_global_rng() -> None:
    ranks = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    def _sample(global_seed: int) -> torch.Tensor:
        torch.manual_seed(global_seed)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(91)
        sampler = StartFrameSampler(
            torch.tensor([100, 100]),
            mode="random",
            random_step_min=10,
            random_step_max=20,
            generator=generator,
        )
        return sampler.sample_steps(ranks)

    torch.testing.assert_close(_sample(2), _sample(2000))


def test_empty_rank_batch_returns_empty() -> None:
    sampler = StartFrameSampler(torch.tensor([100]))
    assert sampler.sample_steps(torch.empty(0, dtype=torch.long)).numel() == 0


# ---------------------------------------------------------------------------
# StartFrameSampler: adaptive mode with a pluggable weight function.
# ---------------------------------------------------------------------------


def _triangle_weights(ranks: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
    """Peak in the middle of each trajectory, zero at both ends."""
    lengths = torch.tensor([100, 60], device=ranks.device).index_select(0, ranks)
    peak = (lengths - 1) // 2
    return (peak - (steps - peak).abs()).clamp_min(0).to(torch.float32)


def test_adaptive_mode_samples_from_provided_weight_function() -> None:
    sampler = StartFrameSampler(
        torch.tensor([100, 60]),
        mode="adaptive",
        weight_fn=_triangle_weights,
    )
    torch.manual_seed(7)
    ranks = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    steps = sampler.sample_steps(ranks)
    assert torch.all(steps >= 0)
    assert torch.all(steps < torch.tensor([100, 60]).index_select(0, ranks))
    # The triangle peaks mid-trajectory, so samples should cluster there.
    assert steps.float().mean() > 20
    assert torch.all(steps != 0) and torch.all(steps != 99)


def test_adaptive_mode_requires_weight_fn() -> None:
    with pytest.raises(ValueError, match="weight_fn"):
        StartFrameSampler(torch.tensor([100]), mode="adaptive")


def test_adaptive_mode_rejects_weight_fn_in_other_modes() -> None:
    with pytest.raises(ValueError, match="only used in adaptive"):
        StartFrameSampler(
            torch.tensor([100]),
            mode="fixed",
            weight_fn=_triangle_weights,
        )


def test_adaptive_mode_falls_back_to_uniform_for_zero_weight_rows() -> None:
    def zero_for_first(ranks: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        weights = torch.ones(ranks.numel(), dtype=torch.float32, device=ranks.device)
        weights[ranks == 0] = 0.0
        return weights

    sampler = StartFrameSampler(
        torch.tensor([100, 60]),
        mode="adaptive",
        weight_fn=zero_for_first,
    )
    torch.manual_seed(3)
    steps = sampler.sample_steps(torch.tensor([0, 1, 0, 1]))
    assert torch.all(steps >= 0)
    assert torch.all(
        steps < torch.tensor([100, 60]).index_select(0, torch.tensor([0, 1, 0, 1]))
    )


def test_adaptive_mode_sanitizes_non_finite_weights() -> None:
    def nan_weights(ranks: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        return torch.full((ranks.numel(),), float("nan"), device=ranks.device)

    sampler = StartFrameSampler(
        torch.tensor([100]),
        mode="adaptive",
        weight_fn=nan_weights,
    )
    # Non-finite weights are zeroed, then the zero-row fallback makes the
    # distribution uniform, so sampling still succeeds.
    steps = sampler.sample_steps(torch.tensor([0, 0, 0]))
    assert torch.all(steps >= 0) and torch.all(steps < 100)


# ---------------------------------------------------------------------------
# StartFrameSampler(adaptive, weight_fn=SonicAdaptiveResetSampler) reproduces
# the SONIC frame distribution (statistical check).
# ---------------------------------------------------------------------------


def test_sonic_sampler_as_weight_function_matches_sonic_distribution() -> None:
    lengths = torch.tensor([200, 120])
    sonic = SonicAdaptiveResetSampler(
        lengths,
        bin_size=50,
        pre_failure_sample_window=0,
        uniform_sampling_rate=0.1,
    )
    # Push failures into one bin so the distribution is clearly non-uniform.
    sonic.num_visits.fill_(100.0)
    sonic.num_failures.fill_(1.0)
    sonic.num_failures[3] = 80.0

    # Standalone SONIC: (rank, step) pairs from the bin distribution.
    torch.manual_seed(42)
    sonic_ranks, sonic_steps = sonic.sample(200_000)

    # Generic adaptive sampler driven by the SONIC weight function.
    frame_sampler = StartFrameSampler(
        lengths,
        mode="adaptive",
        weight_fn=sonic,
    )
    torch.manual_seed(42)
    generic_steps = frame_sampler.sample_steps(sonic_ranks)

    # The generic path must reproduce the standalone frame distribution: both
    # are P(bin) spread uniformly within the bin (the lead-in is disabled).
    expected = torch.bincount(sonic_steps, minlength=int(lengths.max().item()))
    actual = torch.bincount(generic_steps, minlength=int(lengths.max().item()))
    expected = expected.to(torch.float32) / expected.sum()
    actual = actual.to(torch.float32) / actual.sum()
    # Per-frame tolerance; distributions are close on a 200k draw.
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-2)


def test_start_frame_sampler_validates_mode_and_bounds() -> None:
    with pytest.raises(ValueError, match="Unsupported starting-frame mode"):
        StartFrameSampler(torch.tensor([100]), mode="bogus")
    with pytest.raises(ValueError, match="fixed_step must be >= 0"):
        StartFrameSampler(torch.tensor([100]), fixed_step=-1)
    with pytest.raises(ValueError, match="random_step_max must be >= random_step_min"):
        StartFrameSampler(
            torch.tensor([100]), mode="random", random_step_min=5, random_step_max=2
        )
    with pytest.raises(ValueError, match="at least one trajectory"):
        StartFrameSampler(torch.empty(0, dtype=torch.long))
    with pytest.raises(ValueError, match="out-of-range"):
        StartFrameSampler(torch.tensor([100])).sample_steps(torch.tensor([3]))


def test_adaptive_frames_follow_recorded_failures() -> None:
    """Simulate the env pattern: schedule-driven ranks + SONIC adaptive frames.

    Ranks come from a trajectory manager reset schedule; starting frames come
    from StartFrameSampler(mode='adaptive', weight_fn=sonic). Recording
    failures in one bin must shift the sampled frame distribution toward it.
    """
    lengths = torch.tensor([200, 200])
    sonic = SonicAdaptiveResetSampler(lengths, bin_size=50, pre_failure_sample_window=0)
    frame_sampler = StartFrameSampler(
        lengths, mode="adaptive", weight_fn=sonic, device="cpu"
    )

    def draw(seed: int, n: int = 50_000) -> torch.Tensor:
        torch.manual_seed(seed)
        # Ranks alternate 0/1 (round-robin-like schedule).
        ranks = torch.arange(n, dtype=torch.long) % 2
        steps = frame_sampler.sample_steps(ranks)
        # One visit per sampled frame, mirroring _record_adaptive_failure_reset_visits.
        sonic.record_visits(ranks, steps)
        return steps

    before = draw(0)
    # Failures concentrate in trajectory 0's third bin (frames 100..149).
    sonic.record_failures(
        torch.zeros(50, dtype=torch.long), torch.arange(100, 150, dtype=torch.long)
    )
    after = draw(1)

    frac_before = ((before >= 100) & (before < 150)).float().mean().item()
    frac_after = ((after >= 100) & (after < 150)).float().mean().item()
    assert frac_after > frac_before
    assert frac_after > 0.2
