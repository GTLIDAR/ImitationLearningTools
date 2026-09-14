"""Temporal post-processing that preserves already-qualified discrete poses."""

from __future__ import annotations

import numpy as np


def discrete_time_stretch_indices(
    frame_count: int,
    *,
    hold_frames: int,
) -> np.ndarray:
    """Return zero-order-hold indices for a slower discrete trajectory.

    Every transition is separated by ``hold_frames`` control intervals while
    the first and last authored samples remain exact.  Unlike linear or spline
    interpolation, this operation cannot introduce an unaudited configuration
    between collision-qualified states.
    """

    if int(frame_count) != frame_count or int(frame_count) < 2:
        raise ValueError("frame_count must be an integer of at least two.")
    if int(hold_frames) != hold_frames or int(hold_frames) < 1:
        raise ValueError("hold_frames must be a positive integer.")
    frame_count = int(frame_count)
    hold_frames = int(hold_frames)
    output_count = (frame_count - 1) * hold_frames + 1
    return np.arange(output_count, dtype=np.int64) // hold_frames


def stretch_discrete_samples(
    values: np.ndarray,
    *,
    hold_frames: int,
) -> np.ndarray:
    """Time-stretch an array along axis zero with exact zero-order holds."""

    samples = np.asarray(values)
    if samples.ndim < 1 or len(samples) < 2:
        raise ValueError("values must have at least two samples on axis zero.")
    indices = discrete_time_stretch_indices(
        len(samples),
        hold_frames=hold_frames,
    )
    return samples[indices]


__all__ = ["discrete_time_stretch_indices", "stretch_discrete_samples"]
