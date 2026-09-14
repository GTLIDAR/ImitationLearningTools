import numpy as np
import pytest

from iltools.retarget import (
    discrete_time_stretch_indices,
    stretch_discrete_samples,
)


def test_discrete_time_stretch_preserves_authored_states_and_endpoints() -> None:
    values = np.asarray([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]])

    indices = discrete_time_stretch_indices(3, hold_frames=3)
    stretched = stretch_discrete_samples(values, hold_frames=3)

    np.testing.assert_array_equal(indices, [0, 0, 0, 1, 1, 1, 2])
    np.testing.assert_array_equal(stretched, values[indices])
    np.testing.assert_array_equal(stretched[[0, -1]], values[[0, -1]])


def test_discrete_time_stretch_identity_preserves_values() -> None:
    values = np.asarray([1.0, 2.0, 3.0])

    stretched = stretch_discrete_samples(values, hold_frames=1)

    np.testing.assert_array_equal(stretched, values)


@pytest.mark.parametrize(
    ("frame_count", "hold_frames", "message"),
    [(1, 1, "frame_count"), (3, 0, "hold_frames"), (3.5, 1, "frame_count")],
)
def test_discrete_time_stretch_rejects_invalid_counts(
    frame_count: float, hold_frames: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        discrete_time_stretch_indices(frame_count, hold_frames=hold_frames)
