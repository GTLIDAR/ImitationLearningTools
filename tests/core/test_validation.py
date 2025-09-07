import numpy as np
from iltools_core.utils.validation import validate_trajectory


class DummyTraj:
    def __init__(self, states, actions=None):
        self.states = states
        self.actions = actions


def test_validate_trajectory_passes_with_finite_values():
    traj = DummyTraj(states=np.zeros((5, 2)), actions=np.ones((5, 1)))
    # Should not raise
    validate_trajectory(traj)


def test_validate_trajectory_raises_on_nonfinite_states():
    traj = DummyTraj(states=np.array([[0.0, np.nan]]))
    try:
        validate_trajectory(traj)
        raised = False
    except AssertionError:
        raised = True
    assert raised


def test_validate_trajectory_raises_on_nonfinite_actions():
    traj = DummyTraj(states=np.zeros((2, 2)), actions=np.array([[np.inf]]))
    try:
        validate_trajectory(traj)
        raised = False
    except AssertionError:
        raised = True
    assert raised
