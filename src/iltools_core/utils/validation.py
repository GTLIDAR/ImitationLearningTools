
import numpy as np

def validate_trajectory(traj):
    assert np.isfinite(traj.states).all(), "Non‑finite in states"
    if traj.actions is not None:
        assert np.isfinite(traj.actions).all(), "Non‑finite in actions"
