"""Init for src/imtools_datasets."""

from .base_loader import BaseDataset, BaseLoader

try:
    from .loco_mujoco.loader import LocoMuJoCoLoader
except Exception:  # pragma: no cover
    # Optional dependency not installed; expose a placeholder for nicer error messages
    class LocoMuJoCoLoader:  # type: ignore
        def __init__(self, *_, **__):
            raise ImportError(
                "LocoMuJoCoLoader requires the 'loco-mujoco' extra. Install via: pip install -e .[loco-mujoco]"
            )


from .storage import VectorizedTrajectoryDataset
from .replay_memmap import (
    Segment,
    ExpertMemmapBuilder,
    build_trajectory_td,
    build_trajectory_td_from_components,
    concat_components,
)
from .replay_manager import (
    EnvAssignment,
    ExpertReplayManager,
    ExpertReplaySpec,
)
from .replay_export import build_replay_from_zarr
import logging

logging.basicConfig(level=logging.ERROR)
__all__ = [
    "BaseDataset",
    "BaseLoader",
    "LocoMuJoCoLoader",
    "VectorizedTrajectoryDataset",
    "Segment",
    "ExpertMemmapBuilder",
    "build_trajectory_td",
    "build_trajectory_td_from_components",
    "concat_components",
    "EnvAssignment",
    "ExpertReplayManager",
    "ExpertReplaySpec",
    "build_replay_from_zarr",
]
