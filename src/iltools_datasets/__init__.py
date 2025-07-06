"""Init for src/imtools_datasets."""

from .base_loader import BaseDataset, BaseLoader
from .loco_mujoco.loader import LocoMuJoCoLoader
from .storage import VectorizedTrajectoryDataset
import logging

logging.basicConfig(level=logging.ERROR)
__all__ = [
    "BaseDataset",
    "BaseLoader",
    "LocoMuJoCoLoader",
    "VectorizedTrajectoryDataset",
]
