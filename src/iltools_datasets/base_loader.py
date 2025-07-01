import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from torch.utils.data import Dataset as TorchDataset

from iltools_core.metadata_schema import DatasetMeta
from iltools_core.trajectory import Trajectory


class BaseTrajectoryLoader(ABC):
    """
    Abstract base class for all trajectory loaders (unified interface).
    """

    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def __getitem__(self, idx: int) -> Any: ...
    @property
    @abstractmethod
    def metadata(self) -> DatasetMeta: ...

    def get_frequency_info(self) -> Dict[str, Any]:
        """
        Get frequency-related information from the loader.
        Can be overridden by subclasses to provide more specific info.
        """
        info = {}
        if hasattr(self, "original_freq"):
            info["original_frequency"] = self.original_freq
        if hasattr(self, "effective_freq"):
            info["effective_frequency"] = self.effective_freq
        if hasattr(self, "default_control_freq"):
            info["default_control_freq"] = self.default_control_freq
        return info

    def as_dataset(self, **kwargs) -> "BaseTrajectoryDataset":
        raise NotImplementedError


class BaseTrajectoryDataset(TorchDataset, ABC):
    """
    Abstract base class for all trajectory datasets (PyTorch, Zarr, etc).
    """

    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]: ...
    @property
    @abstractmethod
    def metadata(self) -> DatasetMeta: ...
    def get_window(self, idx: int, start: int, length: int) -> Dict[str, Any]:
        raise NotImplementedError

    def collate_batch(self, indices: List[int]) -> Dict[str, Any]:
        raise NotImplementedError

    def as_loader(self, **kwargs) -> BaseTrajectoryLoader:
        raise NotImplementedError
