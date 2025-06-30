from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union
from torch.utils.data import Dataset as TorchDataset
from iltools_core.trajectory import Trajectory
import warnings


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
    def metadata(self) -> Dict[str, Any]: ...
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
    def metadata(self) -> Dict[str, Any]: ...
    def get_window(self, idx: int, start: int, length: int) -> Dict[str, Any]:
        raise NotImplementedError

    def collate_batch(self, indices: List[int]) -> Dict[str, Any]:
        raise NotImplementedError

    def as_loader(self, **kwargs) -> BaseTrajectoryLoader:
        raise NotImplementedError
