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
