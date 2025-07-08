from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset as TorchDataset

from iltools_core.metadata_schema import DatasetMeta


class BaseLoader(ABC):
    """
    Abstract base class for all trajectory loaders. This is the main interface for
    loading trajectories from a dataset. The loader is responsible for loading the
    trajectories from the dataset and prepare for saving onto disk using zarr.
    If the dataset is small, the loader can also be used as a dataset, by converting
    it to a dataset using the as_dataset method.
    """

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of motions in the dataset.
        """
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """
        Returns a motion from the dataset.
        """

    @property
    @abstractmethod
    def metadata(self) -> DatasetMeta:
        """
        Returns the metadata of the dataset.
        """
        ...

    @abstractmethod
    def save(self, path: str, **kwargs) -> None:
        """
        Saves the dataset to a directory.
        """
        ...

    @abstractmethod
    def as_dataset(self, **kwargs) -> "BaseDataset":
        """
        Returns a dataset from the loader.
        """


class BaseDataset(TorchDataset, ABC):
    """
    Abstract base class for all trajectory datasets (PyTorch, Zarr, etc).
    We assume that trajectory data is accessed in a lazy manner, i.e. the data is not
    loaded into memory until it is needed. This is useful for large datasets where
    the data does not fit into memory.
    By default, the dataset is organized as a nested dictionary of trajectories, where the
    keys are the trajectory groups and the values are the trajectories data. Each trajectory
    is a dictionary of observations, actions, rewards, and infos. E.g. trajectory group "walk"
    contains trajectory "walk_0", "walk_1", etc.
    """

    # @abstractmethod
    # def __len__(self) -> int:
    #     """
    #     Returns the number of trajectories in the dataset.
    #     """
    #     ...

    # @abstractmethod
    # def __getitem__(self, idx: int) -> Dict[str, Any]:
    #     """
    #     Returns a trajectory from the dataset.
    #     """
    #     ...

    # @property
    # @abstractmethod
    # def metadata(self) -> DatasetMeta:
    #     """
    #     Returns the metadata of the dataset.
    #     """
    #     ...

    # @abstractmethod
    # def get_window(self, idx: int, start: int, length: int) -> Dict[str, Any]:
    #     """
    #     Returns a window of a trajectory from the dataset.
    #     """
    #     ...

    # @abstractmethod
    # def as_loader(self, **kwargs) -> BaseLoader:
    #     """
    #     Returns a loader from the dataset.
    #     """
    #     ...
