import logging
from abc import ABC, abstractmethod
from typing import Any

from iltools.core.metadata_schema import DatasetMeta


class BaseLoader(ABC):
    """
    Abstract base class for all trajectory loaders. This is the main interface for
    loading trajectories from a dataset. The loader is responsible for loading the
    trajectories from the dataset and prepare for saving onto disk using zarr.
    If the dataset is small, the loader can also be used as a dataset, by converting
    it to a dataset using the as_dataset method.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    @abstractmethod
    def metadata(self) -> DatasetMeta:
        """
        Returns the metadata of the dataset.
        """
        ...

    @abstractmethod
    def _get_trajectories(
        self, **kwargs: Any
    ) -> tuple[list[dict], dict[str, dict[str, dict[str, Any]]]]:
        """
        Returns the trajectories and motion information from the dataset.
        The trajectories are a list of dictionaries, each containing the information about a trajectory.
        The motion information is a dictionary of dictionaries, each containing the information about a motion.
        """
        ...
