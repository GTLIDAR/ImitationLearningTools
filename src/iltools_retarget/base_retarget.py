
from abc import ABC, abstractmethod
from iltools_core.trajectory import Trajectory

class BaseRetarget(ABC):
    """
    Abstract base class for all retargeting methods.
    """

    @abstractmethod
    def retarget(self, trajectory: Trajectory) -> Trajectory:
        """
        Retargets a trajectory to a new robot.
        """
        raise NotImplementedError
