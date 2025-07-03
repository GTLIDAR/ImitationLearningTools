from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union


class DatasetMeta(BaseModel, frozen=True):
    name: str
    source: str
    version: str = Field(..., description="Semantic version of raw dataset")
    citation: str
    num_trajectories: int
    observation_keys: List[str]
    action_keys: Optional[List[str]] = None
    trajectory_lengths: Union[int, List[int]]
    dt: Union[float, List[float]] = Field(
        ..., description="Time step(s) per trajectory (seconds)"
    )
