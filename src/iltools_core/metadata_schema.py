from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any


class DatasetMeta(BaseModel, frozen=True):
    name: str
    source: str
    version: str = Field(..., description="Semantic version of raw dataset")
    citation: str
    num_trajectories: int
    keys: List[str]
    trajectory_lengths: Union[int, List[int]]
    dt: Union[float, List[float]] = Field(
        ..., description="Time step(s) per trajectory (seconds)"
    )
    joint_names: List[str]
    body_names: List[str]
    site_names: List[str]
    metadata: Optional[Dict[str, Any]]
