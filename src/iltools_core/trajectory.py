from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(slots=True)
class Trajectory:
    observations: Dict[str, np.ndarray]
    actions: Optional[Dict[str, np.ndarray]] = None
    rewards: Optional[np.ndarray] = None
    infos: Optional[Dict[str, Any]] = None
    dt: Optional[float] = None
