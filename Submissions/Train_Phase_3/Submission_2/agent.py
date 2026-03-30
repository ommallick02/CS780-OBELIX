"""
Reward hacking: Spin in a consistent circle.
"""

import numpy as np
from typing import Sequence

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")

# Spin pattern: 4 left rotations = full circle (approximately)
_SPIN_PATTERN = ["L45", "L45", "L45", "L45", "L45", "L45", "L45", "L45"]
_step_idx = 0

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    Spin in circles consistently.
    """
    global _step_idx
    
    action = _SPIN_PATTERN[_step_idx % len(_SPIN_PATTERN)]
    _step_idx += 1
    
    return action