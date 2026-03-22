"""
Submission template (NO weights required).

Use this template if your agent does not use any trained model.
Do NOT load torch or any weight files.

The action is decided directly using simple logic or randomness.

The evaluator will import this file and call `policy(obs, rng)`.
"""

from typing import Sequence
import numpy as np

# All possible actions
ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")


def policy(obs, rng):
    left  = obs[0] or obs[1] or obs[2] or obs[3]
    fwd   = obs[4] or obs[5] or obs[6] or obs[7] or obs[8] or obs[9] or obs[10] or obs[11]
    right = obs[12] or obs[13] or obs[14] or obs[15]
    ir    = obs[16]
    stuck = obs[17]

    if stuck:
        return rng.choice(["L45", "R45", "L22", "R22"])  # escape
    if ir:
        return "FW"   # push it!
    if fwd:
        return "FW"   # box is ahead — go for it
    if left and not right:
        return "L22"  # steer left toward box
    if right and not left:
        return "R22"  # steer right toward box
    # No signal — search by sweeping forward with small random turns
    return rng.choice(["FW", "FW", "FW", "L22", "R22"], p=[0.6, 0.1, 0.1, 0.1, 0.1])