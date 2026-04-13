"""Dueling Double DQN (D3QN) with State Stacking."""

from __future__ import annotations
from typing import List, Optional
from collections import deque
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]
_STACK_SIZE: int = 4

class DuelingDQN(nn.Module):
    def __init__(self, in_dim: int = 18 * _STACK_SIZE, n_actions: int = 5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.value_head = nn.Linear(64, 1)
        self.advantage_head = nn.Linear(64, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.encoder(x)
        v = self.value_head(f)
        a = self.advantage_head(f)
        return v + a - a.mean(dim=-1, keepdim=True)

_model: Optional[DuelingDQN] = None
_frame_buffer: Optional[deque] = None
_last_action: Optional[int] = None
_repeat_count: int = 0
_MAX_REPEAT = 2
_CLOSE_Q_DELTA = 0.05

def _reset_buffer(obs: np.ndarray):
    global _frame_buffer
    _frame_buffer = deque(maxlen=_STACK_SIZE)
    for _ in range(_STACK_SIZE - 1):
        _frame_buffer.append(np.zeros(18, dtype=np.float32))
    _frame_buffer.append(obs.astype(np.float32).copy())

def _get_stacked() -> np.ndarray:
    return np.concatenate(list(_frame_buffer), axis=0)

def _load_once():
    global _model
    if _model is not None:
        return
    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError("weights.pth not found")
    m = DuelingDQN()
    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    m.eval()
    _model = m

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _frame_buffer, _last_action, _repeat_count
    _load_once()
    
    if _frame_buffer is None:
        _reset_buffer(obs)
    else:
        _frame_buffer.append(obs.astype(np.float32).copy())
    
    stacked = _get_stacked()
    x = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)
    q = _model(x).squeeze(0).cpu().numpy()
    best = int(np.argmax(q))

    if _last_action is not None:
        order = np.argsort(-q)
        if (float(q[order[0]]) - float(q[order[1]])) < _CLOSE_Q_DELTA:
            if _repeat_count < _MAX_REPEAT:
                best = _last_action
                _repeat_count += 1
            else:
                _repeat_count = 0
        else:
            _repeat_count = 0

    _last_action = best
    return ACTIONS[best]