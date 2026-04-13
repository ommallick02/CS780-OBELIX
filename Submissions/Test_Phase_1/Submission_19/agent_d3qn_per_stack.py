"""D3QN with PER and State Stacking."""

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
    def __init__(self, in_dim=18*_STACK_SIZE, n_actions=5):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.value_head = nn.Linear(64, 1)
        self.advantage_head = nn.Linear(64, n_actions)
    def forward(self, x):
        f = self.encoder(x)
        return self.value_head(f) + self.advantage_head(f) - self.advantage_head(f).mean(dim=-1, keepdim=True)

_model: Optional[DuelingDQN] = None
_frame_buffer: Optional[deque] = None

def _reset_buffer(obs):
    global _frame_buffer
    _frame_buffer = deque(maxlen=_STACK_SIZE)
    for _ in range(_STACK_SIZE-1): _frame_buffer.append(np.zeros(18, dtype=np.float32))
    _frame_buffer.append(obs.astype(np.float32).copy())

def _get_stacked():
    return np.concatenate(list(_frame_buffer), axis=0)

def _load_once():
    global _model
    if _model: return
    wpath = os.path.join(os.path.dirname(__file__), "weights.pth")
    if not os.path.exists(wpath): raise FileNotFoundError("weights.pth missing")
    m = DuelingDQN()
    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    m.eval()
    _model = m

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _frame_buffer
    _load_once()
    if _frame_buffer is None: _reset_buffer(obs)
    else: _frame_buffer.append(obs.astype(np.float32).copy())
    x = torch.tensor(_get_stacked(), dtype=torch.float32).unsqueeze(0)
    return ACTIONS[int(_model(x).squeeze(0).argmax().item())]