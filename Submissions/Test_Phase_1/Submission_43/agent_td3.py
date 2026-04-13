"""TD3 Agent - Submission."""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Sequence

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")

class Actor(nn.Module):
    def __init__(self, input_dim=18, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 5),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.net(x)

_ACTOR = None
_DEVICE = "cpu"

def _load():
    global _ACTOR
    if _ACTOR: 
        return _ACTOR
    
    for fname in ["td3_actor_best.pth", "actor_td3_best.pth", "td3_actor_final.pth", "actor_td3.pth"]:
        p = os.path.join(os.path.dirname(__file__), fname)
        if os.path.exists(p):
            ckpt = torch.load(p, map_location=_DEVICE)
            input_dim = ckpt['net.0.weight'].shape[1]
            hidden = ckpt['net.0.weight'].shape[0]
            _ACTOR = Actor(input_dim, hidden)
            _ACTOR.load_state_dict(ckpt)
            _ACTOR.eval()
            return _ACTOR
    
    raise FileNotFoundError("No model found")

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    actor = _load()
    x = torch.FloatTensor(obs).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        action_idx = torch.argmax(actor(x), dim=-1).item()
    return ACTIONS[action_idx]