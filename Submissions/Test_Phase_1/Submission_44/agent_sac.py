"""Discrete SAC Agent - Submission (Enhanced)."""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Sequence

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = len(ACTIONS)

class Actor(nn.Module):
    def __init__(self, input_dim=18, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS)
        )
    
    def forward(self, x):
        logits = self.net(x)
        probs = torch.softmax(logits, dim=-1)
        return probs, torch.log_softmax(logits, dim=-1)
    
    def get_action(self, x, deterministic=True):
        probs, _ = self.forward(x)
        if deterministic:
            return torch.argmax(probs, dim=-1)
        dist = torch.distributions.Categorical(probs)
        return dist.sample()

_ACTOR = None
_DEVICE = "cpu"

def _load():
    global _ACTOR
    if _ACTOR: 
        return _ACTOR
    
    # Try loading best model first, then final, then legacy
    for fname in ["sac_actor_best.pth", "actor_sac_best.pth", "sac_actor_final.pth", "actor_sac.pth"]:
        p = os.path.join(os.path.dirname(__file__), fname)
        if os.path.exists(p):
            ckpt = torch.load(p, map_location=_DEVICE)
            input_dim = ckpt['net.0.weight'].shape[1]
            hidden = ckpt['net.0.weight'].shape[0]
            _ACTOR = Actor(input_dim, hidden)
            _ACTOR.load_state_dict(ckpt)
            _ACTOR.eval()
            return _ACTOR
    
    raise FileNotFoundError("No trained SAC model found")

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    actor = _load()
    x = torch.FloatTensor(obs).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        action_idx = actor.get_action(x, deterministic=True).item()
    return ACTIONS[action_idx]