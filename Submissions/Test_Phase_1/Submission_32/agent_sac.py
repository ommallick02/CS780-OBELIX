"""Discrete SAC for OBELIX."""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Sequence

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = len(ACTIONS)

class Actor(nn.Module):
    """Outputs action probabilities with temperature."""
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
        log_probs = torch.log_softmax(logits, dim=-1)
        return probs, log_probs
    
    def get_action(self, x, deterministic=False):
        probs, _ = self.forward(x)
        if deterministic:
            return torch.argmax(probs, dim=-1)
        dist = torch.distributions.Categorical(probs)
        return dist.sample()

_ACTOR = None
_DEVICE = "cpu"

def _load():
    global _ACTOR
    if _ACTOR: return _ACTOR
    p = os.path.join(os.path.dirname(__file__), "actor_sac.pth")
    ckpt = torch.load(p, map_location=_DEVICE)
    input_dim = ckpt['net.0.weight'].shape[1]
    hidden = ckpt['net.0.weight'].shape[0]
    _ACTOR = Actor(input_dim, hidden)
    _ACTOR.load_state_dict(ckpt)
    _ACTOR.eval()
    return _ACTOR

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    actor = _load()
    x = torch.FloatTensor(obs).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        action_idx = actor.get_action(x, deterministic=True).item()
    return ACTIONS[action_idx]