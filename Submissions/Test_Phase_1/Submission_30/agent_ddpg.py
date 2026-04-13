"""DDPG for Discrete Actions (OBELIX)."""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Sequence

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = len(ACTIONS)

class Actor(nn.Module):
    """Deterministic policy: outputs action probabilities (softmax for discrete)."""
    def __init__(self, input_dim=18, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
            nn.Softmax(dim=-1)  # Discrete output
        )
    
    def forward(self, x):
        return self.net(x)
    
    def get_action(self, x, deterministic=True):
        probs = self.forward(x)
        if deterministic:
            return torch.argmax(probs, dim=-1)
        # Sample for exploration
        dist = torch.distributions.Categorical(probs)
        return dist.sample()

class Critic(nn.Module):
    """Q(s, a) where a is one-hot encoded."""
    def __init__(self, input_dim=18, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + N_ACTIONS, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, state, action_onehot):
        x = torch.cat([state, action_onehot], dim=-1)
        return self.net(x).squeeze(-1)

_ACTOR = None
_CRITIC = None  # Not used in inference but loaded for consistency
_DEVICE = "cpu"

def _load():
    global _ACTOR
    if _ACTOR: return _ACTOR
    p = os.path.join(os.path.dirname(__file__), "actor.pth")
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