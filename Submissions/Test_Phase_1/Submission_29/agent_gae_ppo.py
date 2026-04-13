"""PPO Agent with GAE for OBELIX submission."""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Sequence

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = len(ACTIONS)

class ActorCritic(nn.Module):
    def __init__(self, input_dim=18, hidden=128):
        super().__init__()
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # Actor head
        self.actor = nn.Linear(hidden, N_ACTIONS)
        # Critic head
        self.critic = nn.Linear(hidden, 1)
    
    def forward(self, x):
        f = self.encoder(x)
        return torch.softmax(self.actor(f), dim=-1), self.critic(f).squeeze(-1)
    
    def get_action(self, x, deterministic=False):
        probs, value = self.forward(x)
        if deterministic:
            return torch.argmax(probs, dim=-1), None, None
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

_MODEL = None
_DEVICE = "cpu"

def _load():
    global _MODEL
    if _MODEL: return _MODEL
    p = os.path.join(os.path.dirname(__file__), "ppo_agent.pt")
    ckpt = torch.load(p, map_location=_DEVICE)
    
    # Infer dims
    input_dim = ckpt['encoder.0.weight'].shape[1]
    hidden = ckpt['encoder.0.weight'].shape[0]
    
    _MODEL = ActorCritic(input_dim, hidden)
    _MODEL.load_state_dict(ckpt)
    _MODEL.eval()
    return _MODEL

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    net = _load()
    x = torch.FloatTensor(obs).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        action_idx = net.get_action(x, deterministic=True)[0].item()
    return ACTIONS[action_idx]
