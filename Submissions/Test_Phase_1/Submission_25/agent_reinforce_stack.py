"""REINFORCE with State Stacking for OBELIX submission."""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Sequence
from collections import deque

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = len(ACTIONS)
_STACK_SIZE = 4

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int = 18 * _STACK_SIZE, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N_ACTIONS),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)
    
    def get_action(self, x: torch.Tensor, deterministic: bool = False):
        probs = self.forward(x)
        if deterministic:
            return torch.argmax(probs, dim=-1), probs
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, probs

_NETWORK = None
_fbuf = None
_DEVICE = "cpu"

def _reset_buffer(obs):
    global _fbuf
    _fbuf = deque(maxlen=_STACK_SIZE)
    for _ in range(_STACK_SIZE - 1):
        _fbuf.append(np.zeros(18, dtype=np.float32))
    _fbuf.append(obs.astype(np.float32).copy())

def _stacked():
    return np.concatenate(list(_fbuf), axis=0)

def _load():
    global _NETWORK
    if _NETWORK is not None:
        return _NETWORK
    p = os.path.join(os.path.dirname(__file__), "reinforce_stack.pt")
    if not os.path.exists(p):
        raise FileNotFoundError("weights.pth not found")
    ckpt = torch.load(p, map_location=_DEVICE)
    in_dim = ckpt['net.0.weight'].shape[1]
    hidden = ckpt['net.0.weight'].shape[0]
    _NETWORK = PolicyNetwork(input_dim=in_dim, hidden_dim=hidden)
    _NETWORK.load_state_dict(ckpt)
    _NETWORK.eval()
    return _NETWORK

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _fbuf
    net = _load()
    if _fbuf is None:
        _reset_buffer(obs)
    else:
        _fbuf.append(obs.astype(np.float32).copy())
    x = torch.FloatTensor(_stacked()).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        action_idx = net.get_action(x, deterministic=True)[0].item()
    return ACTIONS[action_idx]
