"""VPG with State Stacking for OBELIX submission."""

import os
import numpy as np
import torch
import torch.nn as nn
from collections import deque

ACTIONS = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = 5
_STACK_SIZE = 4

class Actor(nn.Module):
    def __init__(self, in_dim=18*_STACK_SIZE, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
        )
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)
    def get_action(self, x, deterministic=False):
        p = self.forward(x)
        if deterministic:
            return torch.argmax(p, dim=-1), p
        d = torch.distributions.Categorical(p)
        a = d.sample()
        return a, d.log_prob(a)

_NETWORK = None
_fbuf = None
_DEVICE = "cpu"

def _reset_buf(obs):
    global _fbuf
    _fbuf = deque(maxlen=_STACK_SIZE)
    for _ in range(_STACK_SIZE-1): _fbuf.append(np.zeros(18, dtype=np.float32))
    _fbuf.append(obs.astype(np.float32).copy())

def _load():
    global _NETWORK
    if _NETWORK: return _NETWORK
    p = os.path.join(os.path.dirname(__file__), "vpg_stack.pt")
    ckpt = torch.load(p, map_location=_DEVICE)
    in_dim = ckpt['net.0.weight'].shape[1]
    hidden = ckpt['net.0.weight'].shape[0]
    _NETWORK = Actor(in_dim, hidden)
    _NETWORK.load_state_dict(ckpt)
    _NETWORK.eval()
    return _NETWORK

def policy(obs, rng):
    global _fbuf
    net = _load()
    if _fbuf is None: _reset_buf(obs)
    else: _fbuf.append(obs.astype(np.float32).copy())
    x = torch.FloatTensor(np.concatenate(list(_fbuf), axis=0)).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        return ACTIONS[int(net.get_action(x, deterministic=True)[0].item())]
