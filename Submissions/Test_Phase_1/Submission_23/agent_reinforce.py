"""REINFORCE agent for OBELIX submission."""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Sequence

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = len(ACTIONS)

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int = 18, hidden_dim: int = 128):
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
_DEVICE = "cpu"

def _load_network():
    global _NETWORK, _DEVICE
    if _NETWORK is not None:
        return _NETWORK
    
    submission_dir = os.path.dirname(os.path.abspath(__file__))
    wpath = os.path.join(submission_dir, "reinforce_agent.pt")
    
    if not os.path.exists(wpath):
        raise FileNotFoundError(f"weights.pth not found in {submission_dir}")
    
    checkpoint = torch.load(wpath, map_location=_DEVICE)
    
    # Infer architecture
    input_dim = checkpoint['net.0.weight'].shape[1]
    hidden_dim = checkpoint['net.0.weight'].shape[0]
    
    _NETWORK = PolicyNetwork(input_dim=input_dim, hidden_dim=hidden_dim)
    _NETWORK.load_state_dict(checkpoint)
    _NETWORK.eval()
    return _NETWORK

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """Deterministic policy for evaluation (greedy action selection)."""
    network = _load_network()
    
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(_DEVICE)
    
    with torch.no_grad():
        action_idx = network.get_action(obs_tensor, deterministic=True)[0].item()
    
    return ACTIONS[action_idx]
