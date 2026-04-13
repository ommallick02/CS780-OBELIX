"""
PPO (Proximal Policy Optimization) policy for OBELIX submission.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Sequence, Tuple

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = len(ACTIONS)


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim: int = 18, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, N_ACTIONS),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        probs = self.actor(features)
        value = self.critic(features)
        return probs, value


# Global
_NETWORK = None
_DEVICE = "cpu"


def _load_network():
    """Load trained PPO network."""
    global _NETWORK, _DEVICE
    
    if _NETWORK is not None:
        return _NETWORK
    
    submission_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_names = [
        "ppo_agent_best.pt",
        "ppo_agent.pt",
        "agent_best.pt",
        "agent.pt"
    ]
    
    for name in possible_names:
        path = os.path.join(submission_dir, name)
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=_DEVICE)
                
                # Infer hidden dim
                state_dict = checkpoint['network_state_dict']
                first_weight = state_dict['shared.0.weight']
                hidden_dim = first_weight.shape[0]
                
                _NETWORK = ActorCriticNetwork(input_dim=18, hidden_dim=hidden_dim)
                _NETWORK.load_state_dict(state_dict)
                _NETWORK.eval()
                
                print(f"Loaded PPO network from {path}")
                return _NETWORK
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue
    
    print("Warning: No trained network found, using random policy")
    _NETWORK = None
    return _NETWORK


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    Select action using trained PPO policy (greedy).
    """
    network = _load_network()
    
    if network is None:
        return ACTIONS[int(rng.integers(N_ACTIONS))]
    
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(_DEVICE)
    
    with torch.no_grad():
        probs, _ = network(obs_tensor)
    
    # Greedy selection
    action_idx = int(torch.argmax(probs).item())
    
    return ACTIONS[action_idx]