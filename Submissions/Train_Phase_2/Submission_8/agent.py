"""
REINFORCE policy for OBELIX submission.
Loads trained policy network and selects actions.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Sequence

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = len(ACTIONS)


class PolicyNetwork(nn.Module):
    """Same architecture as training."""
    
    def __init__(self, input_dim: int = 18, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N_ACTIONS),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Global agent (lazy loading)
_POLICY = None
_DEVICE = None


def _load_policy():
    """Load trained policy from submission directory."""
    global _POLICY, _DEVICE
    
    if _POLICY is not None:
        return _POLICY, _DEVICE
    
    _DEVICE = "cpu"  # Must use CPU for submission
    
    # Search for model files
    submission_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_names = [
        "reinforce_agent_best.pt",
        "reinforce_agent.pt",
        "agent_best.pt",
        "agent.pt"
    ]
    
    for name in possible_names:
        path = os.path.join(submission_dir, name)
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=_DEVICE)
                
                # Try to infer hidden dim from state dict
                first_weight = checkpoint['policy_state_dict']['net.0.weight']
                hidden_dim = first_weight.shape[0]  # Output size of first layer
                
                _POLICY = PolicyNetwork(input_dim=18, hidden_dim=hidden_dim)
                _POLICY.load_state_dict(checkpoint['policy_state_dict'])
                _POLICY.eval()  # Evaluation mode
                
                print(f"Loaded REINFORCE policy from {path}")
                return _POLICY, _DEVICE
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue
    
    # Fallback: random policy
    print("Warning: No trained policy found, using random actions")
    _POLICY = None
    return _POLICY, _DEVICE


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    Select action using trained policy network.
    Deterministic: always pick highest probability action.
    """
    policy_net, device = _load_policy()
    
    if policy_net is None:
        # Random fallback
        return ACTIONS[int(rng.integers(N_ACTIONS))]
    
    # Convert observation to tensor
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    
    # Get action probabilities
    with torch.no_grad():
        probs = policy_net(obs_tensor).cpu().numpy()[0]
    
    # Greedy selection (deterministic for evaluation)
    action_idx = int(np.argmax(probs))
    
    return ACTIONS[action_idx]