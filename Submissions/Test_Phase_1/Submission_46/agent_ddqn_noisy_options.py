"""
Agent for Noisy DDQN with Options Framework.
Loads trained weights and uses a hand-coded option state machine
(Find → Push → Unwedge) to select actions.

Assumes the observation contains enough info to detect box/wedge proximity.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Sequence, Tuple, Optional

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = len(ACTIONS)

# ---- Noisy Network definitions (must match training) ----
class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)

class NoisyDQN(nn.Module):
    def __init__(self, input_dim: int = 18, hidden_dim: int = 128, n_actions: int = N_ACTIONS):
        super().__init__()
        self.fc1 = NoisyLinear(input_dim, hidden_dim)
        self.fc2 = NoisyLinear(hidden_dim, hidden_dim)
        self.fc3 = NoisyLinear(hidden_dim, n_actions)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# ---- Global state for the agent ----
_network: Optional[NoisyDQN] = None
_device = "cpu"
_current_option: str = "Find"
_option_steps: int = 0
_MAX_OPTION_STEPS = 50

def _get_box_distance(obs: np.ndarray) -> float:
    """Extract distance to nearest box from observation."""
    # Modify index to match your environment's observation layout.
    # Example: obs[10] might be distance to nearest box.
    return obs[10] if len(obs) > 10 else 1000.0

def _get_wedge_distance(obs: np.ndarray) -> float:
    """Extract distance to wedge from observation."""
    return obs[11] if len(obs) > 11 else 1000.0

def _option_termination(obs: np.ndarray, option: str) -> bool:
    """Return True if current option should terminate."""
    if option == "Find":
        # Terminate when a box is within pushing distance (e.g., < 50)
        return _get_box_distance(obs) < 50.0
    elif option == "Push":
        # Terminate when box is near wedge (distance < 50)
        return _get_wedge_distance(obs) < 50.0
    elif option == "Unwedge":
        # Termination when wedge is freed – we assume this ends the episode.
        # For safety, never terminate early here.
        return False
    return False

def _update_option(obs: np.ndarray, done: bool) -> str:
    """Update the current option based on termination and episode end."""
    global _current_option, _option_steps
    if done:
        _current_option = "Find"
        _option_steps = 0
        return _current_option

    terminated = _option_termination(obs, _current_option)
    if terminated or _option_steps >= _MAX_OPTION_STEPS:
        # Transition to next option
        if _current_option == "Find":
            _current_option = "Push"
        elif _current_option == "Push":
            _current_option = "Unwedge"
        else:  # Unwedge -> back to Find for next episode
            _current_option = "Find"
        _option_steps = 0
    else:
        _option_steps += 1
    return _current_option

def _load_network():
    global _network
    if _network is not None:
        return
    submission_dir = os.path.dirname(os.path.abspath(__file__))
    # Try to find the trained weights file (same name as used in training)
    weight_path = os.path.join(submission_dir, "ddqn_noisy_options_curriculum.pt")
    if not os.path.exists(weight_path):
        # Fallback to other possible names
        possible = ["ddqn_noisy_options.pt", "model.pt", "weights.pth"]
        for p in possible:
            full = os.path.join(submission_dir, p)
            if os.path.exists(full):
                weight_path = full
                break
        else:
            print("Warning: No trained weights found. Using random policy.")
            _network = None
            return

    checkpoint = torch.load(weight_path, map_location=_device)
    # Infer hidden dimension from loaded weights
    first_weight = checkpoint['q_network']['fc1.weight_mu']
    hidden_dim = first_weight.shape[0]
    _network = NoisyDQN(input_dim=18, hidden_dim=hidden_dim, n_actions=N_ACTIONS)
    _network.load_state_dict(checkpoint['q_network'])
    _network.eval()
    print(f"Loaded Noisy DDQN + Options from {weight_path}")

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    Main policy function called by the OBELIX environment.
    Uses the current option (Find/Push/Unwedge) and the Q-network to select an action.
    """
    global _network, _current_option, _option_steps

    _load_network()
    if _network is None:
        return ACTIONS[int(rng.integers(N_ACTIONS))]

    # Update option based on observation (no reward needed)
    _update_option(obs, done=False)

    # Select action greedily from the Q-network
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(_device)
    with torch.no_grad():
        q_values = _network(obs_tensor)
    action_idx = int(q_values.argmax(dim=1).item())
    return ACTIONS[action_idx]