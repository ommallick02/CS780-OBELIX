"""
D3QN-PER (Double Dueling DQN with Prioritized Experience Replay) + Noisy Networks for OBELIX submission.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Sequence

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = len(ACTIONS)


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


class NoisyDuelingDQN(nn.Module):
    def __init__(self, input_dim: int = 18, hidden_dim: int = 128):
        super().__init__()
        
        self.feature = nn.Sequential(
            NoisyLinear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, N_ACTIONS)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values
    
    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# Global
_NETWORK = None
_DEVICE = "cpu"


def _load_network():
    global _NETWORK, _DEVICE
    
    if _NETWORK is not None:
        return _NETWORK
    
    submission_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_names = [
        "d3qn_per_noisy_agent_best.pt",
        "d3qn_per_noisy_agent.pt",
        "agent_best.pt",
        "agent.pt"
    ]
    
    for name in possible_names:
        path = os.path.join(submission_dir, name)
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=_DEVICE)
                
                # Infer hidden dim from feature layer
                first_weight = checkpoint['q_network']['feature.0.weight_mu']
                hidden_dim = first_weight.shape[0]
                
                _NETWORK = NoisyDuelingDQN(input_dim=18, hidden_dim=hidden_dim)
                _NETWORK.load_state_dict(checkpoint['q_network'])
                _NETWORK.eval()
                
                print(f"Loaded D3QN-PER+Noisy from {path}")
                return _NETWORK
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue
    
    print("Warning: No trained network found, using random policy")
    _NETWORK = None
    return _NETWORK


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    network = _load_network()
    
    if network is None:
        return ACTIONS[int(rng.integers(N_ACTIONS))]
    
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(_DEVICE)
    
    with torch.no_grad():
        q_values = network(obs_tensor)
    
    action_idx = int(q_values.argmax(dim=1).item())
    
    return ACTIONS[action_idx]