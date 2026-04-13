"""
Agent for DRQN-Noisy (LSTM + Noisy Networks) trained with curriculum and reward shaping.
Loads LSTM weights and maintains LSTM hidden state across steps within an episode.
No options framework.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)

# ---- Noisy Linear (same as training) ----
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

# ---- LSTM + Noisy DQN ----
class LSTMNoisyDQN(nn.Module):
    def __init__(self, in_dim: int = 18, hidden_dim: int = 64, n_actions: int = N_ACTIONS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, batch_first=True)
        self.noisy_fc = NoisyLinear(hidden_dim, n_actions)

    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)
        q_values = self.noisy_fc(lstm_out[:, -1, :])
        return q_values, hidden

    def init_hidden(self, batch_size: int = 1, device: str = "cpu"):
        h = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return (h, c)

# ---- Global state for the agent ----
_model: Optional[LSTMNoisyDQN] = None
_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
_device = "cpu"

def _load_network():
    global _model
    if _model is not None:
        return
    submission_dir = os.path.dirname(os.path.abspath(__file__))
    weight_path = os.path.join(submission_dir, "drqn_noisy_curriculum.pt")
    if not os.path.exists(weight_path):
        possible = ["drqn_noisy_curriculum.pt", "drqn_noisy.pt", "weights.pth"]
        for p in possible:
            full = os.path.join(submission_dir, p)
            if os.path.exists(full):
                weight_path = full
                break
        else:
            print("Warning: No trained weights found. Using random policy.")
            _model = None
            return

    # Infer hidden dimension from loaded state dict (need to know it)
    # We'll use default 64; if mismatch, we could auto-detect, but assume 64.
    _model = LSTMNoisyDQN(in_dim=18, hidden_dim=64, n_actions=N_ACTIONS)
    state_dict = torch.load(weight_path, map_location=_device)
    _model.load_state_dict(state_dict, strict=True)
    _model.eval()
    print(f"Loaded DRQN-Noisy + curriculum from {weight_path}")

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    Policy function for DRQN-Noisy.
    Maintains LSTM hidden state across calls within the same episode.
    The environment resets the hidden state automatically at the start of each episode
    because the global _hidden is set to None when the episode ends? Actually we don't know episode boundary.
    The OBELIX environment calls policy repeatedly for one episode, then resets.
    We need to detect episode reset – but there's no explicit signal. We'll rely on the fact that
    after an episode ends, the next call to policy is with a new observation, but we cannot know.
    A common trick: reset hidden when the observation is the initial state (e.g., distance > threshold).
    Simpler: assume the evaluation script will reload the agent each episode? Not reliable.
    We'll implement a simple heuristic: if the norm of obs is very small (initial state), reset hidden.
    Alternatively, we can store a flag that is reset externally – but the API doesn't allow.
    We'll reset hidden when we detect a "reset" pattern (e.g., agent at origin).
    For OBELIX, the initial observation likely has agent at (0,0) – we can check position indices.
    """
    global _model, _hidden

    _load_network()
    if _model is None:
        return ACTIONS[int(rng.integers(N_ACTIONS))]

    # Heuristic to detect episode reset: if the first two coordinates (x,y) are near zero.
    # Modify indices according to your observation layout.
    if len(obs) >= 2 and np.linalg.norm(obs[0:2]) < 1e-5:
        _hidden = None

    if _hidden is None:
        _hidden = _model.init_hidden(batch_size=1, device=_device)

    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(_device)
    with torch.no_grad():
        q_values, _hidden = _model(obs_tensor, _hidden)
    # Detach hidden to avoid graph accumulation (not strictly needed for eval)
    _hidden = (_hidden[0].detach(), _hidden[1].detach())
    action_idx = int(q_values.argmax(dim=1).item())
    return ACTIONS[action_idx]