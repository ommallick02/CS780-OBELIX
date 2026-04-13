"""
D3RQN-Noisy (Dueling Deep Recurrent Q-Network with Noisy Networks) Agent for OBELIX.

This agent loads pretrained LSTM+Dueling+Noisy weights and uses them for evaluation.
The LSTM maintains temporal memory across steps within an episode.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]


class NoisyLinear(nn.Module):
    """Noisy linear layer with factorized noise."""

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


class DuelingLSTMNoisyDQN(nn.Module):
    """Dueling DQN with LSTM and Noisy output layers."""

    def __init__(self, in_dim: int = 18, hidden_dim: int = 64, n_actions: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(in_dim, hidden_dim, batch_first=True)

        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, 1)
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, n_actions)
        )

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)

        features = lstm_out[:, -1, :]
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values, hidden

    def init_hidden(self, batch_size: int = 1, device: str = "cpu"):
        h = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return (h, c)


# Global state
_model: Optional[DuelingLSTMNoisyDQN] = None
_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
_last_action: Optional[int] = None
_repeat_count: int = 0

_MAX_REPEAT = 2
_CLOSE_Q_DELTA = 0.05


def _load_once():
    global _model, _hidden
    if _model is not None:
        return

    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")

    if not os.path.exists(wpath):
        raise FileNotFoundError(
            "weights.pth not found next to agent.py. Train offline and include it in the submission zip."
        )

    m = DuelingLSTMNoisyDQN(in_dim=18, hidden_dim=64, n_actions=5)
    sd = torch.load(wpath, map_location="cpu")
    m.load_state_dict(sd, strict=True)
    m.eval()

    _model = m
    _hidden = None


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _model, _hidden, _last_action, _repeat_count

    _load_once()

    if _hidden is None:
        _hidden = _model.init_hidden(batch_size=1, device="cpu")

    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    q, _hidden = _model(x, _hidden)
    q = q.squeeze(0).cpu().numpy()

    _hidden = (_hidden[0].detach(), _hidden[1].detach())

    best = int(np.argmax(q))

    if _last_action is not None:
        order = np.argsort(-q)
        best_q, second_q = float(q[order[0]]), float(q[order[1]])
        if (best_q - second_q) < _CLOSE_Q_DELTA:
            if _repeat_count < _MAX_REPEAT:
                best = _last_action
                _repeat_count += 1
            else:
                _repeat_count = 0
        else:
            _repeat_count = 0

    _last_action = best
    return ACTIONS[best]
