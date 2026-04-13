import torch
import torch.nn as nn
import numpy as np
from torchviz import make_dot
from typing import Tuple

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

class LSTMNoisyDQN(nn.Module):
    def __init__(self, in_dim: int = 18, hidden_dim: int = 64, n_actions: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, batch_first=True)
        self.noisy_fc = NoisyLinear(hidden_dim, n_actions)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None):
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

if __name__ == "__main__":
    device = torch.device("cpu")
    model = LSTMNoisyDQN(in_dim=18, hidden_dim=64, n_actions=5).to(device)
    
    dummy_input = torch.zeros(1, 1, 18, device=device)
    dummy_hidden = model.init_hidden(batch_size=1, device=device)
    
    out_q, _ = model(dummy_input, dummy_hidden)
    dot = make_dot(out_q, params=dict(model.named_parameters()))
    dot.render("drqn_noisy_architecture", format="png", cleanup=True)
    print("Saved DRQN-Noisy architecture diagram to drqn_noisy_architecture.png")