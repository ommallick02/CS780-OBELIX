"""
VPG+LSTM agent with trained actor (greedy evaluation).
Loads model trained with curriculum + shaping + GAE.
"""

import os
from typing import Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = len(ACTIONS)


class ActorLSTM(nn.Module):
    def __init__(self, input_dim: int = 18, enc_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, enc_dim), nn.ReLU())
        self.lstm = nn.LSTM(enc_dim, hidden_dim, batch_first=False)
        self.policy_head = nn.Linear(hidden_dim, N_ACTIONS)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]):
        enc = self.encoder(x)
        out, hidden = self.lstm(enc, hidden)
        probs = torch.softmax(self.policy_head(out), dim=-1)
        return probs, hidden

    def init_hidden(self, device: torch.device):
        z = torch.zeros(1, 1, self.hidden_dim, device=device)
        return (z, z.clone())


_NETWORK: Optional[ActorLSTM] = None
_HIDDEN: Optional[Tuple] = None
_DEVICE = torch.device("cpu")


def _load_network():
    global _NETWORK
    if _NETWORK is not None:
        return
    submission_dir = os.path.dirname(os.path.abspath(__file__))
    wpath = os.path.join(submission_dir, "vpg_lstm_improved.pt")
    if not os.path.exists(wpath):
        # fallback
        wpath = os.path.join(submission_dir, "vpg_lstm.pt")
    if not os.path.exists(wpath):
        raise FileNotFoundError(f"No weights found in {submission_dir}")
    sd = torch.load(wpath, map_location=_DEVICE)
    # Infer dimensions
    input_dim = sd["encoder.0.weight"].shape[1]
    enc_dim = sd["encoder.0.weight"].shape[0]
    hidden_dim = sd["lstm.weight_ih_l0"].shape[0] // 4
    _NETWORK = ActorLSTM(input_dim, enc_dim, hidden_dim)
    _NETWORK.load_state_dict(sd)
    _NETWORK.eval()
    print(f"Loaded VPG+LSTM actor from {wpath}")


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _NETWORK, _HIDDEN
    _load_network()
    if _HIDDEN is None:
        _HIDDEN = _NETWORK.init_hidden(_DEVICE)
    obs_t = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        probs, _HIDDEN = _NETWORK(obs_t, _HIDDEN)
        action_idx = int(probs.squeeze().argmax().item())
    return ACTIONS[action_idx]