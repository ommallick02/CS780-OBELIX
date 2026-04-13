"""VPG + LSTM agent for OBELIX submission.

Submission structure:
    submission.zip
        agent_vpg_lstm.py   (rename to agent.py)
        vpg_lstm.pt
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
        self.hidden_dim  = hidden_dim
        self.encoder     = nn.Sequential(nn.Linear(input_dim, enc_dim), nn.ReLU())
        self.lstm        = nn.LSTM(enc_dim, hidden_dim, batch_first=False)
        self.policy_head = nn.Linear(hidden_dim, N_ACTIONS)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        enc         = self.encoder(x)
        out, hidden = self.lstm(enc, hidden)
        probs       = torch.softmax(self.policy_head(out), dim=-1)
        return probs, hidden

    def init_hidden(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(1, 1, self.hidden_dim, device=device)
        return (z, z.clone())


# ── Globals ───────────────────────────────────────────────────────────────────
_NETWORK: Optional[ActorLSTM] = None
_HIDDEN:  Optional[Tuple]     = None   # persists across policy() calls within an episode
_DEVICE = torch.device("cpu")


def _load_network() -> ActorLSTM:
    global _NETWORK, _DEVICE

    if _NETWORK is not None:
        return _NETWORK

    submission_dir = os.path.dirname(os.path.abspath(__file__))
    wpath = os.path.join(submission_dir, "vpg_lstm.pt")

    if not os.path.exists(wpath):
        raise FileNotFoundError(f"vpg_lstm.pt not found in {submission_dir}")

    sd = torch.load(wpath, map_location=_DEVICE)

    # Infer architecture from weight shapes
    input_dim  = sd["encoder.0.weight"].shape[1]
    enc_dim    = sd["encoder.0.weight"].shape[0]
    hidden_dim = sd["lstm.weight_ih_l0"].shape[0] // 4   # LSTM has 4 gates

    _NETWORK = ActorLSTM(input_dim, enc_dim, hidden_dim)
    _NETWORK.load_state_dict(sd)
    _NETWORK.eval()
    return _NETWORK


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    Greedy policy for evaluation.

    The LSTM hidden state is initialised on the first call and carried
    across subsequent calls, providing temporal memory within each episode.
    """
    global _HIDDEN

    net = _load_network()

    if _HIDDEN is None:
        _HIDDEN = net.init_hidden(_DEVICE)

    obs_t = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(_DEVICE)
    # shape: (seq_len=1, batch=1, input_dim)

    with torch.no_grad():
        probs, _HIDDEN = net(obs_t, _HIDDEN)
        action_idx = int(probs.squeeze().argmax().item())   # greedy

    return ACTIONS[action_idx]
