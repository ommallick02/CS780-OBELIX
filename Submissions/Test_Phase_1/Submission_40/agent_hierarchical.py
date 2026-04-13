"""Hierarchical Options-based submission agent."""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Sequence

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_OPTIONS = 4
OPTION_FIND, OPTION_ALIGN, OPTION_PUSH, OPTION_RECOVER = 0, 1, 2, 3

class IntraOptionNetwork(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.actor = nn.Linear(hidden_dim, 5)
        self.critic = nn.Linear(hidden_dim, 1)
        self.terminator = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        f = self.shared(x)
        return (torch.softmax(self.actor(f), dim=-1), 
                self.critic(f).squeeze(-1),
                torch.sigmoid(self.terminator(f)).squeeze(-1))

class OptionCritic(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU()
        )
        self.option_values = nn.Linear(hidden_dim, 4)
        
    def forward(self, x):
        return self.option_values(self.encoder(x))

_MODELS = None
_CRITIC = None
_CURRENT_OPTION = None
_DEVICE = "cpu"

def _load():
    global _MODELS, _CRITIC
    if _MODELS is not None:
        return
        
    path = os.path.join(os.path.dirname(__file__), "hierarchical_a2c_best.pt")
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), "hierarchical_a2c.pt")
    
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=_DEVICE)
        _MODELS = [IntraOptionNetwork() for _ in range(4)]
        for i, state in enumerate(ckpt['intra_option_states']):
            _MODELS[i].load_state_dict(state)
            _MODELS[i].eval()
        _CRITIC = OptionCritic()
        _CRITIC.load_state_dict(ckpt['critic_state'])
        _CRITIC.eval()
    else:
        _MODELS = None

def _select_option(obs: np.ndarray, attached: bool, stuck: bool) -> int:
    if _CRITIC is None:
        return OPTION_FIND if not attached else OPTION_PUSH
    
    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        q_values = _CRITIC(obs_t).squeeze().cpu().numpy()
    
    # Apply masks
    valid = np.ones(4, dtype=bool)
    if attached:
        valid[OPTION_FIND] = valid[OPTION_ALIGN] = False
    else:
        valid[OPTION_PUSH] = False
    if not stuck:
        valid[OPTION_RECOVER] = False
    
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        return OPTION_RECOVER if stuck else OPTION_FIND
    
    valid_q = q_values[valid_idx]
    return int(valid_idx[np.argmax(valid_q)])

def _should_terminate(obs: np.ndarray, option_idx: int, attached: bool, stuck: bool) -> bool:
    # Hard termination
    if option_idx == OPTION_FIND and attached:
        return True
    if option_idx == OPTION_PUSH and not attached:
        return True
    if option_idx == OPTION_RECOVER and not stuck:
        return True
    
    # Learned termination
    if _MODELS:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            _, _, term_prob = _MODELS[option_idx](obs_t)
        return term_prob.item() > 0.5
    return False

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load()
    
    if _MODELS is None:
        # Fallback to heuristic
        if obs[16] > 0.5:
            return "FW"
        if obs[0] > 0.5 or obs[2] > 0.5:
            return "L22"
        if obs[14] > 0.5 or obs[12] > 0.5:
            return "R22"
        return "FW"
    
    global _CURRENT_OPTION
    attached = any(obs[4:12] > 0.5) and obs[16] > 0.5  # Rough heuristic for attached
    stuck = bool(obs[17])
    
    # Check termination
    if (_CURRENT_OPTION is None or 
        _should_terminate(obs, _CURRENT_OPTION, attached, stuck)):
        _CURRENT_OPTION = _select_option(obs, attached, stuck)
    
    # Get action from intra-option policy
    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        probs, _, _ = _MODELS[_CURRENT_OPTION](obs_t)
        action_idx = torch.argmax(probs).item()
    
    return ACTIONS[action_idx]