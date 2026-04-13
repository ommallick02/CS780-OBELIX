"""Enhanced GAE-PPO with Options Framework - Submission Agent."""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Sequence

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_OPTIONS = 4
OPTION_FIND, OPTION_ALIGN, OPTION_PUSH, OPTION_RECOVER = range(4)

class OptionActorCritic(nn.Module):
    def __init__(self, input_dim=18, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.actor = nn.Linear(hidden, 5)
        self.critic = nn.Linear(hidden, 1)
        self.terminator = nn.Linear(hidden, 1)
        
    def forward(self, x):
        f = self.shared(x)
        return (torch.softmax(self.actor(f), dim=-1),
                self.critic(f).squeeze(-1),
                torch.sigmoid(self.terminator(f)).squeeze(-1))

class OptionController(nn.Module):
    def __init__(self, input_dim=18, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 4)
        )
    
    def forward(self, x):
        return self.net(x)

_MODELS = None
_CONTROLLER = None
_CURRENT_OPTION = None
_DEVICE = "cpu"

def _load():
    global _MODELS, _CONTROLLER
    if _MODELS is not None:
        return
        
    path = os.path.join(os.path.dirname(__file__), "enhanced_ppo_best.pt")
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), "enhanced_ppo.pt")
    
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=_DEVICE)
        _MODELS = [OptionActorCritic() for _ in range(4)]
        for i, state in enumerate(ckpt['option_nets']):
            _MODELS[i].load_state_dict(state)
            _MODELS[i].eval()
        _CONTROLLER = OptionController()
        _CONTROLLER.load_state_dict(ckpt['controller'])
        _CONTROLLER.eval()

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load()
    global _CURRENT_OPTION
    
    if _MODELS is None:
        return "FW" if obs[16] > 0.5 else ("L22" if obs[0] > obs[14] else "R22")
    
    # Determine state
    attached = any(obs[4:12] > 0.5) and obs[16] > 0.5
    stuck = bool(obs[17])
    
    # Check termination
    if _CURRENT_OPTION is not None:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            _, _, term_prob = _MODELS[_CURRENT_OPTION](obs_t)
        if term_prob.item() > 0.5:
            _CURRENT_OPTION = None
    
    # Select option
    if _CURRENT_OPTION is None:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            q_vals = _CONTROLLER(obs_t).squeeze().cpu().numpy()
        
        valid = np.ones(4, dtype=bool)
        if attached:
            valid[OPTION_FIND] = valid[OPTION_ALIGN] = False
        else:
            valid[OPTION_PUSH] = False
        if not stuck:
            valid[OPTION_RECOVER] = False
        
        valid_idx = np.where(valid)[0]
        if len(valid_idx) > 0:
            valid_q = q_vals[valid_idx]
            _CURRENT_OPTION = int(valid_idx[np.argmax(valid_q)])
        else:
            _CURRENT_OPTION = OPTION_RECOVER if stuck else OPTION_FIND
    
    # Get action
    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        probs, _, _ = _MODELS[_CURRENT_OPTION](obs_t)
        action_idx = torch.argmax(probs).item()
    
    return ACTIONS[action_idx]