"""Feudal PPO Submission Agent with Manager-Worker Hierarchy."""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Sequence

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
GOAL_DIM = 16
MANAGER_INTERVAL = 10

class ManagerNetwork(nn.Module):
    def __init__(self, input_dim=18, hidden=128, goal_dim=GOAL_DIM):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
        self.goal_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, goal_dim), nn.Tanh()
        )
        
    def forward(self, x, hidden_state=None):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        if hidden_state is None:
            h0 = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(x.device)
            c0 = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(x.device)
            hidden_state = (h0, c0)
        lstm_out, new_hidden = self.lstm(x, hidden_state)
        features = lstm_out[:, -1, :]
        goal = self.goal_head(features)
        return goal, new_hidden

class WorkerNetwork(nn.Module):
    def __init__(self, input_dim=18, goal_dim=GOAL_DIM, hidden=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + goal_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden, 5), nn.Softmax(dim=-1)
        )
        
    def forward(self, obs, goal):
        x = torch.cat([obs, goal], dim=-1)
        features = self.encoder(x)
        return self.actor(features)

_MANAGER = None
_WORKER = None
_HIDDEN = None
_GOAL = None
_STEP = 0
_DEVICE = "cpu"

def _load():
    global _MANAGER, _WORKER
    if _MANAGER is not None:
        return
        
    path = os.path.join(os.path.dirname(__file__), "feudal_ppo_best.pt")
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), "feudal_ppo.pt")
    
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=_DEVICE)
        _MANAGER = ManagerNetwork()
        _WORKER = WorkerNetwork()
        
        # Add strict=False to both loaders to ignore the missing value/critic heads
        _MANAGER.load_state_dict(ckpt['manager_state'], strict=False)
        _WORKER.load_state_dict(ckpt['worker_state'], strict=False)
        
        _MANAGER.eval()
        _WORKER.eval()

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load()
    global _HIDDEN, _GOAL, _STEP
    
    if _MANAGER is None:
        # Fallback heuristic
        if obs[16] > 0.5:
            return "FW"
        return "L22" if obs[0] > obs[14] else "R22"
    
    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(_DEVICE)
    
    # Manager updates every MANAGER_INTERVAL steps
    if _STEP % MANAGER_INTERVAL == 0:
        with torch.no_grad():
            goal, _HIDDEN = _MANAGER(obs_t, _HIDDEN)
            _GOAL = goal
    
    # Worker acts every step
    with torch.no_grad():
        probs = _WORKER(obs_t, _GOAL)
        action_idx = torch.argmax(probs).item()
    
    _STEP += 1
    return ACTIONS[action_idx]
