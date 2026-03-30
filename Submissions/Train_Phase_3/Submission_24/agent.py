"""
Dyna-Q policy for OBELIX submission.
Uses learned Q-table and model for inference.
"""

import os
import pickle
from collections import defaultdict
from typing import Dict, Tuple, Sequence
import numpy as np

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = len(ACTIONS)


class StateDiscretizer:
    """Simple state discretizer for OBELIX."""
    
    def discretize(self, obs: np.ndarray) -> Tuple:
        discrete = np.round(obs).astype(int)
        return tuple(discrete)


class DynaQAgent:
    """Dyna-Q agent for inference."""
    
    def __init__(self):
        self.discretizer = StateDiscretizer()
        self.q_table: Dict[Tuple[Tuple, int], float] = defaultdict(float)
        self.n_actions = N_ACTIONS  # <-- ADD THIS LINE
    
    def select_action(self, state: Tuple, rng: np.random.Generator) -> int:
        """Greedy action selection."""
        q_values = [self.q_table.get((state, a), 0.0) for a in range(self.n_actions)]
        max_q = max(q_values)
        
        best_actions = [a for a, q in enumerate(q_values) if abs(q - max_q) < 1e-9]
        if len(best_actions) == 0:
            best_actions = list(range(self.n_actions))
        
        return int(rng.choice(best_actions))
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.q_table = defaultdict(float, data['q_table'])


# Global
_AGENT = None


def _load_agent():
    """Load trained Dyna-Q agent."""
    global _AGENT
    
    if _AGENT is not None:
        return _AGENT
    
    submission_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_names = [
        "dyna_q_agent_best.pkl",
        "dyna_q_agent.pkl",
        "agent_best.pkl",
        "agent.pkl"
    ]
    
    for name in possible_names:
        path = os.path.join(submission_dir, name)
        if os.path.exists(path):
            try:
                _AGENT = DynaQAgent()
                _AGENT.load(path)
                print(f"Loaded Dyna-Q agent from {path}")
                print(f"  - Q-table size: {len(_AGENT.q_table)}")
                return _AGENT
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue
    
    print("Warning: No trained agent found, using random policy")
    _AGENT = DynaQAgent()
    return _AGENT


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    Select action using trained Dyna-Q policy (greedy).
    """
    agent = _load_agent()
    state = agent.discretizer.discretize(obs)
    action_idx = agent.select_action(state, rng)
    return ACTIONS[action_idx]
