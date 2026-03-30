"""
Submission agent for OBELIX.
Implements Q-learning with state clustering for evaluation.
"""

import os
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple, Sequence
import numpy as np

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")


class StateClusterer:
    """State clustering for dimensionality reduction."""
    
    def __init__(self, max_clusters: int = 200, similarity_threshold: float = 0.15):
        self.max_clusters = max_clusters
        self.similarity_threshold = similarity_threshold
        self.clusters: List[np.ndarray] = []
        self.cluster_counts: List[int] = []
        
    def _hamming_distance(self, obs1: np.ndarray, obs2: np.ndarray) -> float:
        return np.mean(np.abs(obs1 - obs2))
    
    def _find_best_cluster(self, obs: np.ndarray) -> Tuple[int, float]:
        if len(self.clusters) == 0:
            return -1, float('inf')
        distances = [self._hamming_distance(obs, c) for c in self.clusters]
        best_idx = int(np.argmin(distances))
        return best_idx, distances[best_idx]
    
    def get_cluster_id(self, obs: np.ndarray) -> int:
        obs = obs.astype(float)
        best_idx, min_dist = self._find_best_cluster(obs)
        
        # Use closest cluster if within threshold
        if best_idx >= 0 and min_dist < self.similarity_threshold:
            return best_idx
        
        # Create new cluster if under limit
        if len(self.clusters) < self.max_clusters:
            self.clusters.append(obs.copy())
            self.cluster_counts.append(1)
            return len(self.clusters) - 1
        
        # At capacity: use closest
        if best_idx >= 0:
            return best_idx
        return 0
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.clusters = data['clusters']
        self.cluster_counts = data['cluster_counts']
        self.max_clusters = data['max_clusters']
        self.similarity_threshold = data['similarity_threshold']


class QLearningAgent:
    """Q-learning agent for inference."""
    
    def __init__(self, n_actions: int = 5):
        self.n_actions = n_actions
        self.q_table: Dict[Tuple[int, int], float] = defaultdict(float)
        self.clusterer = StateClusterer()
        
    def select_action(self, obs: np.ndarray, rng: np.random.Generator, training: bool = False) -> int:
        """Select action greedily (no exploration during evaluation)."""
        cluster_id = self.clusterer.get_cluster_id(obs)
        
        # Greedy selection
        q_values = [self.q_table.get((cluster_id, a), 0.0) for a in range(self.n_actions)]
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        action = int(rng.choice(best_actions))
        
        return action
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.q_table = defaultdict(float, data['q_table'])
        self.clusterer.load(data['clusterer_path'])


# Global agent instance (lazy loading)
_AGENT = None
_AGENT_LOADED = False


def _load_agent():
    """Load trained agent from submission directory."""
    global _AGENT, _AGENT_LOADED
    
    if _AGENT_LOADED:
        return _AGENT
    
    # Search for agent files in submission directory
    submission_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_names = ["agent_best.pkl", "agent.pkl", "qlearning_agent_best.pkl", "qlearning_agent.pkl"]
    
    for name in possible_names:
        path = os.path.join(submission_dir, name)
        if os.path.exists(path):
            try:
                _AGENT = QLearningAgent()
                _AGENT.load(path)
                _AGENT_LOADED = True
                return _AGENT
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue
    
    # Fallback: return untrained agent (random policy)
    _AGENT = QLearningAgent()
    _AGENT_LOADED = True
    return _AGENT


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    Main policy function called by evaluator.
    
    Args:
        obs: 18-dimensional binary observation vector
        rng: Random number generator
    
    Returns:
        Action string: one of "L45", "L22", "FW", "R22", "R45"
    """
    agent = _load_agent()
    action_idx = agent.select_action(obs, rng, training=False)
    return ACTIONS[action_idx]