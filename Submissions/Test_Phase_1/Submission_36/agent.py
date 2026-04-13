"""
MCTS (Monte Carlo Tree Search) for OBELIX submission.
Uses lightweight online planning.
"""

import os
import math
import pickle  # Add this import
from typing import Dict, List, Tuple, Optional
import numpy as np
from typing import Sequence

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = len(ACTIONS)


class MCTSNode:
    """Lightweight MCTS node."""
    
    __slots__ = ['state', 'parent', 'action', 'visits', 'value', 'children', 'untried']
    
    def __init__(self, state: Tuple, parent: Optional['MCTSNode'] = None, action: int = -1):
        self.state = state
        self.parent = parent
        self.action = action
        self.visits = 0
        self.value = 0.0
        self.children: Dict[int, 'MCTSNode'] = {}
        self.untried = list(range(N_ACTIONS))
    
    def is_expanded(self) -> bool:
        return len(self.untried) == 0
    
    def best_child(self, c: float = 1.414) -> 'MCTSNode':
        best_score = float('-inf')
        best_child = None
        
        for child in self.children.values():
            if child.visits == 0:
                score = float('inf')
            else:
                exploitation = child.value / child.visits
                exploration = c * math.sqrt(math.log(self.visits) / child.visits)
                score = exploitation + exploration
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child if best_child else list(self.children.values())[0]
    
    def update(self, reward: float):
        self.visits += 1
        self.value += reward


class MCTSAgent:
    """Lightweight MCTS for submission."""
    
    def __init__(self, n_sims: int = 20, max_depth: int = 10, gamma: float = 0.95, c: float = 1.414):
        self.n_sims = n_sims
        self.max_depth = max_depth
        self.gamma = gamma
        self.c = c  # Make c configurable from saved model
    
    def discretize(self, obs: np.ndarray) -> Tuple:
        """Simple state hashing."""
        return tuple(np.round(obs, decimals=0).astype(int))
    
    def select_action(self, obs: np.ndarray, rng: np.random.Generator) -> int:
        """
        MCTS action selection with random rollouts.
        Note: This is a simplified version that doesn't use the environment model.
        For actual submission, you might want to use a learned rollout policy.
        """
        state = self.discretize(obs)
        root = MCTSNode(state)
        
        # Run simulations
        for _ in range(self.n_sims):
            # Selection
            node = root
            depth = 0
            
            while node.is_expanded() and depth < self.max_depth:
                if not node.children:
                    break
                node = node.best_child(self.c)
                depth += 1
            
            # Expansion
            if node.untried and depth < self.max_depth:
                a = node.untried.pop(rng.integers(len(node.untried)))
                child_state = (node.state, a)
                child = MCTSNode(child_state, node, a)
                node.children[a] = child
                node = child
            
            # Simulation (random rollout approximation)
            reward = self._simulate(node, rng)
            
            # Backpropagation
            while node:
                node.update(reward)
                node = node.parent
        
        # Select best action by visit count
        if not root.children:
            return int(rng.integers(N_ACTIONS))
        
        best_action = max(root.children.items(), key=lambda x: x[1].visits)[0]
        return best_action
    
    def _simulate(self, node: MCTSNode, rng: np.random.Generator) -> float:
        """
        Simulated rollout using random policy.
        Returns estimated value.
        """
        # Simple heuristic: estimate value from state features
        if isinstance(node.state, tuple) and len(node.state) >= 18:
            obs = np.array(node.state[:18])
            # Reward for seeing the box (sensor activation)
            sensor_sum = np.sum(obs[:16])
            # Penalty for being stuck
            stuck_penalty = -200 if obs[17] > 0.5 else 0
            
            return sensor_sum * 5 + stuck_penalty
        
        return 0.0


# Global agent
_AGENT = None


def _load_agent():
    global _AGENT
    if _AGENT is None:
        # Look for the model file in the same directory as this script
        model_path = os.path.join(os.path.dirname(__file__), "mcts_agent.pkl")
        
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Load configuration from saved pickle
                n_sims = data.get('n_simulations', 25)
                config = data.get('config', {})
                max_depth = config.get('max_depth', 10)
                c_exploration = config.get('c_exploration', 1.414)
                
                _AGENT = MCTSAgent(
                    n_sims=n_sims,
                    max_depth=max_depth,
                    c=c_exploration
                )
            except Exception as e:
                print(f"Warning: Failed to load model from {model_path}: {e}")
                print("Falling back to default MCTS agent.")
                _AGENT = MCTSAgent(n_sims=25, max_depth=10)
        else:
            # Fallback to defaults if no pickle file found
            _AGENT = MCTSAgent(n_sims=25, max_depth=10)
            
    return _AGENT


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    MCTS policy.
    NOTE: This uses simulated rollouts, not actual environment interaction.
    For better performance, consider using a learned rollout policy.
    """
    agent = _load_agent()
    action_idx = agent.select_action(obs, rng)
    return ACTIONS[action_idx]
