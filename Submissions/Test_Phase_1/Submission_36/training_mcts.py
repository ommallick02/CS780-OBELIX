"""
MCTS (Monte Carlo Tree Search) for OBELIX.
Uses online planning with random rollouts instead of learning a policy.
"""

import argparse
import os
import math
import copy
from typing import Dict, List, Tuple, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = len(ACTIONS)


class StateDiscretizer:
    """Discretize continuous observations for tree nodes."""
    
    def __init__(self, precision: int = 1):
        self.precision = precision
    
    def discretize(self, obs: np.ndarray) -> Tuple:
        """Convert observation to discrete state (hashable)."""
        # Round to reduce state space size
        discrete = np.round(obs, decimals=self.precision)
        return tuple(discrete.astype(float))
    
    def discretize_with_history(self, obs: np.ndarray, prev_action: int, history_len: int = 3) -> Tuple:
        """Discretize with action history for partial observability."""
        obs_tuple = self.discretize(obs)
        return (obs_tuple, prev_action)


class MCTSNode:
    """Node in the MCTS tree."""
    
    def __init__(self, state: Tuple, parent: Optional['MCTSNode'] = None, action: int = -1):
        self.state = state
        self.parent = parent
        self.action = action  # Action taken to reach this node
        
        self.visits = 0
        self.value = 0.0
        self.children: Dict[int, MCTSNode] = {}
        self.untried_actions = list(range(N_ACTIONS))
    
    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried."""
        return len(self.untried_actions) == 0
    
    def best_child(self, c: float = 1.414) -> 'MCTSNode':
        """
        Select best child using UCB1 formula.
        c = exploration constant (sqrt(2) by default)
        """
        choices_weights = []
        for child in self.children.values():
            if child.visits == 0:
                ucb = float('inf')
            else:
                exploitation = child.value / child.visits
                exploration = c * math.sqrt(math.log(self.visits) / child.visits)
                ucb = exploitation + exploration
            choices_weights.append(ucb)
        
        best_idx = int(np.argmax(choices_weights))
        return list(self.children.values())[best_idx]
    
    def update(self, reward: float):
        """Update node statistics."""
        self.visits += 1
        self.value += reward


class MCTSAgent:
    """
    MCTS Agent with online planning.
    No neural network - pure tree search with random rollouts.
    """
    
    def __init__(
        self,
        n_simulations: int = 50,  # Number of simulations per action selection
        max_depth: int = 20,      # Max depth for rollouts
        gamma: float = 0.99,      # Discount factor
        c_exploration: float = 1.414,  # UCB1 constant
        use_value_heuristic: bool = True  # Use simple heuristic for rollout values
    ):
        self.n_simulations = n_simulations
        self.max_depth = max_depth
        self.gamma = gamma
        self.c_exploration = c_exploration
        self.use_value_heuristic = use_value_heuristic
        
        self.discretizer = StateDiscretizer(precision=0)
        self.root = None
        self.prev_action = 0
        
        # Simple value cache for states (optional)
        self.value_cache: Dict[Tuple, float] = {}
    
    def select_action(self, obs: np.ndarray, env, rng: np.random.Generator, training: bool = True) -> int:
        """
        Select action using MCTS.
        env is passed to simulate rollouts.
        """
        # Create root node
        state = self.discretizer.discretize(obs)
        self.root = MCTSNode(state)
        
        # Run simulations
        for _ in range(self.n_simulations):
            node = self._select(self.root)
            reward = self._simulate(node, env, rng)
            self._backpropagate(node, reward)
        
        # Select best action (most visits or highest value)
        if not self.root.children:
            return int(rng.integers(N_ACTIONS))
        
        best_action = max(
            self.root.children.items(),
            key=lambda x: x[1].visits  # Choose by visit count (robust)
        )[0]
        
        return best_action
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Selection phase: Traverse tree using UCB1 until leaf node.
        """
        current = node
        depth = 0
        
        while current.is_fully_expanded() and depth < self.max_depth:
            if not current.children:
                break
            current = current.best_child(self.c_exploration)
            depth += 1
        
        # Expand if not fully expanded and not terminal
        if not current.is_fully_expanded() and depth < self.max_depth:
            return self._expand(current)
        
        return current
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expansion phase: Add a new child node.
        """
        if not node.untried_actions:
            return node
        
        # Randomly select untried action
        action = node.untried_actions.pop(np.random.randint(len(node.untried_actions)))
        
        # Create child node (state will be determined during simulation)
        child_state = (node.state, action)  # Use state-action as unique identifier
        child = MCTSNode(child_state, parent=node, action=action)
        node.children[action] = child
        
        return child
    
    def _simulate(self, node: MCTSNode, env, rng: np.random.Generator) -> float:
        """
        Simulation phase: Random rollout from node.
        Returns discounted cumulative reward.
        """
        # For OBELIX, we can't easily reset env to node's state
        # So we use the current env state and simulate forward
        # This is approximate but practical
        
        total_reward = 0.0
        discount = 1.0
        depth = 0
        
        # Random policy rollout
        for _ in range(self.max_depth):
            if depth >= self.max_depth:
                break
            
            action = int(rng.integers(N_ACTIONS))
            obs, reward, done = env.step(ACTIONS[action], render=False)  # <-- Fix here
            total_reward += discount * reward
            discount *= self.gamma
            depth += 1
            
            if done:
                break
        
        return total_reward
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """
        Backpropagation phase: Update statistics up the tree.
        """
        current = node
        while current is not None:
            current.update(reward)
            current = current.parent
    
    def rollout_value_heuristic(self, obs: np.ndarray) -> float:
        """
        Simple heuristic value for state (optional).
        Can guide rollouts if environment model is imperfect.
        """
        if not self.use_value_heuristic:
            return 0.0
        
        # Simple heuristic: closer to box is better
        # Box sensors are indices where obs is high
        sensor_sum = np.sum(obs[:16])  # Sonar sensors
        
        # Higher sensor activation = closer to box = higher value
        return sensor_sum * 10.0


def evaluate_agent(agent, OBELIX, difficulty, wall_obstacles,
                   n_episodes=10, max_steps=1000, seed=99999):
    """Evaluate MCTS agent."""
    rewards = []
    
    for i in range(n_episodes):
        env = OBELIX(
            scaling_factor=5,
            arena_size=500,
            max_steps=max_steps,
            wall_obstacles=wall_obstacles,
            difficulty=difficulty,
            box_speed=2,
            seed=seed + i
        )
        
        obs = env.reset(seed=seed + i)
        rng = np.random.default_rng(seed + i)
        total_reward = 0.0
        done = False
        step = 0
        
        while not done and step < max_steps:
            # MCTS action selection (expensive!)
            action_idx = agent.select_action(obs, env, rng, training=False)
            action = ACTIONS[action_idx]
            
            obs, reward, done = env.step(action, render=False)
            total_reward += reward
            step += 1
        
        rewards.append(total_reward)
    
    return np.mean(rewards)


def train(
    obelix_py_path: str,
    episodes: int = 100,  # Fewer episodes because MCTS is slow
    max_steps: int = 1000,
    difficulty: int = 0,
    wall_obstacles: bool = False,
    n_simulations: int = 30,
    save_path: str = "mcts_agent.pkl"
):
    """
    Train/evaluate MCTS agent.
    Note: MCTS doesn't really "train" - it plans online.
    We just evaluate its performance.
    """
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OBELIX = mod.OBELIX
    
    agent = MCTSAgent(n_simulations=n_simulations, max_depth=15)
    
    print(f"\n{'='*60}")
    print(f"Running MCTS Agent")
    print(f"{'='*60}")
    print(f"Episodes: {episodes}")
    print(f"Simulations per step: {n_simulations}")
    print(f"Difficulty: {difficulty}")
    print(f"{'='*60}\n")
    print("WARNING: MCTS is slow - each step runs multiple simulations!")
    
    rewards = []
    
    with tqdm(total=episodes, desc="MCTS", unit="ep") as pbar:
        for episode in range(episodes):
            env = OBELIX(
                scaling_factor=5,
                arena_size=500,
                max_steps=max_steps,
                wall_obstacles=wall_obstacles,
                difficulty=difficulty,
                box_speed=2,
                seed=episode
            )
            
            obs = env.reset(seed=episode)
            rng = np.random.default_rng(episode)
            episode_reward = 0.0
            done = False
            step = 0
            
            while not done and step < max_steps:
                # MCTS action selection
                action_idx = agent.select_action(obs, env, rng)
                action = ACTIONS[action_idx]
                
                obs, reward, done = env.step(action, render=False)
                episode_reward += reward
                step += 1
                
                if done:
                    break
            
            rewards.append(episode_reward)
            pbar.update(1)
            pbar.set_postfix({
                'reward': f'{episode_reward:.1f}',
                'avg': f'{np.mean(rewards):.1f}',
                'steps': step
            })
    
    print(f"\n{'='*60}")
    print("MCTS Evaluation Complete!")
    print(f"Mean reward: {np.mean(rewards):.2f}")
    print(f"Std reward: {np.std(rewards):.2f}")
    print(f"{'='*60}")
    
    # Save configuration (MCTS doesn't have learned parameters)
    import pickle
    with open(save_path, 'wb') as f:
        pickle.dump({
            'n_simulations': n_simulations,
            'mean_reward': np.mean(rewards),
            'config': {
                'max_depth': agent.max_depth,
                'c_exploration': agent.c_exploration
            }
        }, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obelix_py", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--n_simulations", type=int, default=30,
                       help="Number of MCTS simulations per step")
    parser.add_argument("--save_path", type=str, default="mcts_agent.pkl")
    
    args = parser.parse_args()
    
    train(
        obelix_py_path=args.obelix_py,
        episodes=args.episodes,
        max_steps=args.max_steps,
        difficulty=args.difficulty,
        wall_obstacles=args.wall_obstacles,
        n_simulations=args.n_simulations,
        save_path=args.save_path
    )