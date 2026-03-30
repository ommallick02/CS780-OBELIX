"""
Training script for Dyna-Q.
Combines Q-learning with learned model for planning.

python training_dyna_q.py --obelix_py ./obelix.py --episodes 3000 --wall_obstacles --planning_steps 20 --difficulty 2

"""

import argparse
import pickle
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Sequence, Set
import numpy as np
from tqdm import tqdm

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = len(ACTIONS)


class StateDiscretizer:
    """
    Discretize continuous observations into bins for tabular methods.
    Uses simple binning of the 18-dimensional observation vector.
    """
    
    def __init__(self, n_bins: int = 2):
        """
        n_bins=2 means binary: 0 or 1 (exact match for OBELIX's binary sensors)
        For OBELIX, observations are already binary (0/1), so we can use as-is
        or add slight tolerance for near-binary values.
        """
        self.n_bins = n_bins
    
    def discretize(self, obs: np.ndarray) -> Tuple:
        """
        Convert observation to discrete state representation.
        For OBELIX, we can use the binary vector directly as tuple.
        """
        # Round to nearest bin and convert to tuple (hashable)
        discrete = np.round(obs).astype(int)
        return tuple(discrete)
    
    def discretize_batch(self, observations: List[np.ndarray]) -> List[Tuple]:
        """Discretize multiple observations."""
        return [self.discretize(obs) for obs in observations]


class EnvironmentModel:
    """
    Learned model of the environment dynamics.
    Stores observed transitions: (state, action) -> (next_state, reward)
    """
    
    def __init__(self):
        # Deterministic model: store observed transitions
        self.transitions: Dict[Tuple[Tuple, int], Tuple[Tuple, float]] = {}
        
        # For stochastic environments, we could store counts/probabilities
        self.visit_counts: Dict[Tuple[Tuple, int], int] = defaultdict(int)
        
        # Track which states have been visited
        self.known_states: Set[Tuple] = set()
    
    def update(self, state: Tuple, action: int, next_state: Tuple, reward: float):
        """Update model with observed transition."""
        key = (state, action)
        self.transitions[key] = (next_state, reward)
        self.visit_counts[key] += 1
        self.known_states.add(state)
        self.known_states.add(next_state)
    
    def sample_transition(self, state: Tuple, action: int) -> Tuple[Tuple, float]:
        """
        Query model for predicted transition.
        Returns (next_state, reward) or raises KeyError if unknown.
        """
        key = (state, action)
        if key in self.transitions:
            return self.transitions[key]
        raise KeyError(f"Unknown transition: {state}, {action}")
    
    def is_known(self, state: Tuple, action: int) -> bool:
        """Check if we've seen this transition before."""
        return (state, action) in self.transitions
    
    def get_known_state_actions(self) -> List[Tuple[Tuple, int]]:
        """Get all state-action pairs we've experienced."""
        return list(self.transitions.keys())
    
    def n_known_transitions(self) -> int:
        """Number of unique transitions learned."""
        return len(self.transitions)
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'transitions': self.transitions,
                'visit_counts': dict(self.visit_counts),
                'known_states': list(self.known_states)
            }, f)
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.transitions = data['transitions']
        self.visit_counts = defaultdict(int, data['visit_counts'])
        self.known_states = set(data['known_states'])


class DynaQAgent:
    """
    Dyna-Q agent with integrated planning.
    """
    
    def __init__(
        self,
        n_actions: int = 5,
        learning_rate: float = 0.2,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.05,
        planning_steps: int = 10,  # Number of planning steps per real step
        n_bins: int = 2
    ):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.planning_steps = planning_steps
        
        # Q-table: (state_tuple, action) -> value
        self.q_table: Dict[Tuple[Tuple, int], float] = defaultdict(float)
        
        # State discretizer
        self.discretizer = StateDiscretizer(n_bins=n_bins)
        
        # Learned model
        self.model = EnvironmentModel()
        
        # Statistics
        self.episode_count = 0
        self.total_steps = 0
        self.planning_updates = 0
    
    def select_action(self, state: Tuple, rng: np.random.Generator, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        """
        if training and rng.random() < self.epsilon:
            return int(rng.integers(self.n_actions))
        
        # Greedy: select action with max Q-value
        q_values = [self.q_table.get((state, a), 0.0) for a in range(self.n_actions)]
        max_q = max(q_values)
        
        # Find all actions with max value (tie-breaking)
        best_actions = [a for a, q in enumerate(q_values) if abs(q - max_q) < 1e-9]
        
        # Fallback if empty
        if len(best_actions) == 0:
            best_actions = list(range(self.n_actions))
        
        return int(rng.choice(best_actions))
    
    def update_q(self, state: Tuple, action: int, reward: float, next_state: Tuple):
        """
        Standard Q-learning update.
        """
        current_q = self.q_table.get((state, action), 0.0)
        
        # Max Q for next state
        next_q_values = [self.q_table.get((next_state, a), 0.0) for a in range(self.n_actions)]
        next_max_q = max(next_q_values) if next_q_values else 0.0
        
        # Q-learning update
        td_target = reward + self.gamma * next_max_q
        self.q_table[(state, action)] = current_q + self.lr * (td_target - current_q)
    
    def planning(self, rng: np.random.Generator):
        """
        Perform planning updates using learned model.
        This is the key Dyna-Q component.
        """
        if self.model.n_known_transitions() == 0:
            return
        
        # Sample random previously observed state-action pairs
        known_pairs = self.model.get_known_state_actions()
        n_pairs = len(known_pairs)
        
        for _ in range(self.planning_steps):
            # Randomly select a state-action pair from experience
            # Use integer indexing instead of rng.choice to avoid array issues
            idx = int(rng.integers(0, n_pairs))
            state, action = known_pairs[idx]
            
            # Query model for predicted outcome
            try:
                next_state, reward = self.model.sample_transition(state, action)
                
                # Update Q-value using simulated experience
                self.update_q(state, action, reward, next_state)
                self.planning_updates += 1
                
            except KeyError:
                continue  # Skip if somehow missing
    
    def step(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, 
             done: bool, rng: np.random.Generator):
        """
        Execute one step of Dyna-Q: real update + planning.
        """
        # Discretize states
        state = self.discretizer.discretize(obs)
        next_state = self.discretizer.discretize(next_obs)
        
        # Real experience: update Q and model
        self.update_q(state, action, reward, next_state)
        self.model.update(state, action, next_state, reward)
        
        # Planning: simulate additional updates
        self.planning(rng)
        
        self.total_steps += 1
        
        # Decay epsilon at episode end
        if done:
            self.episode_count += 1
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_value(self, state: Tuple) -> float:
        """Get max Q-value for a state."""
        q_values = [self.q_table.get((state, a), 0.0) for a in range(self.n_actions)]
        return max(q_values) if q_values else 0.0
    
    def save(self, path: str):
        """Save agent state."""
        data = {
            'q_table': dict(self.q_table),
            'model_path': path + '.model',
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'planning_updates': self.planning_updates,
            'hyperparams': {
                'lr': self.lr,
                'gamma': self.gamma,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min,
                'planning_steps': self.planning_steps
            }
        }
        self.model.save(data['model_path'])
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved Dyna-Q agent to {path}")
        print(f"  - Q-table entries: {len(self.q_table)}")
        print(f"  - Model transitions: {self.model.n_known_transitions()}")
        print(f"  - Planning updates: {self.planning_updates}")
    
    def load(self, path: str):
        """Load agent state."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = defaultdict(float, data['q_table'])
        self.epsilon = data['epsilon']
        self.episode_count = data['episode_count']
        self.total_steps = data['total_steps']
        self.planning_updates = data.get('planning_updates', 0)
        
        # Load model
        self.model.load(data['model_path'])
        print(f"Loaded Dyna-Q agent from {path}")


def evaluate_agent(agent, OBELIX, difficulty, wall_obstacles,
                   n_episodes=10, max_steps=1000, seed=99999):
    """Evaluate agent without exploration."""
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
        
        for _ in range(max_steps):
            state = agent.discretizer.discretize(obs)
            action_idx = agent.select_action(state, rng, training=False)
            action = ACTIONS[action_idx]
            
            obs, reward, done = env.step(action, render=False)
            total_reward += reward
            
            if done:
                break
        
        rewards.append(total_reward)
    
    return np.mean(rewards)


def train(
    obelix_py_path: str,
    episodes: int = 1000,  # Fewer episodes needed due to planning
    max_steps: int = 1000,
    difficulty: int = 0,
    wall_obstacles: bool = False,
    save_path: str = "dyna_q_agent.pkl",
    eval_interval: int = 50,
    planning_steps: int = 10,
    lr: float = 0.2
):
    """Main Dyna-Q training loop."""
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OBELIX = mod.OBELIX
    
    agent = DynaQAgent(
        n_actions=5,
        learning_rate=lr,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.05,
        planning_steps=planning_steps
    )
    
    print(f"\n{'='*60}")
    print(f"Training Dyna-Q Agent")
    print(f"{'='*60}")
    print(f"Episodes: {episodes}")
    print(f"Planning steps per real step: {planning_steps}")
    print(f"Difficulty: {difficulty}")
    print(f"Wall obstacles: {wall_obstacles}")
    print(f"{'='*60}\n")
    
    best_eval_reward = float('-inf')
    recent_rewards = []
    
    with tqdm(total=episodes, desc="Dyna-Q Training", unit="ep") as pbar:
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
            
            for step in range(max_steps):
                # Discretize current state
                state = agent.discretizer.discretize(obs)
                
                # Select action
                action_idx = agent.select_action(state, rng, training=True)
                action = ACTIONS[action_idx]
                
                # Execute
                next_obs, reward, done = env.step(action, render=False)
                episode_reward += reward
                
                # Dyna-Q update (real + planning)
                agent.step(obs, action_idx, reward, next_obs, done, rng)
                
                obs = next_obs
                
                if done:
                    break
            
            recent_rewards.append(episode_reward)
            if len(recent_rewards) > 100:
                recent_rewards.pop(0)
            
            pbar.update(1)
            pbar.set_postfix({
                'reward': f'{episode_reward:.1f}',
                'avg': f'{np.mean(recent_rewards):.1f}' if recent_rewards else '0.0',
                'epsilon': f'{agent.epsilon:.3f}',
                'model_size': agent.model.n_known_transitions(),
                'q_size': len(agent.q_table),
                'planning': agent.planning_updates
            })
            
            if (episode + 1) % eval_interval == 0:
                pbar.set_description("Evaluating...")
                eval_reward = evaluate_agent(agent, OBELIX, difficulty, wall_obstacles,
                                           n_episodes=10, max_steps=max_steps,
                                           seed=10000 + episode)
                pbar.set_description("Dyna-Q Training")
                
                tqdm.write(f"\n[Ep {episode+1}/{episodes}] "
                          f"Train: {episode_reward:8.1f} | "
                          f"Eval: {eval_reward:8.1f} | "
                          f"Best: {max(best_eval_reward, eval_reward):8.1f} | "
                          f"Model: {agent.model.n_known_transitions():4d} | "
                          f"Planning: {agent.planning_updates:6d}")
                
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    agent.save(save_path.replace('.pkl', '_best.pkl'))
                    tqdm.write(f"*** New best model saved! ***")
            
            if (episode + 1) % 250 == 0:
                agent.save(save_path.replace('.pkl', f'_ep{episode+1}.pkl'))
                tqdm.write(f"Checkpoint saved")
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    agent.save(save_path)
    print(f"Best eval reward: {best_eval_reward:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obelix_py", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=1000,
                       help="Fewer episodes needed due to planning")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--save_path", type=str, default="dyna_q_agent.pkl")
    parser.add_argument("--planning_steps", type=int, default=10,
                       help="Planning steps per real step")
    parser.add_argument("--lr", type=float, default=0.2)
    
    args = parser.parse_args()
    
    train(
        obelix_py_path=args.obelix_py,
        episodes=args.episodes,
        max_steps=args.max_steps,
        difficulty=args.difficulty,
        wall_obstacles=args.wall_obstacles,
        save_path=args.save_path,
        planning_steps=args.planning_steps,
        lr=args.lr
    )