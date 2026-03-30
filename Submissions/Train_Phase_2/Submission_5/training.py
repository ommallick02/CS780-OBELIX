"""
Training script for Q-Learning with State Clustering.
Run this locally to train and save the agent.
"""

import argparse
import pickle
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Sequence
import numpy as np
from tqdm import tqdm

# Actions matching the OBELIX environment
ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")


class StateClusterer:
    """Online clustering of observation vectors."""
    
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
        
        if best_idx >= 0 and min_dist < self.similarity_threshold:
            alpha = 0.1
            self.clusters[best_idx] = (1 - alpha) * self.clusters[best_idx] + alpha * obs
            self.cluster_counts[best_idx] += 1
            return best_idx
        
        if len(self.clusters) < self.max_clusters:
            self.clusters.append(obs.copy())
            self.cluster_counts.append(1)
            return len(self.clusters) - 1
        
        if best_idx >= 0:
            self.cluster_counts[best_idx] += 1
            return best_idx
        
        return 0
    
    def n_clusters(self) -> int:
        return len(self.clusters)
    
    def save(self, path: str):
        data = {
            'clusters': self.clusters,
            'cluster_counts': self.cluster_counts,
            'max_clusters': self.max_clusters,
            'similarity_threshold': self.similarity_threshold
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.clusters = data['clusters']
        self.cluster_counts = data['cluster_counts']
        self.max_clusters = data['max_clusters']
        self.similarity_threshold = data['similarity_threshold']


class QLearningAgent:
    """Tabular Q-learning with eligibility traces."""
    
    def __init__(
        self,
        n_actions: int = 5,
        learning_rate: float = 0.2,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.05,
        eligibility_decay: float = 0.9,
        use_eligibility_traces: bool = True
    ):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lambda_ = eligibility_decay
        self.use_traces = use_eligibility_traces
        
        self.q_table: Dict[Tuple[int, int], float] = defaultdict(float)
        self.eligibility: Dict[Tuple[int, int], float] = defaultdict(float)
        self.clusterer = StateClusterer(max_clusters=200, similarity_threshold=0.15)
        
        self.prev_cluster_id: int = -1
        self.prev_action: int = -1
        self.episode_count = 0
        self.total_steps = 0
    
    def select_action(self, obs: np.ndarray, rng: np.random.Generator, training: bool = True) -> int:
        cluster_id = self.clusterer.get_cluster_id(obs)
        
        if training and rng.random() < self.epsilon:
            action = int(rng.integers(self.n_actions))
        else:
            # Get Q-values for all actions, defaulting to 0.0 if not seen
            q_values = [self.q_table.get((cluster_id, a), 0.0) for a in range(self.n_actions)]
            max_q = max(q_values)
            
            # Find all actions with max Q-value
            best_actions = [a for a, q in enumerate(q_values) if abs(q - max_q) < 1e-9]
            
            # Fallback: if somehow empty, use all actions
            if len(best_actions) == 0:
                best_actions = list(range(self.n_actions))
            
            # Random tie-breaking
            action = int(rng.choice(best_actions))
        
        self.prev_cluster_id = cluster_id
        self.prev_action = action
        return action
    
    def update(self, obs: np.ndarray, action: int, reward: float, 
               next_obs: np.ndarray, done: bool, rng: np.random.Generator):
        cluster_id = self.clusterer.get_cluster_id(obs)
        next_cluster_id = self.clusterer.get_cluster_id(next_obs)
        
        current_q = self.q_table.get((cluster_id, action), 0.0)
        next_q_values = [self.q_table.get((next_cluster_id, a), 0.0) for a in range(self.n_actions)]
        next_max_q = max(next_q_values) if not done else 0.0
        
        td_target = reward + self.gamma * next_max_q
        td_error = td_target - current_q
        
        if self.use_traces:
            self.eligibility[(cluster_id, action)] += 1.0
            for (s, a), e in list(self.eligibility.items()):
                self.q_table[(s, a)] = self.q_table.get((s, a), 0.0) + self.lr * td_error * e
                self.eligibility[(s, a)] = e * self.gamma * self.lambda_
            if done:
                self.eligibility.clear()
        else:
            self.q_table[(cluster_id, action)] = current_q + self.lr * td_error
        
        self.total_steps += 1
        
        if done:
            self.episode_count += 1
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            if self.use_traces:
                self.eligibility.clear()
    
    def save(self, path: str):
        data = {
            'q_table': dict(self.q_table),
            'clusterer_path': path + '.clusters',
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'hyperparams': {
                'lr': self.lr,
                'gamma': self.gamma,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min,
                'lambda': self.lambda_,
                'use_traces': self.use_traces
            }
        }
        self.clusterer.save(data['clusterer_path'])
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved agent to {path} (clusters: {self.clusterer.n_clusters()})")
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.q_table = defaultdict(float, data['q_table'])
        self.epsilon = data['epsilon']
        self.episode_count = data['episode_count']
        self.total_steps = data['total_steps']
        self.clusterer.load(data['clusterer_path'])


def evaluate_agent(agent, OBELIX, difficulty, wall_obstacles, 
                   n_episodes=10, max_steps=1000, seed=99999, 
                   progress_bar=None):
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
            action_idx = agent.select_action(obs, rng, training=False)
            action = ACTIONS[action_idx]
            obs, reward, done = env.step(action, render=False)
            total_reward += reward
            if done:
                break
        
        rewards.append(total_reward)
        
        if progress_bar is not None:
            progress_bar.update(1)
    
    return np.mean(rewards)


def train(
    obelix_py_path: str,
    episodes: int = 3000,
    max_steps: int = 1000,
    difficulty: int = 0,
    wall_obstacles: bool = False,
    save_path: str = "agent.pkl",
    eval_interval: int = 100
):
    """Main training loop with progress bars."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OBELIX = mod.OBELIX
    
    agent = QLearningAgent(
        learning_rate=0.2,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.05,
        use_eligibility_traces=True
    )
    
    print(f"\n{'='*60}")
    print(f"Training Q-Learning Agent")
    print(f"{'='*60}")
    print(f"Episodes: {episodes}")
    print(f"Difficulty: {difficulty} (0=static, 2=blinking, 3=moving+blinking)")
    print(f"Wall obstacles: {wall_obstacles}")
    print(f"Evaluation interval: every {eval_interval} episodes")
    print(f"{'='*60}\n")
    
    best_eval_reward = float('-inf')
    
    with tqdm(total=episodes, desc="Training", unit="ep", 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        
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
                action_idx = agent.select_action(obs, rng, training=True)
                action = ACTIONS[action_idx]
                next_obs, reward, done = env.step(action, render=False)
                episode_reward += reward
                
                agent.update(obs, action_idx, reward, next_obs, done, rng)
                obs = next_obs
                
                if done:
                    break
            
            pbar.update(1)
            pbar.set_postfix({
                'reward': f'{episode_reward:.1f}',
                'epsilon': f'{agent.epsilon:.3f}',
                'clusters': agent.clusterer.n_clusters(),
                'q_entries': len(agent.q_table)
            })
            
            if (episode + 1) % eval_interval == 0:
                pbar.set_description("Evaluating...")
                
                with tqdm(total=10, desc="Eval", leave=False, unit="ep") as eval_pbar:
                    eval_reward = evaluate_agent(agent, OBELIX, difficulty, wall_obstacles, 
                                                 n_episodes=10, max_steps=max_steps, 
                                                 seed=10000 + episode, progress_bar=eval_pbar)
                
                pbar.set_description("Training")
                
                tqdm.write(f"\n[Ep {episode+1}/{episodes}] "
                          f"Train: {episode_reward:8.1f} | "
                          f"Eval: {eval_reward:8.1f} | "
                          f"Best: {max(best_eval_reward, eval_reward):8.1f} | "
                          f"Eps: {agent.epsilon:.3f} | "
                          f"Clusters: {agent.clusterer.n_clusters():3d}")
                
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    agent.save(save_path.replace('.pkl', '_best.pkl'))
                    tqdm.write(f"*** New best model saved! ***")
            
            if (episode + 1) % 500 == 0:
                agent.save(save_path.replace('.pkl', f'_ep{episode+1}.pkl'))
                tqdm.write(f"Checkpoint saved at episode {episode+1}")
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    agent.save(save_path)
    print(f"Final statistics:")
    print(f"  - Clusters: {agent.clusterer.n_clusters()}")
    print(f"  - Q-table entries: {len(agent.q_table)}")
    print(f"  - Best eval reward: {best_eval_reward:.2f}")
    print(f"  - Final epsilon: {agent.epsilon:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obelix_py", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--save_path", type=str, default="agent.pkl")
    
    args = parser.parse_args()
    
    train(
        obelix_py_path=args.obelix_py,
        episodes=args.episodes,
        max_steps=args.max_steps,
        difficulty=args.difficulty,
        wall_obstacles=args.wall_obstacles,
        save_path=args.save_path
    )