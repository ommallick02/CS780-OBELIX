"""
Training script for REINFORCE (Monte Carlo Policy Gradient).
Learns a policy network that maps observations directly to action probabilities.

python training_reinforce.py --obelix_py ./obelix.py --episodes 3000 --difficulty 2 --wall_obstacles --hidden_dim 128
"""

import argparse
import pickle
import os
from typing import List, Tuple, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = len(ACTIONS)


class PolicyNetwork(nn.Module):
    """
    Neural network policy for REINFORCE.
    Outputs probability distribution over 5 actions.
    """
    
    def __init__(self, input_dim: int = 18, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N_ACTIONS),
            nn.Softmax(dim=-1)  # Output probabilities
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return action probabilities."""
        return self.net(x)
    
    def select_action(self, obs: np.ndarray, rng: np.random.Generator, 
                      device: str = "cpu") -> Tuple[int, torch.Tensor]:
        """
        Sample action from policy.
        Returns: (action_index, log_probability)
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        probs = self.forward(obs_tensor)
        
        # Sample from distribution
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action_idx))
        
        return action_idx, log_prob


class REINFORCEAgent:
    """
    REINFORCE agent with baseline for variance reduction.
    """
    
    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        use_baseline: bool = True
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gamma = gamma
        self.use_baseline = use_baseline
        
        # Policy network
        self.policy = PolicyNetwork(input_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Baseline network (state value function to reduce variance)
        if use_baseline:
            self.baseline = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ).to(self.device)
            self.baseline_optimizer = optim.Adam(self.baseline.parameters(), lr=learning_rate)
        
        self.episode_count = 0
        self.total_steps = 0
    
    def run_episode(self, env, seed: int, max_steps: int, render: bool = False) -> Tuple[float, List]:
        """
        Run one episode, collecting trajectory.
        Returns: (total_reward, trajectory)
        """
        rng = np.random.default_rng(seed)
        obs = env.reset(seed=seed)
        
        trajectory = []  # List of (obs, action_idx, log_prob, reward)
        total_reward = 0.0
        done = False
        
        for step in range(max_steps):
            # Select action
            action_idx, log_prob = self.policy.select_action(obs, rng, self.device)
            action = ACTIONS[action_idx]
            
            # Execute action
            next_obs, reward, done = env.step(action, render=render)
            total_reward += reward
            
            # Store transition
            trajectory.append({
                'obs': obs.copy(),
                'action_idx': action_idx,
                'log_prob': log_prob,
                'reward': reward,
                'done': done
            })
            
            obs = next_obs
            self.total_steps += 1
            
            if done:
                break
        
        self.episode_count += 1
        return total_reward, trajectory
    
    def compute_returns(self, trajectory: List) -> List[float]:
        """
        Compute discounted returns G_t for each timestep.
        """
        returns = []
        G = 0.0
        
        # Work backwards through episode
        for t in reversed(range(len(trajectory))):
            reward = trajectory[t]['reward']
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        return returns
    
    def update(self, trajectory: List):
        """
        Update policy using REINFORCE with optional baseline.
        """
        if len(trajectory) == 0:
            return
        
        returns = self.compute_returns(trajectory)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns for stability
        returns_mean = returns_tensor.mean()
        returns_std = returns_tensor.std() + 1e-8
        normalized_returns = (returns_tensor - returns_mean) / returns_std
        
        # Compute advantages (returns - baseline)
        if self.use_baseline:
            obs_batch = torch.FloatTensor([t['obs'] for t in trajectory]).to(self.device)
            baseline_values = self.baseline(obs_batch).squeeze()
            advantages = normalized_returns - baseline_values.detach()
            
            # Update baseline
            baseline_loss = nn.functional.mse_loss(baseline_values, normalized_returns)
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()
        else:
            advantages = normalized_returns
        
        # Policy gradient loss: -log_prob * advantage
        log_probs = torch.stack([t['log_prob'] for t in trajectory])
        policy_loss = -(log_probs * advantages).sum()
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return policy_loss.item()
    
    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'use_baseline': self.use_baseline,
            'baseline_state_dict': self.baseline.state_dict() if self.use_baseline else None,
            'baseline_optimizer_state_dict': self.baseline_optimizer.state_dict() if self.use_baseline else None,
        }, path)
        print(f"Saved agent to {path}")
    
    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_count = checkpoint['episode_count']
        self.total_steps = checkpoint['total_steps']
        
        if self.use_baseline and checkpoint.get('baseline_state_dict'):
            self.baseline.load_state_dict(checkpoint['baseline_state_dict'])
            self.baseline_optimizer.load_state_dict(checkpoint['baseline_optimizer_state_dict'])
        
        print(f"Loaded agent from {path}")


def evaluate_agent(agent, OBELIX, difficulty, wall_obstacles, 
                   n_episodes=10, max_steps=1000, seed=99999):
    """Evaluate agent without exploration (deterministic)."""
    rewards = []
    
    # Set policy to eval mode
    agent.policy.eval()
    
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
        
        rng = np.random.default_rng(seed + i)
        obs = env.reset(seed=seed + i)
        total_reward = 0.0
        done = False
        
        for _ in range(max_steps):
            # Greedy action selection (max probability)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                probs = agent.policy(obs_tensor).cpu().numpy()[0]
            
            action_idx = int(np.argmax(probs))
            action = ACTIONS[action_idx]
            
            obs, reward, done = env.step(action, render=False)
            total_reward += reward
            
            if done:
                break
        
        rewards.append(total_reward)
    
    # Back to train mode
    agent.policy.train()
    return np.mean(rewards)


def train(
    obelix_py_path: str,
    episodes: int = 2000,
    max_steps: int = 1000,
    difficulty: int = 0,
    wall_obstacles: bool = False,
    save_path: str = "reinforce_agent.pth",
    eval_interval: int = 50,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    use_baseline: bool = True
):
    """Main REINFORCE training loop."""
    
    # Import OBELIX
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OBELIX = mod.OBELIX
    
    # Initialize agent
    agent = REINFORCEAgent(
        input_dim=18,
        hidden_dim=hidden_dim,
        learning_rate=lr,
        gamma=0.99,
        use_baseline=use_baseline
    )
    
    print(f"\n{'='*60}")
    print(f"Training REINFORCE Agent")
    print(f"{'='*60}")
    print(f"Episodes: {episodes}")
    print(f"Difficulty: {difficulty}")
    print(f"Wall obstacles: {wall_obstacles}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Learning rate: {lr}")
    print(f"Use baseline: {use_baseline}")
    print(f"Device: {agent.device}")
    print(f"{'='*60}\n")
    
    best_eval_reward = float('-inf')
    recent_rewards = []  # For tracking moving average
    
    with tqdm(total=episodes, desc="Training", unit="ep") as pbar:
        for episode in range(episodes):
            # Create environment
            env = OBELIX(
                scaling_factor=5,
                arena_size=500,
                max_steps=max_steps,
                wall_obstacles=wall_obstacles,
                difficulty=difficulty,
                box_speed=2,
                seed=episode
            )
            
            # Run episode and collect trajectory
            episode_reward, trajectory = agent.run_episode(env, seed=episode, max_steps=max_steps)
            
            # Update policy
            if len(trajectory) > 0:
                loss = agent.update(trajectory)
            
            recent_rewards.append(episode_reward)
            if len(recent_rewards) > 100:
                recent_rewards.pop(0)
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'reward': f'{episode_reward:.1f}',
                'avg100': f'{np.mean(recent_rewards):.1f}' if recent_rewards else '0.0',
                'steps': len(trajectory),
                'total_steps': agent.total_steps
            })
            
            # Evaluation
            if (episode + 1) % eval_interval == 0:
                pbar.set_description("Evaluating...")
                eval_reward = evaluate_agent(agent, OBELIX, difficulty, wall_obstacles, 
                                           n_episodes=10, max_steps=max_steps, 
                                           seed=10000 + episode)
                pbar.set_description("Training")
                
                tqdm.write(f"\n[Ep {episode+1}/{episodes}] "
                          f"Train: {episode_reward:8.1f} | "
                          f"Eval: {eval_reward:8.1f} | "
                          f"Best: {max(best_eval_reward, eval_reward):8.1f} | "
                          f"Steps: {agent.total_steps}")
                
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    agent.save(save_path.replace('.pt', '_best.pt'))
                    tqdm.write(f"*** New best model saved! ***")
            
            # Periodic checkpoint
            if (episode + 1) % 500 == 0:
                agent.save(save_path.replace('.pt', f'_ep{episode+1}.pt'))
                tqdm.write(f"Checkpoint saved at episode {episode+1}")
    
    # Final save
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    agent.save(save_path)
    print(f"Best evaluation reward: {best_eval_reward:.2f}")
    print(f"Total steps: {agent.total_steps}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obelix_py", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--save_path", type=str, default="reinforce_agent.pt")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no_baseline", action="store_true", 
                       help="Disable baseline for variance reduction")
    
    args = parser.parse_args()
    
    train(
        obelix_py_path=args.obelix_py,
        episodes=args.episodes,
        max_steps=args.max_steps,
        difficulty=args.difficulty,
        wall_obstacles=args.wall_obstacles,
        save_path=args.save_path,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        use_baseline=not args.no_baseline
    )