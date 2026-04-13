"""Training script for REINFORCE (Monte Carlo Policy Gradient) on OBELIX."""

import argparse
import os
from typing import List, Tuple
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int = 18, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N_ACTIONS),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)
    
    def sample_action(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_idx, log_prob)."""
        probs = self.forward(x)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

class REINFORCEAgent:
    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        
        self.policy = PolicyNetwork(input_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.episode_count = 0
        self.total_steps = 0
    
    def select_action(self, obs: np.ndarray) -> Tuple[int, float]:
        """Returns (action_idx, log_prob)."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob = self.policy.sample_action(obs_tensor)
        return action.item(), log_prob.item()
    
    def update(self, trajectories: List[Tuple]):
        """
        trajectories: List of (obs, action, log_prob, reward) tuples for the episode
        """
        # Calculate returns (Monte Carlo)
        returns = []
        G = 0
        for _, _, _, reward in reversed(trajectories):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(self.device)
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss: -log_prob * return
        policy_loss = []
        for i, (obs, action, log_prob_old, _) in enumerate(trajectories):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action_tensor = torch.LongTensor([action]).to(self.device)
            
            # Recompute log_prob for gradient tracking
            probs = self.policy(obs_tensor)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(action_tensor)
            
            loss = -log_prob * returns[i]
            policy_loss.append(loss)
        
        total_loss = torch.stack(policy_loss).sum()
        
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        return {'loss': total_loss.item(), 'avg_return': returns.mean().item()}

def evaluate_agent(agent, OBELIX, difficulty, wall_obstacles, 
                   n_episodes=10, max_steps=1000, seed=99999):
    """Evaluate agent deterministically."""
    agent.policy.eval()
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
        total_reward = 0.0
        done = False
        
        for _ in range(max_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action_idx = agent.policy.sample_action(obs_tensor)[0].item()
            
            obs, reward, done = env.step(ACTIONS[action_idx], render=False)
            total_reward += reward
            
            if done:
                break
        
        rewards.append(total_reward)
    
    agent.policy.train()
    return np.mean(rewards), np.std(rewards)

def train(
    obelix_py: str,
    episodes: int = 5000,
    max_steps: int = 2000,
    difficulty: int = 3,
    wall_obstacles: bool = False,
    save_path: str = "reinforce_agent.pt",
    eval_interval: int = 50,
):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OBELIX = mod.OBELIX

    agent = REINFORCEAgent()
    
    print(f"\n{'='*60}")
    print(f"Training REINFORCE (Monte Carlo Policy Gradient)")
    print(f"{'='*60}")
    print(f"Episodes: {episodes}")
    print(f"Difficulty: {difficulty}")
    print(f"Device: {agent.device}")
    print(f"{'='*60}\n")

    best_eval_reward = float('-inf')
    recent_rewards = deque(maxlen=100)

    for ep in tqdm(range(episodes), desc="Training"):
        env = OBELIX(
            scaling_factor=5,
            arena_size=500,
            max_steps=max_steps,
            wall_obstacles=wall_obstacles,
            difficulty=difficulty,
            box_speed=2,
            seed=ep
        )
        
        obs = env.reset(seed=ep)
        trajectory = []  # Store (obs, action, log_prob, reward)
        episode_reward = 0.0
        done = False

        for step in range(max_steps):
            action_idx, log_prob = agent.select_action(obs)
            next_obs, reward, done = env.step(ACTIONS[action_idx], render=False)
            episode_reward += reward
            
            # Store transition (without obs for memory efficiency, store index if needed)
            trajectory.append((obs.copy(), action_idx, log_prob, reward))
            
            obs = next_obs
            agent.total_steps += 1
            
            if done:
                break
        
        # Update policy using collected trajectory
        metrics = agent.update(trajectory)
        agent.episode_count += 1
        recent_rewards.append(episode_reward)

        # Evaluation
        if (ep + 1) % eval_interval == 0:
            eval_mean, eval_std = evaluate_agent(
                agent, OBELIX, difficulty, wall_obstacles,
                n_episodes=10, max_steps=max_steps, seed=10000 + ep
            )
            
            avg_recent = sum(recent_rewards) / len(recent_rewards)
            
            if eval_mean > best_eval_reward:
                best_eval_reward = eval_mean
                torch.save(agent.policy.state_dict(), save_path)
                print(f"\n[Ep {ep+1}] Train: {episode_reward:.1f} | "
                      f"Eval: {eval_mean:.1f}±{eval_std:.1f} | "
                      f"Best: {best_eval_reward:.1f} | SAVED")
            else:
                print(f"\n[Ep {ep+1}] Train: {episode_reward:.1f} | "
                      f"Eval: {eval_mean:.1f}±{eval_std:.1f} | "
                      f"Best: {best_eval_reward:.1f}")

    print(f"\nTraining Complete! Best eval reward: {best_eval_reward:.1f}")
    print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obelix_py", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--save_path", type=str, default="reinforce_agent.pt")
    parser.add_argument("--eval_interval", type=int, default=50)
    
    args = parser.parse_args()
    train(**vars(args))
