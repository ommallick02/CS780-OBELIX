"""Training script for VPG (Vanilla Policy Gradient) with learned Baseline on OBELIX."""

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

class Actor(nn.Module):
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
        return torch.softmax(self.net(x), dim=-1)
    
    def sample_action(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_idx, log_prob)."""
        probs = self.forward(x)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, input_dim: int = 18, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

class VPGAgent:
    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 128,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        
        # Actor (policy)
        self.actor = Actor(input_dim, hidden_dim).to(self.device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Critic (value function baseline)
        self.critic = Critic(input_dim, hidden_dim).to(self.device)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.episode_count = 0
        self.total_steps = 0
    
    def select_action(self, obs: np.ndarray) -> Tuple[int, float]:
        """Returns (action_idx, log_prob)."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob = self.actor.sample_action(obs_tensor)
        return action.item(), log_prob.item()
    
    def update(self, trajectories: List[Tuple]):
        """
        trajectories: List of (obs, action, log_prob, reward) tuples
        """
        # Unpack trajectory
        obs_list = [t[0] for t in trajectories]
        actions = torch.LongTensor([t[1] for t in trajectories]).to(self.device)
        rewards = [t[3] for t in trajectories]
        
        # Calculate returns (Monte Carlo)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Prepare observations tensor
        obs_tensor = torch.FloatTensor(np.array(obs_list)).to(self.device)
        
        # Update Critic (minimize MSE between predicted value and actual return)
        values = self.critic(obs_tensor)
        critic_loss = nn.functional.mse_loss(values, returns)
        
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.optimizer_critic.step()
        
        # Calculate advantages
        with torch.no_grad():
            advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update Actor (policy gradient with advantage)
        probs = self.actor(obs_tensor)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        
        actor_loss = -(log_probs * advantages).mean()
        
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.optimizer_actor.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'avg_return': returns.mean().item()
        }

def evaluate_agent(agent, OBELIX, difficulty, wall_obstacles,
                   n_episodes=10, max_steps=1000, seed=99999):
    """Evaluate agent deterministically."""
    agent.actor.eval()
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
                action_idx = agent.actor.sample_action(obs_tensor)[0].item()
            
            obs, reward, done = env.step(ACTIONS[action_idx], render=False)
            total_reward += reward
            
            if done:
                break
        
        rewards.append(total_reward)
    
    agent.actor.train()
    return np.mean(rewards), np.std(rewards)

def train(
    obelix_py: str,
    episodes: int = 5000,
    max_steps: int = 2000,
    difficulty: int = 3,
    wall_obstacles: bool = False,
    save_path: str = "vpg_agent.pt",
    eval_interval: int = 50,
):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OBELIX = mod.OBELIX

    agent = VPGAgent()
    
    print(f"\n{'='*60}")
    print(f"Training VPG (Vanilla Policy Gradient) with Baseline")
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
        trajectory = []
        episode_reward = 0.0
        done = False

        for step in range(max_steps):
            action_idx, log_prob = agent.select_action(obs)
            next_obs, reward, done = env.step(ACTIONS[action_idx], render=False)
            episode_reward += reward
            
            trajectory.append((obs.copy(), action_idx, log_prob, reward))
            
            obs = next_obs
            agent.total_steps += 1
            
            if done:
                break
        
        metrics = agent.update(trajectory)
        agent.episode_count += 1
        recent_rewards.append(episode_reward)

        # Evaluation
        if (ep + 1) % eval_interval == 0:
            eval_mean, eval_std = evaluate_agent(
                agent, OBELIX, difficulty, wall_obstacles,
                n_episodes=10, max_steps=max_steps, seed=10000 + ep
            )
            
            if eval_mean > best_eval_reward:
                best_eval_reward = eval_mean
                torch.save(agent.actor.state_dict(), save_path)
                print(f"\n[Ep {ep+1}] Train: {episode_reward:.1f} | "
                      f"Eval: {eval_mean:.1f}±{eval_std:.1f} | "
                      f"Best: {best_eval_reward:.1f} | SAVED")
            else:
                print(f"\n[Ep {ep+1}] Train: {episode_reward:.1f} | "
                      f"Eval: {eval_mean:.1f}±{eval_std:.1f} | "
                      f"Best: {best_eval_reward:.1f}")

    print(f"\nTraining Complete! Best eval reward: {best_eval_reward:.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obelix_py", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--save_path", type=str, default="vpg_agent.pt")
    parser.add_argument("--eval_interval", type=int, default=50)
    
    args = parser.parse_args()
    train(**vars(args))