"""
Training script for PPO (Proximal Policy Optimization).
Uses clipped surrogate objective for stable policy updates.

python training_ppo.py --obelix_py ../../../obelix.py --wall_obstacles --clip_epsilon 0.1 --update_epochs 10 --entropy_coef 0.02 --batch_size 32 --n_steps 1024
"""

import argparse
import os
from typing import List, Tuple, Sequence, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = len(ACTIONS)


class ActorCriticNetwork(nn.Module):
    """
    Shared network for actor (policy) and critic (value function).
    """
    
    def __init__(self, input_dim: int = 18, hidden_dim: int = 128):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head: policy probabilities
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, N_ACTIONS),
            nn.Softmax(dim=-1)
        )
        
        # Critic head: state value
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (action_probs, state_value)."""
        features = self.shared(x)
        probs = self.actor(features)
        value = self.critic(features)
        return probs, value
    
    def get_action(self, obs: np.ndarray, device: str) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample action and return (action_idx, log_prob, value).
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        with torch.no_grad():
            probs, value = self.forward(obs_tensor)
        
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action_idx))
        
        return action_idx, log_prob, value.squeeze()


class PPOAgent:
    """
    PPO Agent with clipped surrogate objective.
    Collects trajectories and performs multiple epochs of updates.
    """
    
    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,  # GAE parameter
        clip_epsilon: float = 0.2,  # PPO clipping parameter
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,  # Number of optimization epochs per batch
        batch_size: int = 64,
        n_steps: int = 2048  # Steps to collect before update
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.n_steps = n_steps
        
        # Network
        self.network = ActorCriticNetwork(input_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Learning rate scheduler (optional decay)
        self.scheduler = None
        
        self.episode_count = 0
        self.total_steps = 0
        self.update_count = 0
    
    def collect_trajectories(self, env, seed: int, max_steps: int) -> Tuple[float, Dict]:
        """
        Collect trajectories for n_steps or until done.
        Returns: (episode_reward, trajectory_data)
        """
        rng = np.random.default_rng(seed)
        obs = env.reset(seed=seed)
        
        # Storage
        observations = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []
        
        episode_reward = 0.0
        step_count = 0
        
        for step in range(min(self.n_steps, max_steps)):
            # Select action
            action_idx, log_prob, value = self.network.get_action(obs, self.device)
            action = ACTIONS[action_idx]
            
            # Execute
            next_obs, reward, done = env.step(action, render=False)
            episode_reward += reward
            
            # Store
            observations.append(obs.copy())
            actions.append(action_idx)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(done)
            
            obs = next_obs
            step_count += 1
            self.total_steps += 1
            
            if done:
                break
        
        # Get final value for bootstrapping
        if not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, final_value = self.network(obs_tensor)
            final_value = final_value.item()
        else:
            final_value = 0.0
        
        self.episode_count += 1
        
        # Package data
        trajectory = {
            'observations': np.array(observations),
            'actions': np.array(actions),
            'log_probs': torch.stack(log_probs),
            'rewards': np.array(rewards),
            'values': torch.stack(values),
            'dones': np.array(dones),
            'final_value': final_value,
            'episode_reward': episode_reward,
            'length': step_count
        }
        
        return episode_reward, trajectory
    
    def compute_gae(self, trajectory: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE) returns and advantages.
        """
        rewards = trajectory['rewards']
        values = trajectory['values'].cpu().numpy()
        dones = trajectory['dones']
        final_value = trajectory['final_value']
        
        # Append final value for bootstrapping
        extended_values = np.append(values, final_value)
        
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        
        # Work backwards
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0.0
                next_non_terminal = 0.0
            else:
                next_value = extended_values[t + 1]
                next_non_terminal = 1.0
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - extended_values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        
        # Compute returns (advantages + values)
        returns = advantages + values
        
        return torch.FloatTensor(advantages).to(self.device), torch.FloatTensor(returns).to(self.device)
    
    def update(self, trajectory: Dict):
        """
        Perform PPO update with multiple epochs of minibatch optimization.
        """
        # Compute advantages and returns
        advantages, returns = self.compute_gae(trajectory)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Prepare data tensors
        obs_tensor = torch.FloatTensor(trajectory['observations']).to(self.device)
        actions_tensor = torch.LongTensor(trajectory['actions']).to(self.device)
        old_log_probs = trajectory['log_probs'].to(self.device)
        
        n_samples = len(trajectory['observations'])
        
        # Multiple epochs of updates
        for epoch in range(self.update_epochs):
            # Generate random minibatch indices
            indices = np.random.permutation(n_samples)
            
            # Minibatch updates
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                mb_indices = indices[start:end]
                
                # Get minibatch data
                mb_obs = obs_tensor[mb_indices]
                mb_actions = actions_tensor[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                # Forward pass
                probs, values = self.network(mb_obs)
                values = values.squeeze()
                
                # Compute new log probs
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                
                # Probability ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, mb_returns)
                
                # Total loss
                loss = actor_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        self.update_count += 1
        
        # Return metrics for logging
        with torch.no_grad():
            approx_kl = ((new_log_probs - mb_old_log_probs).abs().mean()).item()
        
        return {
            'actor_loss': actor_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'approx_kl': approx_kl
        }
    
    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'update_count': self.update_count,
            'hyperparams': {
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_epsilon': self.clip_epsilon,
                'entropy_coef': self.entropy_coef
            }
        }, path)
        print(f"Saved PPO agent to {path}")
    
    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_count = checkpoint['episode_count']
        self.total_steps = checkpoint['total_steps']
        self.update_count = checkpoint['update_count']
        print(f"Loaded PPO agent from {path}")


def evaluate_agent(agent, OBELIX, difficulty, wall_obstacles,
                   n_episodes=10, max_steps=1000, seed=99999):
    """Evaluate agent deterministically."""
    agent.network.eval()
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
                probs, _ = agent.network(obs_tensor)
            
            # Greedy action
            action_idx = int(torch.argmax(probs).item())
            action = ACTIONS[action_idx]
            
            obs, reward, done = env.step(action, render=False)
            total_reward += reward
            
            if done:
                break
        
        rewards.append(total_reward)
    
    agent.network.train()
    return np.mean(rewards)


def train(
    obelix_py_path: str,
    episodes: int = 2000,
    max_steps: int = 1000,
    difficulty: int = 0,
    wall_obstacles: bool = False,
    save_path: str = "ppo_agent.pt",
    eval_interval: int = 10,  # PPO updates less frequently, so eval more often
    hidden_dim: int = 128,
    lr: float = 3e-4,
    n_steps: int = 2048,
    update_epochs: int = 4,
    batch_size: int = 64,
    clip_epsilon: float = 0.2,
    entropy_coef: float = 0.01,
    gae_lambda: float = 0.95
):
    """Main PPO training loop."""
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OBELIX = mod.OBELIX
    
    agent = PPOAgent(
        input_dim=18,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=0.99,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_coef,
        update_epochs=update_epochs,
        batch_size=batch_size,
        n_steps=n_steps
    )
    
    print(f"\n{'='*60}")
    print(f"Training PPO Agent")
    print(f"{'='*60}")
    print(f"Total episodes: {episodes}")
    print(f"Difficulty: {difficulty}")
    print(f"N-steps per update: {n_steps}")
    print(f"Update epochs: {update_epochs}")
    print(f"Clip epsilon: {clip_epsilon}")
    print(f"GAE lambda: {gae_lambda}")
    print(f"Device: {agent.device}")
    print(f"{'='*60}\n")
    
    best_eval_reward = float('-inf')
    recent_rewards = []
    
    # PPO collects multiple episodes per update
    pbar = tqdm(total=episodes, desc="PPO Training", unit="ep")
    
    while agent.episode_count < episodes:
        # Create environment
        env = OBELIX(
            scaling_factor=5,
            arena_size=500,
            max_steps=max_steps,
            wall_obstacles=wall_obstacles,
            difficulty=difficulty,
            box_speed=2,
            seed=agent.episode_count
        )
        
        # Collect trajectory
        episode_reward, trajectory = agent.collect_trajectories(
            env, seed=agent.episode_count, max_steps=max_steps
        )
        
        recent_rewards.append(episode_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
        
        pbar.update(1)
        pbar.set_postfix({
            'reward': f'{episode_reward:.1f}',
            'avg': f'{np.mean(recent_rewards):.1f}',
            'steps': agent.total_steps,
            'updates': agent.update_count
        })
        
        # Update policy if enough steps collected
        if trajectory['length'] >= agent.n_steps or agent.episode_count % 10 == 0:
            metrics = agent.update(trajectory)
            
            # Evaluation after update
            if agent.episode_count % eval_interval == 0:
                eval_reward = evaluate_agent(
                    agent, OBELIX, difficulty, wall_obstacles,
                    n_episodes=10, max_steps=max_steps,
                    seed=10000 + agent.episode_count
                )
                
                tqdm.write(f"\n[Ep {agent.episode_count}/{episodes}] "
                          f"Train: {episode_reward:8.1f} | "
                          f"Eval: {eval_reward:8.1f} | "
                          f"Best: {max(best_eval_reward, eval_reward):8.1f} | "
                          f"Updates: {agent.update_count} | "
                          f"KL: {metrics['approx_kl']:.4f}")
                
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    agent.save(save_path.replace('.pt', '_best.pt'))
                    tqdm.write(f"*** New best model saved! ***")
        
        # Periodic checkpoint
        if agent.episode_count % 500 == 0 and agent.episode_count > 0:
            agent.save(save_path.replace('.pt', f'_ep{agent.episode_count}.pt'))
    
    pbar.close()
    
    # Final save
    print(f"\n{'='*60}")
    print("Training Complete!")
    agent.save(save_path)
    print(f"Best eval reward: {best_eval_reward:.2f}")
    print(f"Total updates: {agent.update_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obelix_py", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--save_path", type=str, default="ppo_agent.pt")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=2048,
                       help="Steps to collect before update")
    parser.add_argument("--update_epochs", type=int, default=4,
                       help="Optimization epochs per update")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--clip_epsilon", type=float, default=0.2,
                       help="PPO clipping parameter")
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                       help="GAE lambda parameter")
    
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
        n_steps=args.n_steps,
        update_epochs=args.update_epochs,
        batch_size=args.batch_size,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        gae_lambda=args.gae_lambda
    )