"""
Training script for A2C (Advantage Actor-Critic).
Uses n-step returns for better credit assignment and updates during episodes.

python training_a2c.py --obelix_py ../../../obelix.py --episodes 5000 --difficulty 3 --wall_obstacles --entropy_coef 0.05 --lr 1e-4
"""

import argparse
import os
from typing import List, Tuple, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = len(ACTIONS)


class ActorCriticNetwork(nn.Module):
    """
    Shared network for both actor (policy) and critic (value).
    More efficient than separate networks.
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
            nn.Linear(hidden_dim, N_ACTIONS)
            # No Softmax here - we'll use log_softmax for stability
        )
        
        # Critic head: state value
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (action_logits, state_value)."""
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value
    
    def get_action_and_value(self, obs: np.ndarray, device: str, deterministic: bool = False):
        """
        Sample action and return (action_idx, log_prob, entropy, value).
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, value = self.forward(obs_tensor)
            
            # Use log_softmax for numerical stability
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)
            
            # Check for NaN
            if torch.isnan(probs).any():
                # Fallback to uniform if NaN detected
                probs = torch.ones_like(probs) / N_ACTIONS
                log_probs = torch.log(probs)
            
            dist = torch.distributions.Categorical(probs)
            
            if deterministic:
                action_idx = torch.argmax(probs, dim=-1).item()
            else:
                action_idx = dist.sample().item()
            
            action_log_prob = log_probs[0, action_idx]
            entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        return action_idx, action_log_prob, entropy, value.squeeze()


class A2CAgent:
    """
    A2C Agent with n-step bootstrapping.
    Updates policy and value function during episodes.
    """
    
    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        
        # Network
        self.network = ActorCriticNetwork(input_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        
        self.episode_count = 0
        self.total_steps = 0
    
    def collect_trajectory(self, env, seed: int, max_steps: int) -> Tuple[float, List]:
        """
        Collect trajectory with n-step bootstrapping.
        Returns: (total_reward, trajectory)
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
        
        for step in range(max_steps):
            # Select action with stability checks
            try:
                action_idx, log_prob, entropy, value = self.network.get_action_and_value(
                    obs, self.device, deterministic=False
                )
            except Exception as e:
                # Fallback to random action if network fails
                print(f"Warning: Network error, using random action: {e}")
                action_idx = int(rng.integers(N_ACTIONS))
                log_prob = torch.tensor(-np.log(N_ACTIONS))
                value = torch.tensor(0.0)
            
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
            try:
                _, _, _, final_value = self.network.get_action_and_value(obs, self.device)
            except:
                final_value = torch.tensor(0.0)
        else:
            final_value = torch.tensor(0.0)
        
        self.episode_count += 1
        
        # Package data - pre-convert to numpy array to avoid warning
        trajectory = {
            'observations': np.array(observations, dtype=np.float32),
            'actions': np.array(actions, dtype=np.int64),
            'log_probs': torch.stack(log_probs) if log_probs else torch.tensor([]),
            'rewards': np.array(rewards, dtype=np.float32),
            'values': torch.stack(values) if values else torch.tensor([]),
            'dones': np.array(dones, dtype=np.float32),
            'final_value': final_value,
            'episode_reward': episode_reward,
            'length': step_count
        }
        
        return episode_reward, trajectory
    
    def compute_advantages(self, trajectory: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns.
        """
        rewards = trajectory['rewards']
        values = torch.stack([trajectory['values']] if isinstance(trajectory['values'], torch.Tensor) else [v for v in trajectory['values']])
        values = values.cpu().numpy()
        dones = trajectory['dones']
        final_value = trajectory['final_value'].item() if isinstance(trajectory['final_value'], torch.Tensor) else trajectory['final_value']
        
        n = len(rewards)
        if n == 0:
            return torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
        
        # Append final value
        extended_values = np.append(values, final_value)
        
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        
        for t in reversed(range(n)):
            if dones[t]:
                next_value = 0.0
                next_non_terminal = 0.0
            else:
                next_value = extended_values[t + 1]
                next_non_terminal = 1.0
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - extended_values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        
        return torch.FloatTensor(advantages).to(self.device), torch.FloatTensor(returns).to(self.device)
    
    def update(self, trajectory: dict):
        """
        Update actor and critic using collected trajectory.
        """
        n_samples = trajectory['length']
        if n_samples == 0:
            return {}
        
        # Compute advantages
        advantages, returns = self.compute_advantages(trajectory)
        
        # Skip update if empty
        if len(advantages) == 0:
            return {}
        
        # Normalize advantages (handle small batches)
        if len(advantages) > 1:
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std
        
        # Prepare data
        obs_tensor = torch.FloatTensor(trajectory['observations']).to(self.device)
        actions_tensor = torch.LongTensor(trajectory['actions']).to(self.device)
        old_log_probs = trajectory['log_probs'].to(self.device)
        
        # Forward pass
        logits, values = self.network(obs_tensor)
        values = values.squeeze(-1)  # Ensure correct shape
        
        # Compute new log probs with numerical stability
        log_probs_all = torch.log_softmax(logits, dim=-1)
        new_log_probs = log_probs_all.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # Entropy
        probs = torch.exp(log_probs_all)
        entropy = -(probs * log_probs_all).sum(dim=-1).mean()
        
        # Policy loss (A2C: no clipping unlike PPO)
        policy_loss = -(new_log_probs * advantages).mean()
        
        # Value loss with shape checking
        if values.dim() == 0:
            values = values.unsqueeze(0)
        if returns.dim() == 0:
            returns = returns.unsqueeze(0)
        
        value_loss = nn.functional.mse_loss(values, returns)
        
        # Total loss
        loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss
        
        # Check for NaN before backward
        if torch.isnan(loss):
            print(f"Warning: NaN loss detected, skipping update")
            print(f"  policy_loss: {policy_loss}, value_loss: {value_loss}, entropy: {entropy}")
            return {
                'policy_loss': float('nan'),
                'value_loss': float('nan'),
                'entropy': float('nan')
            }
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }
    
    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'hyperparams': {
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'entropy_coef': self.entropy_coef
            }
        }, path)
        print(f"Saved A2C agent to {path}")
    
    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_count = checkpoint['episode_count']
        self.total_steps = checkpoint['total_steps']


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
            try:
                action_idx, _, _, _ = agent.network.get_action_and_value(
                    obs, agent.device, deterministic=True
                )
            except:
                # Fallback
                action_idx = 2  # FW
            
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
    save_path: str = "a2c_agent.pt",
    eval_interval: int = 50,
    hidden_dim: int = 128,
    lr: float = 3e-4,
    entropy_coef: float = 0.01
):
    """Main A2C training loop."""
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OBELIX = mod.OBELIX
    
    agent = A2CAgent(
        input_dim=18,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=entropy_coef
    )
    
    print(f"\n{'='*60}")
    print(f"Training A2C Agent")
    print(f"{'='*60}")
    print(f"Episodes: {episodes}")
    print(f"Difficulty: {difficulty}")
    print(f"Wall obstacles: {wall_obstacles}")
    print(f"Entropy coef: {entropy_coef}")
    print(f"Device: {agent.device}")
    print(f"{'='*60}\n")
    
    best_eval_reward = float('-inf')
    recent_rewards = []
    
    with tqdm(total=episodes, desc="Training", unit="ep") as pbar:
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
            
            # Collect and learn
            episode_reward, trajectory = agent.collect_trajectory(env, seed=episode, max_steps=max_steps)
            
            # Update if we have data
            if trajectory['length'] > 0:
                metrics = agent.update(trajectory)
            
            recent_rewards.append(episode_reward)
            if len(recent_rewards) > 100:
                recent_rewards.pop(0)
            
            pbar.update(1)
            pbar.set_postfix({
                'reward': f'{episode_reward:.1f}',
                'avg100': f'{np.mean(recent_rewards):.1f}' if recent_rewards else '0.0',
                'steps': agent.total_steps
            })
            
            if (episode + 1) % eval_interval == 0:
                pbar.set_description("Evaluating...")
                eval_reward = evaluate_agent(agent, OBELIX, difficulty, wall_obstacles,
                                           n_episodes=10, max_steps=max_steps,
                                           seed=10000 + episode)
                pbar.set_description("Training")
                
                tqdm.write(f"\n[Ep {episode+1}/{episodes}] "
                          f"Train: {episode_reward:8.1f} | "
                          f"Eval: {eval_reward:8.1f} | "
                          f"Best: {max(best_eval_reward, eval_reward):8.1f}")
                
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    agent.save(save_path.replace('.pt', '_best.pt'))
                    tqdm.write(f"*** New best model saved! ***")
            
            if (episode + 1) % 500 == 0:
                agent.save(save_path.replace('.pt', f'_ep{episode+1}.pt'))
                tqdm.write(f"Checkpoint saved")
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    agent.save(save_path)
    print(f"Best eval reward: {best_eval_reward:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obelix_py", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--save_path", type=str, default="a2c_agent.pt")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    
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
        entropy_coef=args.entropy_coef
    )