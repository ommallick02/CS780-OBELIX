"""
Feudal PPO with Hierarchical Manager-Worker Architecture.
Manager sets directional goals every C steps; Worker executes primitive actions.
Includes Reward Shaping and Curriculum Learning.

python training_feudal_ppo.py --obelix_py ./obelix.py --manager_interval 10 --episodes 10000
"""

import argparse
import os
from typing import List, Tuple, Sequence, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import deque

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = len(ACTIONS)
GOAL_DIM = 16  # Dimension of directional goal vector
MANAGER_INTERVAL = 10  # Manager acts every 10 steps


class ManagerNetwork(nn.Module):
    """
    High-level Manager: Observes state every C steps and produces goal vectors.
    Uses LSTM for temporal memory (important for tracking box movement).
    """
    def __init__(self, input_dim=18, hidden=128, goal_dim=GOAL_DIM):
        super().__init__()
        self.goal_dim = goal_dim
        
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
        self.goal_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, goal_dim),
            nn.Tanh()  # Bound goals to [-1, 1]
        )
        self.value_head = nn.Linear(hidden, 1)
        
    def forward(self, x, hidden_state=None):
        """
        x: (batch, seq_len, input_dim) or (batch, input_dim) -> treated as seq=1
        Returns: goal, value, hidden_state
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add seq dimension
            
        if hidden_state is None:
            h0 = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(x.device)
            c0 = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(x.device)
            hidden_state = (h0, c0)
            
        lstm_out, new_hidden = self.lstm(x, hidden_state)
        features = lstm_out[:, -1, :]  # Last timestep
        
        goal = self.goal_head(features)
        value = self.value_head(features).squeeze(-1)
        
        return goal, value, new_hidden


class WorkerNetwork(nn.Module):
    """
    Low-level Worker: Executes primitive actions conditioned on current goal.
    Observes every step: [observation, goal]
    """
    def __init__(self, input_dim=18, goal_dim=GOAL_DIM, hidden=128):
        super().__init__()
        
        # Encode observation + goal
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + goal_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        
        # Policy head
        self.actor = nn.Sequential(
            nn.Linear(hidden, N_ACTIONS),
            nn.Softmax(dim=-1)
        )
        
        # Value head (conditioned on goal)
        self.critic = nn.Linear(hidden, 1)
        
    def forward(self, obs, goal):
        """
        obs: (batch, input_dim)
        goal: (batch, goal_dim)
        Returns: probs, value
        """
        x = torch.cat([obs, goal], dim=-1)
        features = self.encoder(x)
        probs = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return probs, value
    
    def get_action(self, obs, goal, device):
        """Sample action and return (action_idx, log_prob, value)."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        goal_t = torch.FloatTensor(goal).unsqueeze(0).to(device) if isinstance(goal, np.ndarray) else goal
        
        with torch.no_grad():
            probs, value = self.forward(obs_t, goal_t)
        
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action_idx))
        
        return action_idx, log_prob, value


class PotentialBasedRewardShaper:
    """
    Potential-based reward shaping for OBELIX.
    F(s,a,s') = γΦ(s') - Φ(s)
    """
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        
    def compute_potential(self, obs, attached):
        """Compute potential based on sensor readings and attachment."""
        # Forward sensors: indices 4-11 (even=far, odd=near)
        forward_far = obs[4] + obs[6] + obs[8] + obs[10]
        forward_near = obs[5] + obs[7] + obs[9] + obs[11]
        ir = obs[16]
        stuck = obs[17]
        
        if not attached:
            # Finding: Higher potential when box is in front
            potential = forward_far * 2.0 + forward_near * 4.0 + ir * 6.0
            # Side sensors for orientation
            potential += (obs[0] + obs[2] + obs[12] + obs[14]) * 0.5
        else:
            # Pushing: High potential for maintaining attachment and not being stuck
            potential = 15.0
            if stuck:
                potential -= 10.0
        
        return potential
    
    def shape_reward(self, obs, next_obs, reward, attached, done):
        """Apply potential-based shaping."""
        curr_pot = self.compute_potential(obs, attached)
        
        if done:
            next_pot = 0.0
        else:
            next_attached = attached or (50 < reward < 150)
            next_pot = self.compute_potential(next_obs, next_attached)
        
        # Potential-based shaping formula
        shaped = reward + self.gamma * next_pot - curr_pot
        
        # Intrinsic bonus for exploration
        if reward > 0 and not attached:
            shaped += 0.5
            
        return shaped


class AdaptiveCurriculum:
    """Manages progressive difficulty based on performance."""
    def __init__(self):
        self.levels = [
            (0, False, 0),   # Static, no walls
            (0, True, 0),    # Static, walls
            (2, False, 0),   # Blinking, no walls
            (2, True, 0),    # Blinking, walls
            (3, False, 1),   # Moving slow, no walls
            (3, True, 1),    # Moving slow, walls
            (3, False, 3),   # Moving fast, no walls
            (3, True, 3),    # Moving fast, walls
        ]
        self.current = 0
        self.success_history = deque(maxlen=20)
        self.advance_threshold = 0.75
        self.regress_threshold = 0.20
        
    def get_config(self):
        return {
            'difficulty': self.levels[self.current][0],
            'wall_obstacles': self.levels[self.current][1],
            'box_speed': self.levels[self.current][2]
        }
    
    def update(self, episode_reward):
        success = episode_reward > 1500
        self.success_history.append(success)
        
        if len(self.success_history) >= 20:
            rate = sum(self.success_history) / len(self.success_history)
            if rate > self.advance_threshold and self.current < len(self.levels) - 1:
                self.current += 1
                self.success_history.clear()
                return 1
            elif rate < self.regress_threshold and self.current > 0:
                self.current -= 1
                self.success_history.clear()
                return -1
        return 0


class FeudalPPOAgent:
    """
    Feudal PPO: Hierarchical RL with Manager-Worker architecture.
    Manager sets goals every C steps; Worker executes actions conditioned on goals.
    """
    
    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 128,
        goal_dim: int = GOAL_DIM,
        manager_interval: int = 10,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        batch_size: int = 64
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.goal_dim = goal_dim
        self.manager_interval = manager_interval
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
        # Networks
        self.manager = ManagerNetwork(input_dim, hidden_dim, goal_dim).to(self.device)
        self.worker = WorkerNetwork(input_dim, goal_dim, hidden_dim).to(self.device)
        
        # Optimizers
        self.manager_opt = optim.Adam(self.manager.parameters(), lr=lr)
        self.worker_opt = optim.Adam(self.worker.parameters(), lr=lr)
        
        # Shaping and curriculum
        self.shaper = PotentialBasedRewardShaper(gamma)
        self.curriculum = AdaptiveCurriculum()
        
        # Stats
        self.episode_count = 0
        self.total_steps = 0
        
    def collect_trajectories(self, env, seed: int, max_steps: int):
        """
        Collect trajectories with Manager-Worker interactions.
        Manager updates every manager_interval steps.
        """
        rng = np.random.default_rng(seed)
        obs = env.reset(seed=seed)
        
        # Storage for Worker (every step)
        w_observations = []
        w_actions = []
        w_log_probs = []
        w_rewards = []
        w_values = []
        w_dones = []
        w_goals = []  # Which goal was active
        
        # Storage for Manager (every manager_interval steps)
        m_observations = []
        m_goals = []  # Goals produced
        m_rewards = []  # Accumulated rewards over interval
        m_values = []
        m_dones = []
        
        episode_reward = 0.0
        step_count = 0
        attached = False
        
        # Manager state
        manager_hidden = None
        current_goal = np.zeros(self.goal_dim, dtype=np.float32)
        last_manager_value = 0.0
        manager_accumulated_reward = 0.0
        steps_since_manager = 0
        
        while step_count < max_steps:
            # Manager decision every C steps
            if steps_since_manager % self.manager_interval == 0:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    new_goal, manager_value, manager_hidden = self.manager(
                        obs_t, manager_hidden
                    )
                
                current_goal = new_goal.squeeze().cpu().numpy()
                last_manager_value = manager_value.item()
                
                # Store manager transition if not first step
                if steps_since_manager > 0:
                    m_observations.append(obs)  # Previous obs where goal was set
                    m_goals.append(current_goal)
                    m_rewards.append(manager_accumulated_reward)
                    m_values.append(last_manager_value)
                    m_dones.append(False)  # Will be updated later
                
                manager_accumulated_reward = 0.0
            
            # Worker acts every step
            action_idx, log_prob, worker_value = self.worker.get_action(
                obs, current_goal, self.device
            )
            action = ACTIONS[action_idx]
            
            # Step environment
            next_obs, reward, done = env.step(action, render=False)
            
            # Track attachment
            attached = attached or env.enable_push
            
            # Shape reward for worker
            shaped_reward = self.shaper.shape_reward(obs, next_obs, reward, attached, done)
            
            # Store worker experience
            w_observations.append(obs.copy())
            w_actions.append(action_idx)
            w_log_probs.append(log_prob)
            w_rewards.append(shaped_reward)
            w_values.append(worker_value)
            w_dones.append(done)
            w_goals.append(current_goal.copy())
            
            # Accumulate for manager
            manager_accumulated_reward += reward  # Use original reward for manager
            
            episode_reward += reward
            obs = next_obs
            step_count += 1
            steps_since_manager += 1
            self.total_steps += 1
            
            if done:
                break
        
        # Final manager transition
        if steps_since_manager > 0:
            m_observations.append(obs)
            m_goals.append(current_goal)
            m_rewards.append(manager_accumulated_reward)
            m_values.append(0.0)  # Terminal
            m_dones.append(True)
        
        self.episode_count += 1
        
        return episode_reward, {
            'worker': {
                'obs': np.array(w_observations),
                'actions': np.array(w_actions),
                'log_probs': torch.stack(w_log_probs).flatten(),  # <--- Flatten to 1D
                'rewards': np.array(w_rewards),
                'values': torch.stack(w_values).flatten(),        # <--- Flatten to 1D
                'dones': np.array(w_dones),
                'goals': np.array(w_goals)
            },
            'manager': {
                'obs': np.array(m_observations),
                'goals': np.array(m_goals),
                'rewards': np.array(m_rewards),
                'values': torch.FloatTensor(m_values),
                'dones': np.array(m_dones)
            },
            'length': step_count
        }
    
    def compute_gae(self, rewards, values, dones, final_value=0.0):
        """Compute Generalized Advantage Estimation."""
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        
        # Extend values with final value for bootstrapping
        extended_values = np.append(values, final_value)
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_val = 0.0
                non_terminal = 0.0
            else:
                next_val = extended_values[t + 1]
                non_terminal = 1.0
            
            delta = rewards[t] + self.gamma * next_val * non_terminal - extended_values[t]
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        return torch.FloatTensor(advantages).to(self.device), torch.FloatTensor(returns).to(self.device)
    
    def update_worker(self, w_data):
        """Update Worker network using PPO."""
        obs = torch.FloatTensor(w_data['obs']).to(self.device)
        actions = torch.LongTensor(w_data['actions']).to(self.device)
        old_log_probs = w_data['log_probs'].to(self.device)
        goals = torch.FloatTensor(w_data['goals']).to(self.device)
        
        # Compute GAE
        advantages, returns = self.compute_gae(
            w_data['rewards'],
            w_data['values'].cpu().numpy(),
            w_data['dones']
        )
        
        # Normalize advantages (SAFE VERSION)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = torch.zeros_like(advantages)
        
        n_samples = len(obs)
        
        for epoch in range(self.update_epochs):
            indices = np.random.permutation(n_samples)
            
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                mb = indices[start:end]
                
                # Forward pass
                probs, values = self.worker(obs[mb], goals[mb])
                values = values.view(-1)  # <-- USE view(-1) INSTEAD OF squeeze()
                
                # Log probs
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(actions[mb])
                entropy = dist.entropy().mean()
                
                # PPO loss
                ratio = torch.exp(new_log_probs - old_log_probs[mb])
                surr1 = ratio * advantages[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[mb]
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, returns[mb])
                
                # Total loss
                loss = actor_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                self.worker_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.worker.parameters(), self.max_grad_norm)
                self.worker_opt.step()
    
    def update_manager(self, m_data):
        """Update Manager network using PPO."""
        if len(m_data['obs']) == 0:
            return
        
        obs = torch.FloatTensor(m_data['obs']).to(self.device)
        # Manager's "action" is the goal it produced
        goals = torch.FloatTensor(m_data['goals']).to(self.device)
        old_log_probs = torch.zeros(len(obs)).to(self.device)  # Placeholder, manager is deterministic-ish
        
        # Compute GAE for manager
        advantages, returns = self.compute_gae(
            m_data['rewards'],
            m_data['values'].cpu().numpy(),
            m_data['dones']
        )
        
        # Normalize (SAFE VERSION)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = torch.zeros_like(advantages) # Mean is 0 when N=1
        
        n_samples = len(obs)
        
        for epoch in range(self.update_epochs):
            indices = np.random.permutation(n_samples)
            
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                mb = indices[start:end]
                
                # Manager forward
                pred_goals, values, _ = self.manager(obs[mb].unsqueeze(1))
                values = values.view(-1)  # <-- USE view(-1) INSTEAD OF squeeze()
                
                # Manager loss: MSE on goal prediction (simplified) + value loss
                # Actually for continuous goals, we can use MSE loss on goals as "policy loss"
                # But since goals are continuous outputs, we use deterministic policy gradient style
                goal_loss = nn.functional.mse_loss(pred_goals, goals[mb])
                value_loss = nn.functional.mse_loss(values, returns[mb])
                
                # Advantage-weighted goal loss
                weighted_goal_loss = (goal_loss * advantages[mb].abs()).mean()
                
                loss = weighted_goal_loss + self.value_loss_coef * value_loss
                
                self.manager_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.manager.parameters(), self.max_grad_norm)
                self.manager_opt.step()
    
    def save(self, path):
        """Save both networks."""
        torch.save({
            'manager_state': self.manager.state_dict(),
            'worker_state': self.worker.state_dict(),
            'manager_opt': self.manager_opt.state_dict(),
            'worker_opt': self.worker_opt.state_dict(),
            'curriculum_level': self.curriculum.current
        }, path)
    
    def load(self, path):
        """Load both networks."""
        ckpt = torch.load(path, map_location=self.device)
        self.manager.load_state_dict(ckpt['manager_state'])
        self.worker.load_state_dict(ckpt['worker_state'])
        self.manager_opt.load_state_dict(ckpt['manager_opt'])
        self.worker_opt.load_state_dict(ckpt['worker_opt'])


def train(obelix_py_path: str, episodes: int = 5000, max_steps: int = 2000, 
          save_path: str = "feudal_ppo.pt", wall_obstacles: bool = False, manager_interval: int = 10):
    """Main training loop for Feudal PPO."""
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OBELIX = mod.OBELIX
    
    agent = FeudalPPOAgent()
    
    if wall_obstacles:
        agent.curriculum.current = 1  # Note: Feudal uses .current not .current_level
        print("Wall obstacles enabled (Level 1)")
    rng = np.random.default_rng(42)
    
    best_reward = float('-inf')
    recent_rewards = deque(maxlen=50)
    
    print(f"Training Feudal PPO")
    print(f"Manager interval: {agent.manager_interval}")
    print(f"Goal dimension: {agent.goal_dim}")
    print(f"Device: {agent.device}")
    
    for ep in tqdm(range(episodes)):
        # Curriculum config
        config = agent.curriculum.get_config()
        
        env = OBELIX(
            scaling_factor=5,
            arena_size=500,
            max_steps=max_steps,
            difficulty=config['difficulty'],
            wall_obstacles=config['wall_obstacles'],
            box_speed=config['box_speed'],
            seed=ep
        )
        
        # Collect
        ep_reward, trajectory = agent.collect_trajectories(env, ep, max_steps)
        
        # Update both levels
        agent.update_worker(trajectory['worker'])
        agent.update_manager(trajectory['manager'])
        
        recent_rewards.append(ep_reward)
        
        # Curriculum update
        change = agent.curriculum.update(ep_reward)
        if change == 1:
            tqdm.write(f"↑ Advanced to level {agent.curriculum.current} at ep {ep}")
        elif change == -1:
            tqdm.write(f"↓ Regressed to level {agent.curriculum.current} at ep {ep}")
        
        # Logging
        if (ep + 1) % 50 == 0:
            avg = np.mean(recent_rewards)
            config = agent.curriculum.get_config()
            tqdm.write(
                f"[Ep {ep+1}] R: {ep_reward:8.1f} | Avg: {avg:8.1f} | "
                f"Level: {agent.curriculum.current} | "
                f"D{config['difficulty']} W{config['wall_obstacles']}"
            )
            
            if avg > best_reward:
                best_reward = avg
                agent.save(save_path.replace('.pt', '_best.pt'))
                tqdm.write(f"*** New best: {best_reward:.1f} ***")
        
        if (ep + 1) % 500 == 0:
            agent.save(save_path.replace('.pt', f'_ep{ep+1}.pt'))
    
    agent.save(save_path)
    print(f"Done. Best: {best_reward:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obelix_py", required=True)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--save_path", default="feudal_ppo.pt")
    parser.add_argument("--manager_interval", type=int, default=10)
    parser.add_argument("--wall_obstacles", action="store_true",
                       help="Enable wall obstacles")
    args = parser.parse_args()
    
    train(args.obelix_py, 
          episodes=args.episodes, 
          max_steps=args.max_steps,
          save_path=args.save_path, 
          wall_obstacles=args.wall_obstacles,
          manager_interval=args.manager_interval)
