"""Discrete Soft Actor-Critic (SAC) with Reward Shaping and Curriculum Learning."""

import argparse
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = 5

class Actor(nn.Module):
    def __init__(self, input_dim=18, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS)
        )
    
    def forward(self, x):
        logits = self.net(x)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs
    
    def get_action(self, x, deterministic=False):
        logits = self.net(x)
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            return torch.argmax(probs, dim=-1)
        dist = torch.distributions.Categorical(probs)
        return dist.sample()

class Critic(nn.Module):
    def __init__(self, input_dim=18, hidden=128):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS)
        )
        self.net2 = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS)
        )
    
    def forward(self, x):
        return self.net1(x), self.net2(x)

class ReplayBuffer:
    def __init__(self, cap=100000):
        self.buf = deque(maxlen=cap)
    
    def push(self, *t):
        self.buf.append(t)
    
    def sample(self, batch):
        idx = np.random.choice(len(self.buf), batch, replace=False)
        batch = [self.buf[i] for i in idx]
        s, a, r, s2, d = zip(*batch)
        return (torch.FloatTensor(np.array(s)),
                torch.LongTensor(a),
                torch.FloatTensor(r),
                torch.FloatTensor(np.array(s2)),
                torch.FloatTensor(d))
    
    def __len__(self):
        return len(self.buf)

class PotentialBasedRewardShaper:
    """
    Potential-based reward shaping: F(s,a,s') = γΦ(s') - Φ(s)
    Preserves optimal policy while providing denser learning signal.
    """
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        
    def compute_potential(self, obs, attached):
        """
        Φ(s): Higher potential for states closer to goal.
        - Finding: Box visibility in forward sensors
        - Pushing: Proximity to boundary (maintained attachment)
        """
        # Forward sensors (indices 4-11): 4,6,8,10 (far), 5,7,9,11 (near)
        forward_far = obs[4] + obs[6] + obs[8] + obs[10]
        forward_near = obs[5] + obs[7] + obs[9] + obs[11]
        ir_sensor = obs[16]
        stuck = obs[17]
        
        if not attached:
            # Finding phase: reward box detection in front
            # IR (direct contact) > Near (close) > Far (detected)
            potential = forward_far * 2.0 + forward_near * 4.0 + ir_sensor * 6.0
            
            # Side sensors (0,2 left; 12,14 right) help orienting
            side_detection = obs[0] + obs[2] + obs[12] + obs[14]
            potential += side_detection * 0.5
            
        else:
            # Pushing phase: high potential when attached and not stuck
            potential = 12.0  # Base for attachment
            if stuck:
                potential -= 8.0  # Penalty for being stuck while pushing
        
        return potential
    
    def shape_reward(self, obs, next_obs, reward, attached, done):
        """Apply potential-based shaping formula."""
        curr_pot = self.compute_potential(obs, attached)
        
        if done:
            next_pot = 0.0
        else:
            # Determine if newly attached
            next_attached = attached or (50 < reward < 150)  # Attachment bonus range
            next_pot = self.compute_potential(next_obs, next_attached)
        
        # Potential-based shaping: F = γΦ(s') - Φ(s)
        shaped = reward + self.gamma * next_pot - curr_pot
        
        # Small intrinsic bonus for positive sensor hits during exploration
        if reward > 0 and not attached:
            shaped += 0.5
            
        return shaped

class AdaptiveCurriculum:
    """
    Manages progressive difficulty across 8 levels.
    Auto-adjusts based on 20-episode success rate.
    """
    def __init__(self):
        self.levels = [
            (0, False, 0),   # 0: Static, no walls (easiest)
            (0, True, 0),    # 1: Static, walls
            (2, False, 0),   # 2: Blinking, no walls
            (2, True, 0),    # 3: Blinking, walls
            (3, False, 1),   # 4: Moving slow, no walls
            (3, True, 1),    # 5: Moving slow, walls
            (3, False, 3),   # 6: Moving fast, no walls
            (3, True, 3),    # 7: Moving fast, walls (hardest)
        ]
        self.current_level = 0
        self.success_history = deque(maxlen=20)
        self.advance_threshold = 0.75  # Advance at 75% success
        self.regress_threshold = 0.20  # Regress at 20% success
        
    def get_config(self):
        """Get current difficulty configuration."""
        return {
            'difficulty': self.levels[self.current_level][0],
            'wall_obstacles': self.levels[self.current_level][1],
            'box_speed': self.levels[self.current_level][2]
        }
    
    def update(self, episode_reward):
        """
        Update curriculum based on episode performance.
        Success: Received terminal bonus (+2000) -> total > 1500
        """
        success = episode_reward > 1500
        self.success_history.append(success)
        
        if len(self.success_history) >= 20:
            success_rate = sum(self.success_history) / len(self.success_history)
            
            # Advance if performing well
            if success_rate > self.advance_threshold:
                if self.current_level < len(self.levels) - 1:
                    self.current_level += 1
                    self.success_history.clear()
                    return 1  # Advanced
            
            # Regress if performing poorly
            elif success_rate < self.regress_threshold:
                if self.current_level > 0:
                    self.current_level -= 1
                    self.success_history.clear()
                    return -1  # Regressed
        
        return 0  # No change

def import_obelix(p):
    import importlib.util
    spec = importlib.util.spec_from_file_location("ob", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

class DiscreteSACAgent:
    def __init__(self, input_dim=18, hidden=128, gamma=0.99, tau=0.005, 
                 alpha=0.2, auto_alpha=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha
        
        self.actor = Actor(input_dim, hidden).to(self.device)
        self.critic = Critic(input_dim, hidden).to(self.device)
        self.critic_target = Critic(input_dim, hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # Temperature alpha (entropy coefficient)
        if auto_alpha:
            self.target_entropy = -np.log(1.0 / N_ACTIONS)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=3e-4)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha
        
        self.replay = ReplayBuffer()
        
        # Reward shaping and curriculum
        self.shaper = PotentialBasedRewardShaper(gamma)
        self.curriculum = AdaptiveCurriculum()
        
        # Stats
        self.episode_count = 0
        self.total_steps = 0
    
    def select_action(self, obs, deterministic=False):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor.get_action(obs_t, deterministic=deterministic)
        return action.item()
    
    def update(self, batch_size=64):
        if len(self.replay) < batch_size * 10:
            return {}
        
        s, a, r, s2, d = self.replay.sample(batch_size)
        s, a, r, s2, d = [x.to(self.device) for x in [s, a, r, s2, d]]
        
        with torch.no_grad():
            next_probs, next_log_probs = self.actor(s2)
            
            # Target Q-values
            next_q1, next_q2 = self.critic_target(s2)
            next_q = torch.min(next_q1, next_q2)
            
            # Soft state value: V(s) = E[Q(s,a) - alpha*log(pi(a|s))]
            alpha_val = self.alpha if isinstance(self.alpha, (int, float)) else self.alpha.item()
            next_v = (next_probs * (next_q - alpha_val * next_log_probs)).sum(dim=-1)
            
            target_q = r + self.gamma * (1 - d) * next_v
        
        # Update critics (twin Q-networks)
        q1, q2 = self.critic(s)
        q1_a = q1.gather(1, a.unsqueeze(1)).squeeze(1)
        q2_a = q2.gather(1, a.unsqueeze(1)).squeeze(1)
        
        critic_loss = F.mse_loss(q1_a, target_q) + F.mse_loss(q2_a, target_q)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        # Update actor
        probs, log_probs = self.actor(s)
        q1, q2 = self.critic(s)
        q = torch.min(q1, q2)
        
        # Policy loss: maximize E[Q(s,a)] - alpha*H(pi)
        alpha_val = self.alpha.detach() if isinstance(self.alpha, torch.Tensor) else self.alpha
        actor_loss = (probs * (alpha_val * log_probs - q)).sum(dim=-1).mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        # Update temperature alpha (optional automatic tuning)
        alpha_info = {}
        if self.auto_alpha:
            # Update alpha to match target entropy
            alpha_loss = -(self.log_alpha * ((log_probs * probs).sum(dim=-1).detach() + self.target_entropy)).mean()
            
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp()
            alpha_info = {'alpha_loss': alpha_loss.item()}
        
        # Soft update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        result = {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha,
        }
        result.update(alpha_info)
        return result

def train(obelix_py, episodes=5000, max_steps=2000, save_prefix="sac", wall_obstacles: bool = False):
    OBELIX = import_obelix(obelix_py)
    agent = DiscreteSACAgent()
    
    if wall_obstacles:
        agent.curriculum.current_level = 1
        print("Wall obstacles enabled (Level 1)")
    rng = np.random.default_rng(42)
    
    best_reward = float("-inf")
    recent_rewards = deque(maxlen=50)
    
    print(f"Training Discrete SAC with Reward Shaping & Curriculum")
    print(f"Device: {agent.device}")
    print(f"Auto Alpha Tuning: {agent.auto_alpha}")
    print(f"Starting at curriculum level 0 (Easiest)")
    
    for ep in tqdm(range(episodes), desc="SAC"):
        # Get current curriculum configuration
        config = agent.curriculum.get_config()
        
        # Create environment with adaptive difficulty
        env = OBELIX(
            scaling_factor=5,
            arena_size=500,
            max_steps=max_steps,
            difficulty=config['difficulty'],
            wall_obstacles=config['wall_obstacles'],
            box_speed=config['box_speed'],
            seed=ep
        )
        
        obs = env.reset(seed=ep)
        done = False
        ep_reward = 0.0
        ep_steps = 0
        attached = False
        
        while not done and ep_steps < max_steps:
            # Stochastic policy for exploration (SAC's strength)
            action = agent.select_action(obs, deterministic=False)
            next_obs, reward, done = env.step(ACTIONS[action], render=False)
            
            # Track attachment status for reward shaping
            attached = attached or env.enable_push
            
            # Shape reward (use shaped for learning, original for tracking)
            shaped_reward = agent.shaper.shape_reward(obs, next_obs, reward, attached, done)
            
            # Store transition with SHAPED reward
            agent.replay.push(obs, action, shaped_reward, next_obs, float(done))
            
            obs = next_obs
            ep_reward += reward  # Track original reward
            ep_steps += 1
            agent.total_steps += 1
            
            # Update policy (SAC typically updates every step)
            if len(agent.replay) > 1000:
                agent.update()
        
        agent.episode_count += 1
        recent_rewards.append(ep_reward)
        
        # Update curriculum based on original reward
        curriculum_change = agent.curriculum.update(ep_reward)
        
        # Logging every 50 episodes
        if (ep + 1) % 50 == 0:
            avg_reward = np.mean(recent_rewards)
            current_config = agent.curriculum.get_config()
            
            status = "="
            if curriculum_change == 1:
                status = "↑ ADVANCED"
            elif curriculum_change == -1:
                status = "↓ REGRESSED"
            
            tqdm.write(
                f"{status} [Ep {ep+1}] R: {ep_reward:8.1f} | "
                f"Avg50: {avg_reward:8.1f} | Level: {agent.curriculum.current_level}/7 | "
                f"α: {agent.alpha.item() if isinstance(agent.alpha, torch.Tensor) else agent.alpha:.3f} | "
                f"D{current_config['difficulty']} W{current_config['wall_obstacles']} S{current_config['box_speed']}"
            )
            
            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(agent.actor.state_dict(), f"{save_prefix}_actor_best.pth")
                torch.save(agent.critic.state_dict(), f"{save_prefix}_critic_best.pth")
                tqdm.write(f"*** New best model saved (avg: {best_reward:.1f}) ***")
        
        # Periodic checkpoint every 500 episodes
        if (ep + 1) % 500 == 0:
            torch.save(agent.actor.state_dict(), f"{save_prefix}_actor_ep{ep+1}.pth")
    
    # Final save
    torch.save(agent.actor.state_dict(), f"{save_prefix}_actor_final.pth")
    torch.save(agent.critic.state_dict(), f"{save_prefix}_critic_final.pth")
    print(f"\nTraining complete!")
    print(f"Best average reward: {best_reward:.2f}")
    print(f"Final curriculum level: {agent.curriculum.current_level}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", required=True)
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--save_prefix", default="sac")
    ap.add_argument("--wall_obstacles", action="store_true",
                   help="Enable wall obstacles")
    args = ap.parse_args()
    
    train(args.obelix_py, episodes=args.episodes, max_steps=args.max_steps,
          save_prefix=args.save_prefix, wall_obstacles=args.wall_obstacles)