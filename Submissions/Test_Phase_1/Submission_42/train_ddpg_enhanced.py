"""DDPG with Reward Shaping and Curriculum Learning."""

import argparse
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = 5

class Actor(nn.Module):
    def __init__(self, input_dim=18, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, input_dim=18, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + N_ACTIONS, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, state, action_onehot):
        x = torch.cat([state, action_onehot], dim=-1)
        return self.net(x).squeeze(-1)

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
        self.prev_potential = None
        
    def compute_potential(self, obs, attached):
        """
        Φ(s): Higher potential for desirable states.
        - Finding phase (not attached): Potential based on box visibility in forward sensors
        - Pushing phase (attached): Potential based on progress toward boundary
        """
        # Forward sensors: indices 4-11 (far: even, near: odd)
        # Front-left (4,5), Front-center-left (6,7), Front-center-right (8,9), Front-right (10,11)
        forward_far = obs[4] + obs[6] + obs[8] + obs[10]
        forward_near = obs[5] + obs[7] + obs[9] + obs[11]
        ir_sensor = obs[16]  # Infrared - direct contact
        stuck = obs[17]
        
        if not attached:
            # Finding phase: reward discovering and approaching box
            # IR (5 pts) > Near (3 pts) > Far (2 pts)
            potential = forward_far * 2.0 + forward_near * 3.0 + ir_sensor * 5.0
            
            # Small bonus for side sensors to encourage turning toward box
            side_left = obs[0] + obs[2]  # Left sensors
            side_right = obs[12] + obs[14]  # Right sensors
            potential += (side_left + side_right) * 0.5
            
        else:
            # Pushing phase: reward being near boundary (indicated by stuck or proximity)
            # Since we don't have direct boundary distance, use stuck flag as proxy
            # or just maintain high potential while attached to encourage completion
            potential = 10.0  # Base potential for being attached
            
            # Bonus if facing boundary (approximated by forward movement working)
            if not stuck:
                potential += 5.0
        
        # Penalty for being stuck
        if stuck:
            potential -= 10.0
            
        return potential
    
    def shape_reward(self, obs, next_obs, reward, attached, done):
        """Apply potential-based shaping."""
        curr_potential = self.compute_potential(obs, attached)
        
        # Compute next potential (handle terminal states)
        if done:
            next_potential = 0.0
        else:
            # For next_obs, check if attachment status changed
            # In OBELIX, attachment is determined by environment state
            # We approximate: if reward includes +100 for attachment, it's newly attached
            next_attached = attached or (reward > 50 and reward < 150)  # Attachment bonus range
            next_potential = self.compute_potential(next_obs, next_attached)
        
        # Potential-based shaping formula
        shaped = reward + self.gamma * next_potential - curr_potential
        
        # Additional intrinsic motivation for exploration
        if reward > 0 and not attached:
            # Positive sensor discovery - small bonus
            shaped += 0.5
            
        return shaped, curr_potential

class AdaptiveCurriculum:
    """
    Manages progressive difficulty based on performance.
    Difficulty levels: (difficulty_flag, wall_obstacles, box_speed)
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
        self.advance_threshold = 0.75  # 75% success to advance
        self.regress_threshold = 0.20  # 20% success to regress
        
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
        Success = received the +2000 terminal bonus (approx > 1500 total)
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
            
            # Regress if performing poorly (catastrophic forgetting)
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

class DDPGAgent:
    def __init__(self, input_dim=18, hidden=128, gamma=0.99, tau=0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        
        self.actor = Actor(input_dim, hidden).to(self.device)
        self.actor_target = Actor(input_dim, hidden).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(input_dim, hidden).to(self.device)
        self.critic_target = Critic(input_dim, hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.replay = ReplayBuffer()
        
        # Reward shaping
        self.shaper = PotentialBasedRewardShaper(gamma)
        
        # Curriculum
        self.curriculum = AdaptiveCurriculum()
        
        # Tracking
        self.episode_count = 0
        self.total_steps = 0
    
    def select_action(self, obs, noise_scale=0.1, rng=None):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.actor(obs_t)
        
        # Add exploration noise (Dirichlet noise for discrete)
        if noise_scale > 0 and rng is not None:
            noise = torch.distributions.Dirichlet(torch.ones(N_ACTIONS) * 0.5).sample()
            probs = (1 - noise_scale) * probs + noise_scale * noise.to(self.device)
            probs = probs / probs.sum()
        
        action = torch.argmax(probs, dim=-1).item()
        return action
    
    def update(self, batch_size=64):
        if len(self.replay) < batch_size * 10:
            return {}
        
        s, a, r, s2, d = self.replay.sample(batch_size)
        s = s.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        s2 = s2.to(self.device)
        d = d.to(self.device)
        
        # One-hot actions for critic
        a_onehot = torch.nn.functional.one_hot(a, N_ACTIONS).float()
        
        # Critic update
        with torch.no_grad():
            next_probs = self.actor_target(s2)
            # Expected Q-value for discrete actions
            next_q = torch.zeros(batch_size).to(self.device)
            for i in range(N_ACTIONS):
                a_i = torch.nn.functional.one_hot(
                    torch.full((batch_size,), i, dtype=torch.long), N_ACTIONS
                ).float().to(self.device)
                q_i = self.critic_target(s2, a_i)
                next_q += next_probs[:, i] * q_i
            
            target_q = r + self.gamma * (1 - d) * next_q
        
        current_q = self.critic(s, a_onehot)
        critic_loss = nn.functional.mse_loss(current_q, target_q)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        # Actor update: maximize Q(s, pi(s))
        probs = self.actor(s)
        actor_loss = 0
        for i in range(N_ACTIONS):
            a_i = torch.nn.functional.one_hot(
                torch.full((batch_size,), i, dtype=torch.long), N_ACTIONS
            ).float().to(self.device)
            q_i = self.critic(s, a_i)
            actor_loss += -probs[:, i] * q_i
        
        actor_loss = actor_loss.mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        # Soft update targets
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'avg_q': current_q.mean().item()
        }

def train(obelix_py, episodes=5000, max_steps=2000, save_prefix="ddpg", wall_obstacles: bool = False):
    OBELIX = import_obelix(obelix_py)
    agent = DDPGAgent()
    
    # Force wall level if requested
    if wall_obstacles:
        agent.curriculum.current_level = 1  # Static + Walls
        print("Wall obstacles enabled (Level 1)")

    rng = np.random.default_rng(42)
    
    best_reward = float("-inf")
    recent_rewards = deque(maxlen=50)
    
    print(f"Training DDPG with Reward Shaping & Curriculum")
    print(f"Starting at curriculum level 0 (Easiest)")
    
    for ep in tqdm(range(episodes), desc="DDPG"):
        # Get current curriculum config
        config = agent.curriculum.get_config()
        
        # Create environment with current difficulty
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
        
        # Decaying exploration noise
        noise = max(0.1, 1.0 - ep / 2000)
        
        while not done and ep_steps < max_steps:
            action = agent.select_action(obs, noise_scale=noise, rng=rng)
            next_obs, reward, done = env.step(ACTIONS[action], render=False)
            
            # Track attachment status for shaping
            newly_attached = (not attached and reward > 50 and reward < 150)
            attached = attached or newly_attached or env.enable_push
            
            # Shape the reward
            shaped_reward, potential = agent.shaper.shape_reward(
                obs, next_obs, reward, attached, done
            )
            
            # Store transition with SHAPED reward
            agent.replay.push(obs, action, shaped_reward, next_obs, float(done))
            
            obs = next_obs
            ep_reward += reward  # Track original reward for curriculum
            ep_steps += 1
            agent.total_steps += 1
            
            # Update policy
            if len(agent.replay) > 1000 and agent.total_steps % 4 == 0:
                metrics = agent.update()
        
        agent.episode_count += 1
        recent_rewards.append(ep_reward)
        
        # Update curriculum based on original reward
        curriculum_change = agent.curriculum.update(ep_reward)
        
        # Logging
        if (ep + 1) % 50 == 0:
            avg_reward = np.mean(recent_rewards)
            config = agent.curriculum.get_config()
            
            status = "="
            if curriculum_change == 1:
                status = "↑"
            elif curriculum_change == -1:
                status = "↓"
            
            tqdm.write(
                f"{status} [Ep {ep+1}] Reward: {ep_reward:8.1f} | "
                f"Avg50: {avg_reward:8.1f} | Level: {agent.curriculum.current_level} | "
                f"Config: {config}"
            )
            
            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(agent.actor.state_dict(), f"{save_prefix}_actor_best.pth")
                torch.save(agent.critic.state_dict(), f"{save_prefix}_critic_best.pth")
                tqdm.write(f"*** New best model saved (avg: {best_reward:.1f}) ***")
        
        # Periodic checkpoint
        if (ep + 1) % 500 == 0:
            torch.save(agent.actor.state_dict(), f"{save_prefix}_actor_ep{ep+1}.pth")
    
    # Final save
    torch.save(agent.actor.state_dict(), f"{save_prefix}_actor_final.pth")
    torch.save(agent.critic.state_dict(), f"{save_prefix}_critic_final.pth")
    print(f"Training complete. Best avg reward: {best_reward:.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", required=True)
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--save_prefix", default="ddpg")
    ap.add_argument("--wall_obstacles", action="store_true",
                   help="Enable wall obstacles in environment")
    args = ap.parse_args()
    
    train(args.obelix_py, episodes=args.episodes, max_steps=args.max_steps,
          save_prefix=args.save_prefix, wall_obstacles=args.wall_obstacles)