"""TD3 with Reward Shaping and Curriculum Learning."""

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
    """Twin critics for TD3."""
    def __init__(self, input_dim=18, hidden=128):
        super().__init__()
        # Q1
        self.net1 = nn.Sequential(
            nn.Linear(input_dim + N_ACTIONS, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        # Q2
        self.net2 = nn.Sequential(
            nn.Linear(input_dim + N_ACTIONS, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, state, action_onehot):
        x = torch.cat([state, action_onehot], dim=-1)
        return self.net1(x).squeeze(-1), self.net2(x).squeeze(-1)
    
    def q1_forward(self, state, action_onehot):
        x = torch.cat([state, action_onehot], dim=-1)
        return self.net1(x).squeeze(-1)

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
    Potential-based reward shaping for TD3.
    F(s,a,s') = γΦ(s') - Φ(s)
    """
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        
    def compute_potential(self, obs, attached):
        """Compute potential based on sensor readings."""
        # Forward sensors (indices 4-11)
        forward_far = obs[4] + obs[6] + obs[8] + obs[10]
        forward_near = obs[5] + obs[7] + obs[9] + obs[11]
        ir_sensor = obs[16]
        stuck = obs[17]
        
        if not attached:
            # Finding phase: potential from box detection
            potential = forward_far * 2.0 + forward_near * 4.0 + ir_sensor * 6.0
            # Side sensors help orienting
            potential += (obs[0] + obs[2] + obs[12] + obs[14]) * 1.0
        else:
            # Pushing phase: high potential for maintaining attachment
            potential = 15.0
            if stuck:
                potential -= 5.0  # But reduce if stuck
        
        return potential
    
    def shape_reward(self, obs, next_obs, reward, attached, done):
        """Apply potential-based shaping."""
        curr_pot = self.compute_potential(obs, attached)
        
        if done:
            next_pot = 0.0
        else:
            # Determine next attachment status
            next_attached = attached or (50 < reward < 150)
            next_pot = self.compute_potential(next_obs, next_attached)
        
        shaped = reward + self.gamma * next_pot - curr_pot
        
        # Intrinsic bonus for positive discoveries
        if reward > 0 and not attached:
            shaped += 1.0
            
        return shaped

class AdaptiveCurriculum:
    """Curriculum learning manager."""
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
        self.current_level = 0
        self.success_history = deque(maxlen=20)
        self.advance_threshold = 0.75
        self.regress_threshold = 0.20
        
    def get_config(self):
        return {
            'difficulty': self.levels[self.current_level][0],
            'wall_obstacles': self.levels[self.current_level][1],
            'box_speed': self.levels[self.current_level][2]
        }
    
    def update(self, episode_reward):
        success = episode_reward > 1500
        self.success_history.append(success)
        
        if len(self.success_history) >= 20:
            rate = sum(self.success_history) / len(self.success_history)
            if rate > self.advance_threshold and self.current_level < len(self.levels) - 1:
                self.current_level += 1
                self.success_history.clear()
                return 1
            elif rate < self.regress_threshold and self.current_level > 0:
                self.current_level -= 1
                self.success_history.clear()
                return -1
        return 0

def import_obelix(p):
    import importlib.util
    spec = importlib.util.spec_from_file_location("ob", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

class TD3Agent:
    def __init__(self, input_dim=18, hidden=128, gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        
        self.actor = Actor(input_dim, hidden).to(self.device)
        self.actor_target = Actor(input_dim, hidden).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(input_dim, hidden).to(self.device)
        self.critic_target = Critic(input_dim, hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.replay = ReplayBuffer()
        self.total_it = 0
        
        # Reward shaping
        self.shaper = PotentialBasedRewardShaper(gamma)
        
        # Curriculum
        self.curriculum = AdaptiveCurriculum()
        
        # Stats
        self.episode_count = 0
    
    def select_action(self, obs, noise=0.1, rng=None):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.actor(obs_t)
        
        # Temperature-based exploration
        if noise > 0:
            logits = torch.log(probs + 1e-8)
            logits = logits + noise * torch.randn_like(logits)
            probs = torch.softmax(logits / (1 + noise), dim=-1)
        
        return torch.argmax(probs, dim=-1).item()
    
    def update(self, batch_size=64):
        self.total_it += 1
        
        if len(self.replay) < batch_size * 10:
            return {}
        
        s, a, r, s2, d = self.replay.sample(batch_size)
        s, a, r, s2, d = [x.to(self.device) for x in [s, a, r, s2, d]]
        
        a_onehot = nn.functional.one_hot(a, N_ACTIONS).float()
        
        with torch.no_grad():
            # Target policy smoothing
            next_probs = self.actor_target(s2)
            
            # Compute expected Q over all actions (discrete case)
            next_q1 = torch.zeros(batch_size).to(self.device)
            next_q2 = torch.zeros(batch_size).to(self.device)
            for i in range(N_ACTIONS):
                a_i = nn.functional.one_hot(
                    torch.full((batch_size,), i, dtype=torch.long), N_ACTIONS
                ).float().to(self.device)
                q1_i, q2_i = self.critic_target(s2, a_i)
                next_q1 += next_probs[:, i] * q1_i
                next_q2 += next_probs[:, i] * q2_i
            
            # Clipped double Q-learning
            next_q = torch.min(next_q1, next_q2)
            target_q = r + self.gamma * (1 - d) * next_q
        
        # Update twin critics
        current_q1, current_q2 = self.critic(s, a_onehot)
        critic_loss = nn.functional.mse_loss(current_q1, target_q) + \
                      nn.functional.mse_loss(current_q2, target_q)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        actor_loss = None
        
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Actor loss: maximize Q1
            probs = self.actor(s)
            actor_loss = 0
            for i in range(N_ACTIONS):
                a_i = nn.functional.one_hot(
                    torch.full((batch_size,), i, dtype=torch.long), N_ACTIONS
                ).float().to(self.device)
                q_i = self.critic.q1_forward(s, a_i)
                actor_loss += -probs[:, i] * q_i
            
            actor_loss = actor_loss.mean()
            
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            
            # Soft update
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item() if actor_loss else 0
        }

def train(obelix_py, episodes=5000, max_steps=2000, save_prefix="td3", wall_obstacles: bool = False):
    OBELIX = import_obelix(obelix_py)
    agent = TD3Agent()
    
    if wall_obstacles:
        agent.curriculum.current_level = 1
        print("Wall obstacles enabled (Level 1)")

    rng = np.random.default_rng(42)
    
    best_reward = float("-inf")
    recent_rewards = deque(maxlen=50)
    
    print(f"Training TD3 with Reward Shaping & Curriculum")
    
    for ep in tqdm(range(episodes), desc="TD3"):
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
        
        obs = env.reset(seed=ep)
        done = False
        ep_reward = 0.0
        ep_steps = 0
        attached = False
        
        # Decaying exploration
        noise = max(0.1, 1.0 - ep / 3000)
        
        while not done and ep_steps < max_steps:
            action = agent.select_action(obs, noise=noise, rng=rng)
            next_obs, reward, done = env.step(ACTIONS[action], render=False)
            
            # Track attachment
            attached = attached or env.enable_push
            
            # Shape reward
            shaped_reward = agent.shaper.shape_reward(obs, next_obs, reward, attached, done)
            
            # Store transition
            agent.replay.push(obs, action, shaped_reward, next_obs, float(done))
            
            obs = next_obs
            ep_reward += reward
            ep_steps += 1
            
            # Update
            if len(agent.replay) > 1000:
                agent.update()
        
        agent.episode_count += 1
        recent_rewards.append(ep_reward)
        
        # Update curriculum
        change = agent.curriculum.update(ep_reward)
        
        # Logging
        if (ep + 1) % 50 == 0:
            avg_reward = np.mean(recent_rewards)
            config = agent.curriculum.get_config()
            
            status = "="
            if change == 1: status = "↑"
            elif change == -1: status = "↓"
            
            tqdm.write(
                f"{status} [Ep {ep+1}] R: {ep_reward:8.1f} | "
                f"Avg50: {avg_reward:8.1f} | Lvl: {agent.curriculum.current_level} | "
                f"D{config['difficulty']} W{config['wall_obstacles']} S{config['box_speed']}"
            )
            
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(agent.actor.state_dict(), f"{save_prefix}_actor_best.pth")
                tqdm.write(f"*** New best: {best_reward:.1f} ***")
        
        if (ep + 1) % 500 == 0:
            torch.save(agent.actor.state_dict(), f"{save_prefix}_actor_ep{ep+1}.pth")
    
    torch.save(agent.actor.state_dict(), f"{save_prefix}_actor_final.pth")
    print(f"Complete. Best: {best_reward:.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", required=True)
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--save_prefix", default="td3")
    ap.add_argument("--wall_obstacles", action="store_true",
                   help="Enable wall obstacles")
    args = ap.parse_args()
    
    train(args.obelix_py, episodes=args.episodes, max_steps=args.max_steps,
          save_prefix=args.save_prefix, wall_obstacles=args.wall_obstacles)