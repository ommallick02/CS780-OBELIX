"""TD3 Training for Discrete Actions."""

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
    
    def select_action(self, obs, noise=0.1):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.actor(obs_t)
        
        # Add noise for exploration (smooth noise for discrete via softmax temperature)
        if noise > 0:
            # Use temperature-based noise
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
            
            # Compute expected Q over all actions
            next_q1 = torch.zeros(batch_size).to(self.device)
            next_q2 = torch.zeros(batch_size).to(self.device)
            for i in range(N_ACTIONS):
                a_i = nn.functional.one_hot(torch.full((batch_size,), i, dtype=torch.long), N_ACTIONS).float().to(self.device)
                q1_i, q2_i = self.critic_target(s2, a_i)
                next_q1 += next_probs[:, i] * q1_i
                next_q2 += next_probs[:, i] * q2_i
            
            # Clipped double Q-learning
            next_q = torch.min(next_q1, next_q2)
            target_q = r + self.gamma * (1 - d) * next_q
        
        # Update critics
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
                a_i = nn.functional.one_hot(torch.full((batch_size,), i, dtype=torch.long), N_ACTIONS).float().to(self.device)
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
        
        return {'critic_loss': critic_loss.item(), 
                'actor_loss': actor_loss.item() if actor_loss else 0}

def train(obelix_py, episodes=5000, difficulty=3, wall_obstacles=False):
    OBELIX = import_obelix(obelix_py)
    agent = TD3Agent()
    
    best = float("-inf")
    recent = deque(maxlen=50)
    
    for ep in tqdm(range(episodes), desc="TD3"):
        env = OBELIX(scaling_factor = 5, difficulty=difficulty, wall_obstacles=wall_obstacles, seed=ep)
        obs = env.reset(seed=ep)
        done = False
        ep_ret = 0
        
        noise = max(0.1, 1.0 - ep / 3000)
        
        while not done:
            action = agent.select_action(obs, noise=noise)
            next_obs, reward, done = env.step(ACTIONS[action], render=False)
            ep_ret += reward
            
            agent.replay.push(obs, action, reward, next_obs, float(done))
            obs = next_obs
            
            if len(agent.replay) > 1000:
                agent.update()
        
        recent.append(ep_ret)
        if len(recent) >= 50 and (ep + 1) % 50 == 0:
            avg = sum(recent) / len(recent)
            if avg > best:
                best = avg
                torch.save(agent.actor.state_dict(), "actor_td3.pth")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", required=True)
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--difficulty", type=int, default=3)
    ap.add_argument("--wall_obstacles", action="store_true")
    args = ap.parse_args()
    train(args.obelix_py, episodes=args.episodes, difficulty=args.difficulty,
          wall_obstacles=args.wall_obstacles)