"""DDPG Training for Discrete Actions."""

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
    
    def select_action(self, obs, noise_scale=0.1):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.actor(obs_t)
        
        # Add exploration noise (Dirichlet noise)
        if noise_scale > 0:
            noise = torch.distributions.Dirichlet(torch.ones(N_ACTIONS) * 0.5).sample()
            probs = (1 - noise_scale) * probs + noise_scale * noise.to(self.device)
            probs = probs / probs.sum()
        
        action = torch.argmax(probs, dim=-1).item()
        return action
    
    def update(self, batch_size=64):
        if len(self.replay) < batch_size * 10:
            return {}
        
        s, a, r, s2, d = self.replay.sample(batch_size)
        s, a, r, s2, d = s.to(self.device), a.to(self.device), r.to(self.device), s2.to(self.device), d.to(self.device)
        
        # One-hot actions for critic
        a_onehot = torch.nn.functional.one_hot(a, N_ACTIONS).float()
        
        # Critic update
        with torch.no_grad():
            next_probs = self.actor_target(s2)
            # For discrete DDPG, we use the expected Q-value
            next_q = torch.zeros(batch_size).to(self.device)
            for i in range(N_ACTIONS):
                a_i = torch.nn.functional.one_hot(torch.full((batch_size,), i, dtype=torch.long), N_ACTIONS).float().to(self.device)
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
            a_i = torch.nn.functional.one_hot(torch.full((batch_size,), i, dtype=torch.long), N_ACTIONS).float().to(self.device)
            q_i = self.critic(s, a_i)
            actor_loss += -probs[:, i] * q_i  # Policy gradient style
        
        actor_loss = actor_loss.mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        # Soft update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item()}

def train(obelix_py, episodes=5000, difficulty=3, wall_obstacles=False):
    OBELIX = import_obelix(obelix_py)
    agent = DDPGAgent()
    
    best = float("-inf")
    recent = deque(maxlen=50)
    
    for ep in tqdm(range(episodes), desc="DDPG"):
        env = OBELIX(scaling_factor = 5, difficulty=difficulty, wall_obstacles=wall_obstacles, seed=ep)
        obs = env.reset(seed=ep)
        done = False
        ep_ret = 0
        
        noise = max(0.1, 1.0 - ep / 2000)  # Decay noise
        
        while not done:
            action = agent.select_action(obs, noise_scale=noise)
            next_obs, reward, done = env.step(ACTIONS[action], render=False)
            ep_ret += reward
            
            agent.replay.push(obs, action, reward, next_obs, float(done))
            obs = next_obs
            
            if len(agent.replay) > 1000 and agent.replay.__len__() % 4 == 0:
                agent.update()
        
        recent.append(ep_ret)
        if len(recent) >= 50 and (ep + 1) % 50 == 0:
            avg = sum(recent) / len(recent)
            if avg > best:
                best = avg
                torch.save(agent.actor.state_dict(), "actor.pth")
                torch.save(agent.critic.state_dict(), "critic.pth")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", required=True)
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--difficulty", type=int, default=3)
    ap.add_argument("--wall_obstacles", action="store_true")
    args = ap.parse_args()
    train(args.obelix_py, episodes=args.episodes, difficulty=args.difficulty,
          wall_obstacles=args.wall_obstacles)