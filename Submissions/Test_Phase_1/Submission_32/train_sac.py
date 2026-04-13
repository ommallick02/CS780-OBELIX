"""Discrete Soft Actor-Critic (SAC) for OBELIX."""

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
        """
        Sample action from the policy.
        If deterministic=True, return argmax (for evaluation).
        If deterministic=False, sample from the distribution (for training).
        """
        logits = self.net(x)
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            # Greedy action selection for evaluation
            action = torch.argmax(probs, dim=-1)
        else:
            # Stochastic sampling for training exploration
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        
        return action

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
        
        # Temperature alpha
        if auto_alpha:
            self.target_entropy = -np.log(1.0 / N_ACTIONS)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=3e-4)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha
        
        self.replay = ReplayBuffer()
    
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
            
            # Target Q
            next_q1, next_q2 = self.critic_target(s2)
            next_q = torch.min(next_q1, next_q2)
            
            # Soft state value: V(s) = sum_a pi(a|s) * [Q(s,a) - alpha * log pi(a|s)]
            next_v = (next_probs * (next_q - (self.alpha * next_log_probs if isinstance(self.alpha, torch.Tensor) else self.alpha * next_log_probs))).sum(dim=-1)
            
            target_q = r + self.gamma * (1 - d) * next_v
        
        # Update critics
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
        
        # Policy loss: maximize expected Q - alpha * entropy
        actor_loss = (probs * ((self.alpha.detach() if isinstance(self.alpha, torch.Tensor) else self.alpha) * log_probs - q)).sum(dim=-1).mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        # Update alpha
        alpha_loss = torch.tensor(0.0)
        if self.auto_alpha:
            alpha = self.log_alpha.exp()
            alpha_loss = -(self.log_alpha * (log_probs * probs).sum(dim=-1).detach() + self.target_entropy).mean()
            
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp()
        
        # Soft update target
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha,
            'alpha_loss': alpha_loss.item() if isinstance(alpha_loss, torch.Tensor) else 0
        }

def train(obelix_py, episodes=5000, difficulty=3, wall_obstacles=False):
    OBELIX = import_obelix(obelix_py)
    agent = DiscreteSACAgent()
    
    best = float("-inf")
    recent = deque(maxlen=50)
    
    for ep in tqdm(range(episodes), desc="SAC"):
        env = OBELIX(scaling_factor = 5, difficulty=difficulty, wall_obstacles=wall_obstacles, seed=ep)
        obs = env.reset(seed=ep)
        done = False
        ep_ret = 0
        
        while not done:
            # Stochastic action during training
            action = agent.select_action(obs, deterministic=False)
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
                torch.save(agent.actor.state_dict(), "actor_sac.pth")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", required=True)
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--difficulty", type=int, default=3)
    ap.add_argument("--wall_obstacles", action="store_true")
    args = ap.parse_args()
    train(args.obelix_py, episodes=args.episodes, difficulty=args.difficulty,
          wall_obstacles=args.wall_obstacles)