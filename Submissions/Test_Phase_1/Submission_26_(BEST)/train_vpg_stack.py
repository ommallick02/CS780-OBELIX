"""Training VPG with State Stacking."""

import argparse
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ACTIONS = ("L45", "L22", "FW", "R22", "R45")
STACK_SIZE = 4

class Actor(nn.Module):
    def __init__(self, in_dim=18*STACK_SIZE, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 5),
        )
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)
    def sample(self, x):
        p = self.forward(x)
        d = torch.distributions.Categorical(p)
        a = d.sample()
        return a, d.log_prob(a)

class Critic(nn.Module):
    def __init__(self, in_dim=18*STACK_SIZE, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def import_obelix(p):
    import importlib.util
    spec = importlib.util.spec_from_file_location("ob", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def train(obelix_py, episodes=5000, difficulty=3, wall_obstacles=False, save_path="vpg_stack.pt"):
    OBELIX = import_obelix(obelix_py)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    actor = Actor().to(device)
    critic = Critic().to(device)
    opt_a = optim.Adam(actor.parameters(), lr=3e-4)
    opt_c = optim.Adam(critic.parameters(), lr=1e-3)
    gamma = 0.99
    
    best = float('-inf')
    recent = deque(maxlen=100)
    
    for ep in tqdm(range(episodes)):
        env = OBELIX(scaling_factor=5, difficulty=difficulty, wall_obstacles=wall_obstacles, seed=ep)
        obs = env.reset(seed=ep)
        
        fbuf = deque(maxlen=STACK_SIZE)
        for _ in range(STACK_SIZE): fbuf.append(np.zeros(18, dtype=np.float32))
        fbuf.append(obs.astype(np.float32))
        stacked = np.concatenate(list(fbuf), axis=0)
        
        traj = []
        ep_ret = 0.0
        
        for _ in range(2000):
            with torch.no_grad():
                s = torch.FloatTensor(stacked).unsqueeze(0).to(device)
                a, _ = actor.sample(s)
                a = a.item()
            
            s2, r, done = env.step(ACTIONS[a], render=False)
            ep_ret += r
            
            fbuf.append(s2.astype(np.float32))
            s_next = np.concatenate(list(fbuf), axis=0)
            
            traj.append((stacked, a, r))
            stacked = s_next
            
            if done: break
        
        # Compute returns
        returns = []
        G = 0
        for _, _, r in reversed(traj):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(device)
        
        # Update critic
        obs_list = torch.FloatTensor(np.array([t[0] for t in traj])).to(device)
        values = critic(obs_list)
        critic_loss = nn.functional.mse_loss(values, returns)
        
        opt_c.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        opt_c.step()
        
        # Update actor with advantage
        with torch.no_grad():
            advantages = returns - critic(obs_list)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        actions = torch.LongTensor([t[1] for t in traj]).to(device)
        probs = actor(obs_list)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        
        actor_loss = -(log_probs * advantages).mean()
        
        opt_a.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        opt_a.step()
        
        recent.append(ep_ret)
        if len(recent) >= 50 and (ep + 1) % 50 == 0:
            avg = sum(recent) / len(recent)
            if avg > best:
                best = avg
                torch.save(actor.state_dict(), save_path)
    
    print(f"Best avg: {best:.1f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", required=True)
    ap.add_argument("--out", default="vpg_stack.pt")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--difficulty", type=int, default=3)
    ap.add_argument("--wall_obstacles", action="store_true")
    args = ap.parse_args()
    train(args.obelix_py, episodes=args.episodes, difficulty=args.difficulty,
          wall_obstacles=args.wall_obstacles, save_path=args.out)