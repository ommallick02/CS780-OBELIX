"""Training script for REINFORCE with State Stacking."""

import argparse
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)
STACK_SIZE = 4

class PolicyNetwork(nn.Module):
    def __init__(self, in_dim=18*STACK_SIZE, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
        )
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)
    def sample(self, x):
        p = self.forward(x)
        d = torch.distributions.Categorical(p)
        a = d.sample()
        return a, d.log_prob(a)

def import_obelix(p):
    import importlib.util
    spec = importlib.util.spec_from_file_location("ob", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def train(obelix_py, episodes=5000, max_steps=2000, difficulty=3, 
          wall_obstacles=False, save_path="reinforce_stack.pt"):
    OBELIX = import_obelix(obelix_py)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    policy = PolicyNetwork().to(device)
    opt = optim.Adam(policy.parameters(), lr=3e-4)
    gamma = 0.99
    
    best = float('-inf')
    recent = deque(maxlen=100)
    
    for ep in tqdm(range(episodes)):
        env = OBELIX(scaling_factor=5, difficulty=difficulty, wall_obstacles=wall_obstacles, seed=ep)
        obs = env.reset(seed=ep)
        
        # Init frame buffer
        fbuf = deque(maxlen=STACK_SIZE)
        for _ in range(STACK_SIZE): fbuf.append(np.zeros(18, dtype=np.float32))
        fbuf.append(obs.astype(np.float32))
        stacked = np.concatenate(list(fbuf), axis=0)
        
        traj = []
        ep_ret = 0.0
        
        for _ in range(max_steps):
            with torch.no_grad():
                s = torch.FloatTensor(stacked).unsqueeze(0).to(device)
                a, lp = policy.sample(s)
                a = a.item()
            
            s2, r, done = env.step(ACTIONS[a], render=False)
            ep_ret += r
            
            fbuf.append(s2.astype(np.float32))
            s_next = np.concatenate(list(fbuf), axis=0)
            
            traj.append((stacked, a, lp.item(), r))
            stacked = s_next
            
            if done:
                break
        
        # Update
        returns = []
        G = 0
        for _, _, _, r in reversed(traj):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        loss = []
        for i, (s, a, _, _) in enumerate(traj):
            s = torch.FloatTensor(s).unsqueeze(0).to(device)
            a = torch.LongTensor([a]).to(device)
            probs = policy(s)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(a)
            loss.append(-log_prob * returns[i])
        
        opt.zero_grad()
        torch.stack(loss).sum().backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        
        recent.append(ep_ret)
        if len(recent) >= 50 and (ep + 1) % 50 == 0:
            avg = sum(recent) / len(recent)
            if avg > best:
                best = avg
                torch.save(policy.state_dict(), save_path)
    
    print(f"Best avg: {best:.1f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", required=True)
    ap.add_argument("--out", default="reinforce_stack.pt")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--difficulty", type=int, default=3)
    ap.add_argument("--wall_obstacles", action="store_true")
    args = ap.parse_args()
    train(args.obelix_py, episodes=args.episodes, difficulty=args.difficulty,
          wall_obstacles=args.wall_obstacles, save_path=args.out)