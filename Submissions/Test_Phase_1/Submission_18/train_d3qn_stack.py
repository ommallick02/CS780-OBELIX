"""Train D3QN with State Stacking."""

from __future__ import annotations
import argparse, random
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
STACK_SIZE = 4

class DuelingDQN(nn.Module):
    def __init__(self, in_dim=18*STACK_SIZE, n_actions=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.value_head = nn.Linear(64, 1)
        self.advantage_head = nn.Linear(64, n_actions)

    def forward(self, x):
        f = self.encoder(x)
        v = self.value_head(f)
        a = self.advantage_head(f)
        return v + a - a.mean(dim=-1, keepdim=True)

@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool

class Replay:
    def __init__(self, cap=100_000):
        self.buf = deque(maxlen=cap)
    def add(self, t):
        self.buf.append(t)
    def sample(self, batch):
        idx = np.random.choice(len(self.buf), batch, replace=False)
        items = [self.buf[i] for i in idx]
        s = torch.tensor(np.stack([it.s for it in items]), dtype=torch.float32)
        a = torch.tensor([it.a for it in items], dtype=torch.int64)
        r = torch.tensor([it.r for it in items], dtype=torch.float32)
        s2 = torch.tensor(np.stack([it.s2 for it in items]), dtype=torch.float32)
        d = torch.tensor([it.done for it in items], dtype=torch.float32)
        return s, a, r, s2, d
    def __len__(self): return len(self.buf)

def import_obelix(obelix_py):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", required=True)
    ap.add_argument("--out", default="weights.pth")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--difficulty", type=int, default=3)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--target_sync", type=int, default=2000)
    ap.add_argument("--eps_decay_steps", type=int, default=200000)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)
    q = DuelingDQN()
    tgt = DuelingDQN()
    tgt.load_state_dict(q.state_dict())
    tgt.eval()
    
    opt = optim.Adam(q.parameters(), lr=args.lr)
    replay = Replay()
    steps = 0
    best_avg = float("-inf")
    recent = deque(maxlen=50)

    def eps_by_step(t):
        if t >= args.eps_decay_steps: return 0.05
        return 1.0 + (t / args.eps_decay_steps) * (0.05 - 1.0)

    for ep in tqdm(range(args.episodes), desc="D3QN-Stack"):
        env = OBELIX(scaling_factor=args.scaling_factor,wall_obstacles=args.wall_obstacles, difficulty=args.difficulty, seed=args.seed+ep)
        obs = env.reset(seed=args.seed+ep)
        
        frame_buf = deque(maxlen=STACK_SIZE)
        for _ in range(STACK_SIZE): frame_buf.append(np.zeros(18, dtype=np.float32))
        frame_buf.append(obs.astype(np.float32))
        stacked = np.concatenate(list(frame_buf), axis=0)
        
        ep_ret = 0.0
        
        for _ in range(args.max_steps):
            if np.random.rand() < eps_by_step(steps):
                a = np.random.randint(5)
            else:
                with torch.no_grad():
                    qs = q(torch.tensor(stacked).unsqueeze(0)).squeeze(0).numpy()
                a = int(np.argmax(qs))
            
            s2, r, done = env.step(ACTIONS[a], render=False)
            ep_ret += r
            frame_buf.append(s2.astype(np.float32))
            stacked_next = np.concatenate(list(frame_buf), axis=0)
            
            replay.add(Transition(stacked, a, float(r), stacked_next, done))
            stacked = stacked_next
            steps += 1
            
            if len(replay) >= 256:
                sb, ab, rb, s2b, db = replay.sample(256)
                with torch.no_grad():
                    next_a = q(s2b).argmax(dim=1)
                    next_val = tgt(s2b).gather(1, next_a.unsqueeze(1)).squeeze(1)
                    y = rb + args.gamma * (1 - db) * next_val
                pred = q(sb).gather(1, ab.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, y)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(q.parameters(), 5.0); opt.step()
                
                if steps % args.target_sync == 0:
                    tgt.load_state_dict(q.state_dict())
            
            if done: break
        
        recent.append(ep_ret)
        avg = sum(recent)/len(recent)
        if avg > best_avg:
            best_avg = avg
            torch.save(q.state_dict(), args.out)
    
    print(f"Best avg50: {best_avg:.1f}")

if __name__ == "__main__":
    main()