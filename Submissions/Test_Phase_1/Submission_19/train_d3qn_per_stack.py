"""Train D3QN-PER with State Stacking."""

import argparse, random
from collections import deque
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
        self.encoder = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.value_head = nn.Linear(64, 1)
        self.advantage_head = nn.Linear(64, n_actions)
    def forward(self, x):
        f = self.encoder(x)
        return self.value_head(f) + self.advantage_head(f) - self.advantage_head(f).mean(dim=-1, keepdim=True)

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity, dtype=np.float32)
        self.data = [None]*capacity
        self.write = 0
        self.n = 0
        
    def _propagate(self, idx, delta):
        parent = (idx-1)//2
        self.tree[parent] += delta
        if parent != 0: 
            self._propagate(parent, delta)
            
    def total(self): 
        return self.tree[0]
        
    def max_p(self): 
        return self.tree[self.capacity-1:self.capacity-1+self.n].max() if self.n>0 else 1.0
        
    def add(self, p, data):
        leaf = self.write + self.capacity - 1
        self._propagate(leaf, p - self.tree[leaf])
        self.tree[leaf] = p
        self.data[self.write] = data
        self.write = (self.write+1)%self.capacity
        self.n = min(self.n+1, self.capacity)
        
    def update(self, leaf, p):
        delta = p - self.tree[leaf]
        self._propagate(leaf, delta)
        self.tree[leaf] = p
        
    def get(self, v):
        idx = 0
        while idx < self.capacity-1:
            l = 2*idx+1
            r = l+1
            if v <= self.tree[l]: 
                idx = l
            else: 
                v -= self.tree[l]
                idx = r
        data_idx = idx - (self.capacity-1)
        # Safety check - return None if data is empty
        if data_idx < 0 or data_idx >= len(self.data) or self.data[data_idx] is None:
            return None, 0.0, None
        return idx, self.tree[idx], self.data[data_idx]
        
    def __len__(self): 
        return self.n

class PER:
    def __init__(self, cap=100000, alpha=0.6, beta_start=0.4, eps=1e-6):
        self.tree = SumTree(cap)
        self.alpha = alpha
        self.beta = beta_start
        self.eps = eps
        
    def add(self, t, td_error=1.0):
        p = (abs(td_error) + self.eps)**self.alpha
        self.tree.add(p, t)
        
    def sample(self, batch, beta):
        indices = []
        priorities = []
        samples = []
        segment = self.tree.total()/batch if self.tree.total() > 0 else 1e-8
        
        attempts = 0
        while len(samples) < batch and attempts < batch * 3:  # Retry limit
            attempts += 1
            a, b = segment*len(samples), segment*(len(samples)+1)
            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            
            # Skip if invalid data
            if data is None or idx is None:
                continue
                
            indices.append(idx)
            priorities.append(p)
            samples.append(data)
            
        # If we couldn't get enough samples, pad with random valid ones
        if len(samples) < batch:
            # Get all valid indices
            valid_indices = [i for i, d in enumerate(self.tree.data) if d is not None]
            if valid_indices:
                needed = batch - len(samples)
                for i in np.random.choice(valid_indices, min(needed, len(valid_indices)), replace=False):
                    samples.append(self.tree.data[i])
                    # Find tree index for this data
                    tree_idx = i + self.tree.capacity - 1
                    indices.append(tree_idx)
                    priorities.append(self.tree.tree[tree_idx])
        
        if len(samples) == 0:
            # Return None to signal caller to skip update
            return None
            
        # Safety check for weights
        probs = np.array(priorities) / (self.tree.total() + 1e-8)
        weights = (len(self.tree) * probs + 1e-10) ** (-beta)
        weights_max = weights.max()
        if weights_max > 0:
            weights /= weights_max
        else:
            weights = np.ones_like(weights)
            
        s = np.stack([x.s for x in samples]).astype(np.float32)
        a = np.array([x.a for x in samples], dtype=np.int64)
        r = np.array([x.r for x in samples], dtype=np.float32)
        s2 = np.stack([x.s2 for x in samples]).astype(np.float32)
        d = np.array([x.done for x in samples], dtype=np.float32)
        return s, a, r, s2, d, weights.astype(np.float32), indices
        
    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            if idx is not None:  # Safety check
                self.tree.update(idx, (abs(err)+self.eps)**self.alpha)
                
    def __len__(self): 
        return len(self.tree)

class Transition:
    def __init__(self, s, a, r, s2, done):
        self.s, self.a, self.r, self.s2, self.done = s, a, r, s2, done

def import_obelix(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", required=True)
    ap.add_argument("--out", default="weights.pth")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--difficulty", type=int, default=3)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    OBELIX = import_obelix(args.obelix_py)
    
    q = DuelingDQN()
    tgt = DuelingDQN()
    tgt.load_state_dict(q.state_dict())
    tgt.eval()
    opt = optim.Adam(q.parameters(), lr=1e-3)
    
    replay = PER(cap=100000)
    steps = 0
    best_avg = float("-inf")
    recent = deque(maxlen=50)
    beta = 0.4

    def eps(t): 
        return max(0.05, 1.0 - t/200000)

    for ep in tqdm(range(args.episodes), desc="D3QN-PER-Stack"):
        env = OBELIX(scaling_factor=args.scaling_factor, wall_obstacles=args.wall_obstacles, difficulty=args.difficulty, seed=args.seed+ep)
        obs = env.reset(seed=args.seed+ep)
        
        fbuf = deque(maxlen=STACK_SIZE)
        for _ in range(STACK_SIZE): 
            fbuf.append(np.zeros(18, dtype=np.float32))
        fbuf.append(obs.astype(np.float32))
        stacked = np.concatenate(list(fbuf), axis=0)
        
        ep_ret = 0.0
        
        for _ in range(2000):
            if np.random.rand() < eps(steps):
                a = np.random.randint(5)
            else:
                with torch.no_grad():
                    qs = q(torch.tensor(stacked).unsqueeze(0)).squeeze(0).numpy()
                a = int(np.argmax(qs))
            
            s2, r, done = env.step(ACTIONS[a], render=False)
            ep_ret += r
            fbuf.append(s2.astype(np.float32))
            s_next = np.concatenate(list(fbuf), axis=0)
            
            replay.add(Transition(stacked, a, float(r), s_next, done), td_error=1.0)  # Initial error=1.0
            stacked = s_next
            steps += 1
            
            if len(replay) >= 256:
                beta = min(1.0, beta + 0.001)
                result = replay.sample(256, beta)
                
                # Skip update if sampling failed
                if result is None:
                    continue
                    
                sb, ab, rb, s2b, db, w, idxs = result
                sb, ab, rb, s2b, db, w = [torch.tensor(x) for x in [sb, ab, rb, s2b, db, w]]
                
                with torch.no_grad():
                    next_a = q(s2b).argmax(dim=1)
                    next_val = tgt(s2b).gather(1, next_a.unsqueeze(1)).squeeze(1)
                    y = rb + 0.99*(1-db)*next_val
                
                pred = q(sb).gather(1, ab.unsqueeze(1)).squeeze(1)
                td = (y - pred).detach().cpu().numpy()
                replay.update_priorities(idxs, td)
                
                loss = (w * nn.functional.smooth_l1_loss(pred, y, reduction='none')).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(q.parameters(), 5.0); opt.step()
                
                if steps % 2000 == 0: 
                    tgt.load_state_dict(q.state_dict())
            
            if done: 
                break
        
        recent.append(ep_ret)
        avg = sum(recent)/len(recent)
        if avg > best_avg:
            best_avg = avg
            torch.save(q.state_dict(), args.out)
    
    print(f"Best: {best_avg:.1f}")

if __name__ == "__main__":
    main()