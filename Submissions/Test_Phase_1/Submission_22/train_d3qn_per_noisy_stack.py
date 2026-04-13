"""Train D3QN-PER-Noisy with State Stacking."""

import argparse
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ACTIONS = ("L45", "L22", "FW", "R22", "R45")
STACK_SIZE = 4

class NoisyLinear(nn.Module):
    def __init__(self, in_f, out_f, sigma=0.017):
        super().__init__()
        # FIX: Store dimensions as instance attributes
        self.in_f = in_f
        self.out_f = out_f
        
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_f, in_f))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_f, in_f))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_f))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_f))
        self.register_buffer('w_eps', torch.FloatTensor(out_f, in_f))
        self.register_buffer('b_eps', torch.FloatTensor(out_f))
        self.sigma_init = sigma
        self.reset_params()
        self.reset_noise()

    def reset_params(self):
        mu_r = 1/np.sqrt(self.in_f)
        # FIX: Wrap in no_grad to prevent in-place modification errors
        with torch.no_grad():
            self.weight_mu.uniform_(-mu_r, mu_r)
            self.weight_sigma.fill_(self.sigma_init/np.sqrt(self.in_f))
            self.bias_mu.uniform_(-mu_r, mu_r)
            self.bias_sigma.fill_(self.sigma_init/np.sqrt(self.out_f))

    def reset_noise(self):
        eps_in = torch.randn(self.in_f).sign().mul_(torch.randn(self.in_f).abs().sqrt())
        eps_out = torch.randn(self.out_f).sign().mul_(torch.randn(self.out_f).abs().sqrt())
        self.w_eps.copy_(eps_out.outer(eps_in)); self.b_eps.copy_(eps_out)

    def forward(self, x):
        if self.training: 
            w = self.weight_mu + self.weight_sigma*self.w_eps
            b = self.bias_mu + self.bias_sigma*self.b_eps
        else: 
            w, b = self.weight_mu, self.bias_mu
        return nn.functional.linear(x, w, b)

class NoisyDuelingDQN(nn.Module):
    def __init__(self, in_dim=18*STACK_SIZE, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(NoisyLinear(in_dim, hidden), nn.ReLU())
        self.val = nn.Sequential(NoisyLinear(hidden, hidden), nn.ReLU(), NoisyLinear(hidden, 1))
        self.adv = nn.Sequential(NoisyLinear(hidden, hidden), nn.ReLU(), NoisyLinear(hidden, 5))
    def forward(self, x):
        f = self.feature(x)
        return self.val(f) + self.adv(f) - self.adv(f).mean(dim=1, keepdim=True)
    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear): m.reset_noise()

class SumTree:
    def __init__(self, cap):
        self.cap = cap; self.tree = np.zeros(2*cap, dtype=np.float32); self.data = [None]*cap
        self.write = 0; self.n = 0
        
    def _prop(self, i, d):
        p = (i-1)//2; self.tree[p] += d
        if p != 0: self._prop(p, d)
        
    def total(self): 
        return self.tree[0]
        
    def max(self): 
        return self.tree[self.cap-1:self.cap-1+self.n].max() if self.n>0 else 1.0
        
    def add(self, p, d):
        i = self.write + self.cap - 1; self._prop(i, p-self.tree[i]); self.tree[i] = p; self.data[self.write] = d
        self.write = (self.write+1)%self.cap; self.n = min(self.n+1, self.cap)
        
    def update(self, i, p): 
        self._prop(i, p-self.tree[i]); self.tree[i] = p
        
    def get(self, v):
        i = 0
        while i < self.cap-1:
            l, r = 2*i+1, 2*i+2
            if v <= self.tree[l]: i = l
            else: v -= self.tree[l]; i = r
            
        dataIdx = i - (self.cap - 1)
        
        # FIX: Floating point safety check. 
        # If precision errors push us into an uninitialized node, fallback to the latest valid node.
        if self.data[dataIdx] is None:
            dataIdx = (self.write - 1) % self.cap
            i = dataIdx + self.cap - 1
            
        return i, self.tree[i], self.data[dataIdx]

class PER:
    def __init__(self, cap, alpha=0.6, beta=0.4, eps=1e-6):
        self.tree = SumTree(cap); self.alpha = alpha; self.beta = beta; self.eps = eps
        
    def add(self, t, td=1.0): 
        self.tree.add((abs(td)+self.eps)**self.alpha, t)
        
    def sample(self, batch, beta):
        idxs, priorities, samples = [], [], []
        segment = self.tree.total()/batch
        for i in range(batch):
            a, b = segment*i, segment*(i+1)
            idx, p, d = self.tree.get(np.random.uniform(a,b))
            idxs.append(idx); priorities.append(p); samples.append(d)
        
        probs = np.array(priorities)/self.tree.total()
        
        # FIX: Changed len(self.tree) to self.tree.n
        weights = (self.tree.n * probs)**(-beta) 
        weights /= weights.max()
        
        s = np.stack([x[0] for x in samples]).astype(np.float32)
        a = np.array([x[1] for x in samples], dtype=np.int64)
        r = np.array([x[2] for x in samples], dtype=np.float32)
        s2 = np.stack([x[3] for x in samples]).astype(np.float32)
        d = np.array([x[4] for x in samples], dtype=np.float32)
        return s, a, r, s2, d, weights.astype(np.float32), idxs
        
    def update_priorities(self, idxs, td):
        for i, e in zip(idxs, td): 
            self.tree.update(i, (abs(e)+self.eps)**self.alpha)
            
    def __len__(self): 
        return self.tree.n

def import_obelix(p):
    import importlib.util
    spec = importlib.util.spec_from_file_location("ob", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.OBELIX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", required=True)
    ap.add_argument("--out", default="d3qn_per_noisy_stack.pt")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--difficulty", type=int, default=3)
    ap.add_argument("--wall_obstacles", action="store_true")
    # FIX: Added missing scaling factor argument to match OBELIX expectations
    ap.add_argument("--scaling_factor", type=int, default=5) 
    args = ap.parse_args()

    OBELIX = import_obelix(args.obelix_py)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    q = NoisyDuelingDQN().to(device)
    tgt = NoisyDuelingDQN().to(device)
    tgt.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=1e-4)
    
    replay = PER(cap=100000)
    steps = 0; best = float("-inf"); recent = deque(maxlen=50)
    beta = 0.4
    
    for ep in tqdm(range(args.episodes)):
        # FIX: Added scaling_factor into environment init
        env = OBELIX(
            scaling_factor=args.scaling_factor, 
            difficulty=args.difficulty, 
            wall_obstacles=args.wall_obstacles, 
            seed=ep
        )
        obs = env.reset(seed=ep)
        
        fbuf = deque(maxlen=STACK_SIZE)
        for _ in range(STACK_SIZE): fbuf.append(np.zeros(18, dtype=np.float32))
        fbuf.append(obs.astype(np.float32))
        stacked = np.concatenate(list(fbuf), axis=0)
        
        ep_ret = 0.0; q.reset_noise()
        
        for _ in range(2000):
            with torch.no_grad():
                a = int(q(torch.tensor(stacked).unsqueeze(0).to(device)).argmax(dim=1).item())
            s2, r, done = env.step(ACTIONS[a], render=False)
            ep_ret += r
            fbuf.append(s2.astype(np.float32))
            s_next = np.concatenate(list(fbuf), axis=0)
            replay.add((stacked, a, r, s_next, done))
            stacked = s_next; steps += 1
            
            if len(replay) >= 1000 and steps % 4 == 0:
                beta = min(1.0, beta + 0.001)
                s, a, r, s2, d, w, idxs = replay.sample(64, beta)
                s, a, r, s2, d, w = [torch.tensor(x).to(device) for x in [s, a, r, s2, d, w]]
                
                with torch.no_grad():
                    next_a = q(s2).argmax(dim=1)
                    next_val = tgt(s2).gather(1, next_a.unsqueeze(1)).squeeze(1)
                    y = r + 0.99*(1-d)*next_val
                
                pred = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
                td = (y - pred).detach().cpu().numpy()
                replay.update_priorities(idxs, td)
                
                loss = (w * nn.functional.smooth_l1_loss(pred, y, reduction='none')).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(q.parameters(), 1.0); opt.step()
                q.reset_noise()
                
                if steps % 1000 == 0: tgt.load_state_dict(q.state_dict())
            if done: break
        
        recent.append(ep_ret)
        avg = sum(recent)/len(recent)
        if avg > best: best = avg; torch.save(q.state_dict(), args.out)
    
    print(f"Best: {best:.1f}")

if __name__ == "__main__":
    main()