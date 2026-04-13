"""Train D3QN-Noisy with State Stacking."""

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
        # Store dimensions as instance attributes
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
        # Use no_grad() to avoid in-place operation error
        with torch.no_grad():
            self.weight_mu.uniform_(-mu_r, mu_r)
            self.weight_sigma.fill_(self.sigma_init/np.sqrt(self.in_f))
            self.bias_mu.uniform_(-mu_r, mu_r)
            self.bias_sigma.fill_(self.sigma_init/np.sqrt(self.out_f))
            
    def reset_noise(self):
        eps_in = torch.randn(self.in_f).sign().mul_(torch.randn(self.in_f).abs().sqrt())
        eps_out = torch.randn(self.out_f).sign().mul_(torch.randn(self.out_f).abs().sqrt())
        self.w_eps.copy_(eps_out.outer(eps_in))
        self.b_eps.copy_(eps_out)
        
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
            if isinstance(m, NoisyLinear): 
                m.reset_noise()

class Replay:
    def __init__(self, cap=100000): 
        self.buf = deque(maxlen=cap)
        
    def push(self, *t): 
        self.buf.append(t)
        
    def sample(self, batch):
        idx = np.random.choice(len(self.buf), batch, replace=False)
        s, a, r, s2, d = zip(*[self.buf[i] for i in idx])
        
        return [
            torch.FloatTensor(np.array(s)),
            torch.LongTensor(np.array(a)),    # Actions are indices, must be int64
            torch.FloatTensor(np.array(r)),
            torch.FloatTensor(np.array(s2)),
            torch.FloatTensor(np.array(d))
        ]
    
    def __len__(self): 
        return len(self.buf)

def import_obelix(p):
    import importlib.util
    spec = importlib.util.spec_from_file_location("ob", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", required=True)
    ap.add_argument("--out", default="d3qn_noisy_stack.pt")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--difficulty", type=int, default=3)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--scaling_factor", type=int, default=5)  # Added
    args = ap.parse_args()

    OBELIX = import_obelix(args.obelix_py)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    q = NoisyDuelingDQN().to(device)
    tgt = NoisyDuelingDQN().to(device)
    tgt.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=1e-4)
    replay = Replay()
    
    steps = 0
    best = float("-inf")
    recent = deque(maxlen=50)
    
    for ep in tqdm(range(args.episodes)):
        # Fixed: added scaling_factor
        env = OBELIX(
            scaling_factor=args.scaling_factor,
            difficulty=args.difficulty, 
            wall_obstacles=args.wall_obstacles, 
            seed=ep
        )
        obs = env.reset(seed=ep)
        
        fbuf = deque(maxlen=STACK_SIZE)
        for _ in range(STACK_SIZE): 
            fbuf.append(np.zeros(18, dtype=np.float32))
        fbuf.append(obs.astype(np.float32))
        stacked = np.concatenate(list(fbuf), axis=0)
        
        ep_ret = 0.0
        q.reset_noise()
        
        for _ in range(2000):
            with torch.no_grad():
                a = int(q(torch.tensor(stacked).unsqueeze(0).to(device)).argmax(dim=1).item())
            s2, r, done = env.step(ACTIONS[a], render=False)
            ep_ret += r
            fbuf.append(s2.astype(np.float32))
            s_next = np.concatenate(list(fbuf), axis=0)
            replay.push(stacked, a, r, s_next, done)
            stacked = s_next
            steps += 1
            
            if len(replay) >= 640 and steps % 4 == 0:
                s, a, r, s2, d = [x.to(device) for x in replay.sample(64)]
                with torch.no_grad():
                    next_a = q(s2).argmax(dim=1)
                    next_val = tgt(s2).gather(1, next_a.unsqueeze(1)).squeeze(1)
                    y = r + 0.99*(1-d)*next_val
                pred = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 1.0)
                opt.step()
                q.reset_noise()
                if steps % 1000 == 0: 
                    tgt.load_state_dict(q.state_dict())
            if done: 
                break
        
        recent.append(ep_ret)
        avg = sum(recent)/len(recent)
        if avg > best: 
            best = avg
            torch.save(q.state_dict(), args.out)
    
    print(f"Best: {best:.1f}")

if __name__ == "__main__":
    main()