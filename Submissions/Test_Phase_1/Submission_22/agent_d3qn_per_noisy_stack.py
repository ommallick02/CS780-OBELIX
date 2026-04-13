"""D3QN-PER-Noisy with State Stacking."""

import os, numpy as np, torch, torch.nn as nn
from collections import deque

ACTIONS = ("L45", "L22", "FW", "R22", "R45")
N_ACT = 5
_STACK_SIZE = 4

class NoisyLinear(nn.Module):
    def __init__(self, in_f, out_f, sigma=0.017):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_f, in_f))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_f, in_f))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_f))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_f))
        self.register_buffer('w_eps', torch.FloatTensor(out_f, in_f))
        self.register_buffer('b_eps', torch.FloatTensor(out_f))
        self.sigma_init = sigma
        self.reset_params(); self.reset_noise()
    @torch.no_grad()
    def reset_params(self):
        mu_r = 1/np.sqrt(self.in_f)
        self.weight_mu.uniform_(-mu_r, mu_r)
        self.weight_sigma.fill_(self.sigma_init/np.sqrt(self.in_f))
        self.bias_mu.uniform_(-mu_r, mu_r)
        self.bias_sigma.fill_(self.sigma_init/np.sqrt(self.out_f))
    def reset_noise(self):
        eps_in = self._scale(self.in_f); eps_out = self._scale(self.out_f)
        self.w_eps.copy_(eps_out.outer(eps_in)); self.b_eps.copy_(eps_out)
    def _scale(self, size):
        x = torch.randn(size); return x.sign().mul_(x.abs().sqrt())
    def forward(self, x):
        if self.training: w = self.weight_mu + self.weight_sigma*self.w_eps; b = self.bias_mu + self.bias_sigma*self.b_eps
        else: w, b = self.weight_mu, self.bias_mu
        return nn.functional.linear(x, w, b)

class NoisyDuelingDQN(nn.Module):
    def __init__(self, in_dim=18*_STACK_SIZE, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(NoisyLinear(in_dim, hidden), nn.ReLU())
        self.val = nn.Sequential(NoisyLinear(hidden, hidden), nn.ReLU(), NoisyLinear(hidden, 1))
        self.adv = nn.Sequential(NoisyLinear(hidden, hidden), nn.ReLU(), NoisyLinear(hidden, N_ACT))
    def forward(self, x):
        f = self.feature(x)
        return self.val(f) + self.adv(f) - self.adv(f).mean(dim=1, keepdim=True)
    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear): m.reset_noise()

_NETWORK = None
_fbuf = None

def _reset_buf(obs):
    global _fbuf
    _fbuf = deque(maxlen=_STACK_SIZE)
    for _ in range(_STACK_SIZE-1): _fbuf.append(np.zeros(18, dtype=np.float32))
    _fbuf.append(obs.astype(np.float32).copy())

def _load():
    global _NETWORK
    if _NETWORK: return _NETWORK
    p = os.path.join(os.path.dirname(__file__), "d3qn_per_noisy_stack.pt")
    ckpt = torch.load(p, map_location="cpu")
    hidden = ckpt['feature.0.weight_mu'].shape[0]
    net = NoisyDuelingDQN(hidden=hidden)
    net.load_state_dict(ckpt)
    net.eval()
    _NETWORK = net
    return net

def policy(obs, rng):
    global _fbuf
    net = _load()
    if _fbuf is None: _reset_buf(obs)
    else: _fbuf.append(obs.astype(np.float32).copy())
    x = torch.FloatTensor(np.concatenate(list(_fbuf), axis=0)).unsqueeze(0)
    with torch.no_grad(): return ACTIONS[int(net(x).argmax(dim=1).item())]
