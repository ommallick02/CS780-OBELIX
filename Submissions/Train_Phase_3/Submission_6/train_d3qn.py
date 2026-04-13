"""Offline trainer: Dueling Double DQN (D3QN) + replay buffer for OBELIX.

Run locally to create weights.pth, then submit agent.py + weights.pth.

Example:
python train_d3qn.py --obelix_py ../../../obelix.py --wall_obstacles

                    ALGORITHM: DUELING DOUBLE DEEP Q-NETWORK (D3QN)

D3QN combines two independent improvements over vanilla DQN:

1. DOUBLE DQN  (removes overestimation bias)
   Use the online network to SELECT the next action, target network to EVALUATE:
       target = r + γ × Q_target( s', argmax_a Q_online(s', a) )

2. DUELING ARCHITECTURE  (better value estimation)
   Instead of mapping obs directly to Q-values, the network splits into two
   streams after the shared encoder:

       • Value stream     V(s)      — how good is this state in general?
       • Advantage stream A(s, a)  — how much better is action a vs average?

   Recombined as:
       Q(s, a) = V(s) + ( A(s, a) − mean_a A(s, a) )

   Subtracting the mean keeps the decomposition unique (identifiability) and
   prevents advantage values from drifting to arbitrary scales.

   Original paper: https://arxiv.org/abs/1511.06581
   Double DQN paper: https://arxiv.org/pdf/1509.06461
"""

from __future__ import annotations
import argparse, copy, random
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


# ── Dueling network ───────────────────────────────────────────────────────────
class DuelingDQN(nn.Module):
    """
    Shared encoder → Value head + Advantage head → Q recombination.

        obs (18,)
          └─ encoder: Linear(18→64) → ReLU → Linear(64→64) → ReLU
                ├─ value_head:     Linear(64→1)         →  V(s)
                └─ advantage_head: Linear(64→n_actions) →  A(s, a)

        Q(s, a) = V(s) + A(s, a) − mean_a A(s, a)
    """
    def __init__(self, in_dim: int = 18, n_actions: int = 5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),     nn.ReLU(),
        )
        self.value_head     = nn.Linear(64, 1)
        self.advantage_head = nn.Linear(64, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.encoder(x)
        v = self.value_head(f)
        a = self.advantage_head(f)
        return v + a - a.mean(dim=-1, keepdim=True)


# ── Replay buffer ─────────────────────────────────────────────────────────────
@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool

class Replay:
    def __init__(self, cap: int = 100_000):
        self.buf: Deque[Transition] = deque(maxlen=cap)
    def add(self, t: Transition):
        self.buf.append(t)
    def sample(self, batch: int):
        idx = np.random.choice(len(self.buf), size=batch, replace=False)
        items = [self.buf[i] for i in idx]
        s  = np.stack([it.s  for it in items]).astype(np.float32)
        a  = np.array([it.a  for it in items], dtype=np.int64)
        r  = np.array([it.r  for it in items], dtype=np.float32)
        s2 = np.stack([it.s2 for it in items]).astype(np.float32)
        d  = np.array([it.done for it in items], dtype=np.float32)
        return s, a, r, s2, d
    def __len__(self): return len(self.buf)


# ── Obelix loader ─────────────────────────────────────────────────────────────
def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",       type=str,   required=True)
    ap.add_argument("--out",             type=str,   default="weights.pth")
    ap.add_argument("--episodes",        type=int,   default=5000)
    ap.add_argument("--max_steps",       type=int,   default=2000)
    ap.add_argument("--difficulty",      type=int,   default=3)
    ap.add_argument("--wall_obstacles",  action="store_true")
    ap.add_argument("--box_speed",       type=int,   default=2)
    ap.add_argument("--scaling_factor",  type=int,   default=5)
    ap.add_argument("--arena_size",      type=int,   default=500)
    ap.add_argument("--gamma",           type=float, default=0.99)
    ap.add_argument("--lr",              type=float, default=1e-3)
    ap.add_argument("--batch",           type=int,   default=256)
    ap.add_argument("--replay",          type=int,   default=100_000)
    ap.add_argument("--warmup",          type=int,   default=2_000)
    ap.add_argument("--target_sync",     type=int,   default=2_000)
    ap.add_argument("--eps_start",       type=float, default=1.0)
    ap.add_argument("--eps_end",         type=float, default=0.05)
    ap.add_argument("--eps_decay_steps", type=int,   default=200_000)
    ap.add_argument("--seed",            type=int,   default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    q   = DuelingDQN()
    tgt = DuelingDQN()
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt    = optim.Adam(q.parameters(), lr=args.lr)
    replay = Replay(args.replay)
    steps  = 0

    recent_returns  = deque(maxlen=50)
    best_avg_return = float("-inf")

    def eps_by_step(t: int) -> float:
        if t >= args.eps_decay_steps:
            return args.eps_end
        return args.eps_start + (t / args.eps_decay_steps) * (args.eps_end - args.eps_start)

    pbar = tqdm(range(args.episodes), desc="D3QN", unit="ep", dynamic_ncols=True)

    for ep in pbar:
        env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=args.seed + ep,
        )
        s         = env.reset(seed=args.seed + ep)
        ep_ret    = 0.0
        last_loss = float("nan")

        for _ in range(args.max_steps):
            eps = eps_by_step(steps)
            if np.random.rand() < eps:
                a = np.random.randint(len(ACTIONS))
            else:
                with torch.no_grad():
                    qs = q(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                a = int(np.argmax(qs))

            s2, r, done = env.step(ACTIONS[a], render=False)
            ep_ret += float(r)
            replay.add(Transition(s=s, a=a, r=float(r), s2=s2, done=bool(done)))
            s      = s2
            steps += 1

            if len(replay) >= max(args.warmup, args.batch):
                sb, ab, rb, s2b, db = replay.sample(args.batch)
                sb_t  = torch.tensor(sb)
                ab_t  = torch.tensor(ab)
                rb_t  = torch.tensor(rb)
                s2b_t = torch.tensor(s2b)
                db_t  = torch.tensor(db)

                with torch.no_grad():
                    next_a   = q(s2b_t).argmax(dim=1)
                    next_val = tgt(s2b_t).gather(1, next_a.unsqueeze(1)).squeeze(1)
                    y = rb_t + args.gamma * (1.0 - db_t) * next_val

                pred      = q(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)
                loss      = nn.functional.smooth_l1_loss(pred, y)
                last_loss = loss.item()

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

                if steps % args.target_sync == 0:
                    tgt.load_state_dict(q.state_dict())

            if done:
                break

        recent_returns.append(ep_ret)
        avg_return = sum(recent_returns) / len(recent_returns)

        # Save best model immediately when new best is found (overwrite previous)
        if avg_return > best_avg_return:
            best_avg_return = avg_return
            torch.save(q.state_dict(), args.out)
            pbar.set_postfix({
                "return": f"{ep_ret:>8.1f}",
                "avg50":  f"{avg_return:>8.1f}",
                "loss":   f"{last_loss:.4f}",
                "eps":    f"{eps_by_step(steps):.3f}",
                "replay": f"{len(replay):>6}",
                "saved":  "NEW BEST",
            })
        else:
            pbar.set_postfix({
                "return": f"{ep_ret:>8.1f}",
                "avg50":  f"{avg_return:>8.1f}",
                "loss":   f"{last_loss:.4f}",
                "eps":    f"{eps_by_step(steps):.3f}",
                "replay": f"{len(replay):>6}",
            })

    pbar.close()

    print(f"Training complete. Best model (avg50={best_avg_return:.1f}) saved at: {args.out}")


if __name__ == "__main__":
    main()
