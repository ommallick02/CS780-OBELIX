"""
Training script for D3RQN-PER (Dueling Deep Recurrent Q-Network with Prioritized Experience Replay) for OBELIX.

Example:
python train_d3rqn_per.py --obelix_py ../../../obelix.py --wall_obstacles

            ALGORITHM: DUELING DEEP RECURRENT Q-NETWORK + PRIORITIZED EXPERIENCE REPLAY (D3RQN-PER)

D3RQN-PER combines four improvements:
1. LSTM for temporal memory (like DRQN)
2. Double DQN for reduced overestimation bias
3. Dueling architecture for better value estimation
4. Prioritized Experience Replay for efficient learning

Architecture:
    Input (18-dim observation)
        ↓
    LSTM Layer (hidden_size=64)
        ↓
    Value Stream → V(s)        ↘
                               → Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
    Advantage Stream → A(s,a) ↗

Key features:
- LSTM maintains hidden state across time steps within an episode
- Dueling streams separate state value from action advantages
- PER samples important transitions more frequently
- Double DQN target calculation for stable learning
- Saves best model during training (overwrites previous)
"""

from __future__ import annotations
import argparse, copy, random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


# ── Dueling LSTM Q-Network ───────────────────────────────────────────────────
class DuelingLSTMDQN(nn.Module):
    """Dueling DQN with LSTM for temporal memory."""

    def __init__(self, in_dim: int = 18, hidden_dim: int = 64, n_actions: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(in_dim, hidden_dim, batch_first=True)

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)

        features = lstm_out[:, -1, :]
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values, hidden

    def init_hidden(self, batch_size: int = 1, device: str = "cpu"):
        h = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return (h, c)


# ── Sum-tree for PER ─────────────────────────────────────────────────────────
class SumTree:
    """Binary tree for efficient prioritized sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float32)
        self.data: List = [None] * capacity
        self.write = 0
        self.n_stored = 0

    def _propagate(self, idx: int, delta: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    @property
    def total(self) -> float:
        return float(self.tree[0])

    @property
    def max_priority(self) -> float:
        leaves = self.tree[self.capacity - 1 : self.capacity - 1 + self.n_stored]
        return float(leaves.max()) if self.n_stored > 0 else 1.0

    def add(self, priority: float, data) -> None:
        leaf = self.write + self.capacity - 1
        delta = priority - self.tree[leaf]
        self.tree[leaf] = priority
        self._propagate(leaf, delta)
        self.data[self.write] = data
        self.write = (self.write + 1) % self.capacity
        self.n_stored = min(self.n_stored + 1, self.capacity)

    def update(self, leaf_idx: int, priority: float) -> None:
        delta = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)

    def get(self, value: float) -> Tuple[int, float, object]:
        for _ in range(20):
            v = min(max(value, 0.0), self.tree[0] - 1e-6)
            idx = 0
            while idx < self.capacity - 1:
                left = 2 * idx + 1
                right = left + 1
                if v <= self.tree[left]:
                    idx = left
                else:
                    v -= self.tree[left]
                    idx = right
            data_idx = idx - (self.capacity - 1)
            if self.data[data_idx] is not None:
                return idx, float(self.tree[idx]), self.data[data_idx]
            value = np.random.uniform(0.0, self.tree[0])

        for i in range(self.n_stored):
            if self.data[i] is not None:
                leaf = self.capacity - 1 + i
                return leaf, float(self.tree[leaf]), self.data[i]

        raise RuntimeError("SumTree.get: no valid transition")

    def __len__(self) -> int:
        return self.n_stored


# ── Prioritized Replay Buffer ─────────────────────────────────────────────────
@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool


class PrioritizedReplay:
    def __init__(self, capacity: int = 100_000, alpha: float = 0.6, beta_start: float = 0.4, eps: float = 1e-6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.eps = eps

    def add(self, t: Transition) -> None:
        priority = (self.tree.max_priority + self.eps) ** self.alpha
        self.tree.add(priority, t)

    def sample(self, batch: int, beta: float):
        n = len(self.tree)
        segment = self.tree.total / batch

        leaf_idxs: List[int] = []
        priorities: List[float] = []
        transitions: List[Transition] = []

        for i in range(batch):
            lo = max(0.0, segment * i)
            hi = min(self.tree.total - 1e-6, segment * (i + 1))
            val = np.random.uniform(lo, hi)
            leaf_idx, priority, data = self.tree.get(val)
            leaf_idxs.append(leaf_idx)
            priorities.append(priority)
            transitions.append(data)

        probs = np.array(priorities, dtype=np.float32) / self.tree.total
        is_weights = (1.0 / (n * probs + 1e-10)) ** beta
        is_weights /= is_weights.max()

        s = np.stack([t.s for t in transitions]).astype(np.float32)
        a = np.array([t.a for t in transitions], dtype=np.int64)
        r = np.array([t.r for t in transitions], dtype=np.float32)
        s2 = np.stack([t.s2 for t in transitions]).astype(np.float32)
        d = np.array([t.done for t in transitions], dtype=np.float32)
        return s, a, r, s2, d, is_weights.astype(np.float32), leaf_idxs

    def update_priorities(self, leaf_idxs: List[int], td_errors: np.ndarray) -> None:
        for idx, err in zip(leaf_idxs, td_errors):
            self.tree.update(idx, (abs(float(err)) + self.eps) ** self.alpha)

    def __len__(self) -> int:
        return len(self.tree)


def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights.pth")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--difficulty", type=int, default=3)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)

    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--replay", type=int, default=100_000)
    ap.add_argument("--warmup", type=int, default=2_000)
    ap.add_argument("--target_sync", type=int, default=2_000)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay_steps", type=int, default=200_000)

    ap.add_argument("--per_alpha", type=float, default=0.6)
    ap.add_argument("--per_beta_start", type=float, default=0.4)
    ap.add_argument("--per_eps", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lstm_hidden", type=int, default=64)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    print(f"Using device: {device}")

    OBELIX = import_obelix(args.obelix_py)

    q = DuelingLSTMDQN(in_dim=18, hidden_dim=args.lstm_hidden, n_actions=5).to(device)
    tgt = DuelingLSTMDQN(in_dim=18, hidden_dim=args.lstm_hidden, n_actions=5).to(device)
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt = optim.Adam(q.parameters(), lr=args.lr)
    replay = PrioritizedReplay(
        capacity=args.replay,
        alpha=args.per_alpha,
        beta_start=args.per_beta_start,
        eps=args.per_eps
    )
    steps = 0
    total_steps = args.episodes * args.max_steps

    recent_returns = deque(maxlen=50)
    best_avg_return = float("-inf")

    def eps_by_step(t: int) -> float:
        if t >= args.eps_decay_steps:
            return args.eps_end
        return args.eps_start + (t / args.eps_decay_steps) * (args.eps_end - args.eps_start)

    def beta_by_step(t: int) -> float:
        return min(1.0, args.per_beta_start + (1.0 - args.per_beta_start) * (t / total_steps))

    pbar = tqdm(range(args.episodes), desc="D3RQN-PER", unit="ep", dynamic_ncols=True)

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

        s = env.reset(seed=args.seed + ep)
        ep_ret = 0.0
        last_loss = float("nan")
        hidden = q.init_hidden(batch_size=1, device=device)

        for _ in range(args.max_steps):
            eps = eps_by_step(steps)

            with torch.no_grad():
                s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                q_values, hidden = q(s_t, hidden)

                if np.random.rand() < eps:
                    a = np.random.randint(len(ACTIONS))
                else:
                    a = int(q_values.squeeze(0).argmax().item())

            hidden = (hidden[0].detach(), hidden[1].detach())

            s2, r, done = env.step(ACTIONS[a], render=False)
            ep_ret += float(r)
            replay.add(Transition(s=s, a=a, r=float(r), s2=s2, done=bool(done)))
            s = s2
            steps += 1

            if len(replay) >= max(args.warmup, args.batch):
                beta = beta_by_step(steps)
                sb, ab, rb, s2b, db, is_w, leaf_idxs = replay.sample(args.batch, beta)

                sb_t = torch.tensor(sb, device=device)
                ab_t = torch.tensor(ab, device=device)
                rb_t = torch.tensor(rb, device=device)
                s2b_t = torch.tensor(s2b, device=device)
                db_t = torch.tensor(db, device=device)
                w_t = torch.tensor(is_w, device=device)

                with torch.no_grad():
                    next_q, _ = q(s2b_t.unsqueeze(1))
                    next_a = torch.argmax(next_q, dim=1)

                    next_q_tgt, _ = tgt(s2b_t.unsqueeze(1))
                    next_val = next_q_tgt.gather(1, next_a.unsqueeze(1)).squeeze(1)
                    y = rb_t + args.gamma * (1.0 - db_t) * next_val

                pred, _ = q(sb_t.unsqueeze(1))
                pred = pred.gather(1, ab_t.unsqueeze(1)).squeeze(1)
                td_errors = (y - pred).detach()

                loss = (w_t * nn.functional.smooth_l1_loss(pred, y, reduction="none")).mean()
                last_loss = loss.item()

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

                replay.update_priorities(leaf_idxs, td_errors.cpu().numpy())

                if steps % args.target_sync == 0:
                    tgt.load_state_dict(q.state_dict())

            if done:
                break

        recent_returns.append(ep_ret)
        avg_return = sum(recent_returns) / len(recent_returns)

        if avg_return > best_avg_return:
            best_avg_return = avg_return
            torch.save(q.state_dict(), args.out)
            pbar.set_postfix({
                "return": f"{ep_ret:>8.1f}",
                "avg50": f"{avg_return:>8.1f}",
                "loss": f"{last_loss:.4f}",
                "eps": f"{eps_by_step(steps):.3f}",
                "beta": f"{beta_by_step(steps):.3f}",
                "replay": f"{len(replay):>6}",
                "saved": "NEW BEST",
            })
        else:
            pbar.set_postfix({
                "return": f"{ep_ret:>8.1f}",
                "avg50": f"{avg_return:>8.1f}",
                "loss": f"{last_loss:.4f}",
                "eps": f"{eps_by_step(steps):.3f}",
                "beta": f"{beta_by_step(steps):.3f}",
                "replay": f"{len(replay):>6}",
            })

    pbar.close()
    print(f"Training complete. Best model (avg50={best_avg_return:.1f}) saved at: {args.out}")


if __name__ == "__main__":
    main()
