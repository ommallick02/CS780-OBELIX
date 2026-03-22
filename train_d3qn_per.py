"""Offline trainer: Dueling Double DQN + Prioritized Experience Replay (D3QN-PER) for OBELIX.

Run locally to create weights.pth, then submit agent.py + weights.pth.

Example:
  python train_d3qn_per.py --obelix_py ./obelix.py --out weights.pth --episodes 2000 --difficulty 0 --wall_obstacles

            ALGORITHM: DUELING DOUBLE DQN + PRIORITIZED EXPERIENCE REPLAY (D3QN-PER)

This combines three improvements over vanilla DQN:

1. DOUBLE DQN  (removes overestimation bias)
   Use online network to SELECT the next action, target network to EVALUATE:
       target = r + γ × Q_target( s', argmax_a Q_online(s', a) )

2. DUELING ARCHITECTURE  (better value estimation)
   Split the network into a Value stream V(s) and Advantage stream A(s,a):
       Q(s, a) = V(s) + A(s, a) − mean_a A(s, a)

3. PRIORITIZED EXPERIENCE REPLAY  (learn more from surprising transitions)
   Sample transitions proportional to |TD-error|^alpha instead of uniformly:
       P(i) = p_i^alpha / Σ p_j^alpha      where p_i = |δ_i| + eps

   Correct the sampling bias with importance-sampling (IS) weights:
       w_i = ( 1 / (N · P(i)) )^beta

   beta is annealed from beta_start → 1.0 over training so the correction
   becomes exact only at convergence.

   SumTree gives O(log N) sampling and O(log N) priority updates.

   Original PER paper: https://arxiv.org/abs/1511.05952
   Dueling DQN paper:  https://arxiv.org/abs/1511.06581
   Double DQN paper:   https://arxiv.org/pdf/1509.06461
"""

from __future__ import annotations
import argparse, random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


# ── Dueling network ───────────────────────────────────────────────────────────
class DuelingDQN(nn.Module):
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


# ── Sum-tree ──────────────────────────────────────────────────────────────────
class SumTree:
    """
    Binary tree: leaves hold priorities, internal nodes hold subtree sums.

    Layout (capacity=4):
        index:  0          <- root (total sum)
               1    2
              3  4  5  6   <- leaves (priorities)

    O(log N) update, O(log N) sampling, O(1) total-sum and max-priority.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity, dtype=np.float32)
        self.data: List = [None] * capacity
        self.write    = 0
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
        leaf  = self.write + self.capacity - 1
        delta = priority - self.tree[leaf]
        self.tree[leaf] = priority
        self._propagate(leaf, delta)
        self.data[self.write] = data
        self.write    = (self.write + 1) % self.capacity
        self.n_stored = min(self.n_stored + 1, self.capacity)

    def update(self, leaf_idx: int, priority: float) -> None:
        delta = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)

    def get(self, value: float) -> Tuple[int, float, object]:
        """Walk the tree to find the leaf whose cumulative range contains value.

        Retries with a fresh random value if the landed slot is still None
        (can happen due to floating-point drift onto zero-priority leaves).
        """
        for _ in range(20):
            v   = min(max(value, 0.0), self.tree[0] - 1e-6)
            idx = 0
            while idx < self.capacity - 1:
                left  = 2 * idx + 1
                right = left + 1
                if v <= self.tree[left]:
                    idx = left
                else:
                    v  -= self.tree[left]
                    idx = right
            data_idx = idx - (self.capacity - 1)
            if self.data[data_idx] is not None:
                return idx, float(self.tree[idx]), self.data[data_idx]
            # Drifted onto empty leaf — resample uniformly within valid range
            value = np.random.uniform(0.0, self.tree[0])

        # Fallback: linear scan for first populated slot
        for i in range(self.n_stored):
            if self.data[i] is not None:
                leaf = self.capacity - 1 + i
                return leaf, float(self.tree[leaf]), self.data[i]

        raise RuntimeError("SumTree.get: no valid transition — buffer may be empty")

    def __len__(self) -> int:
        return self.n_stored


# ── Prioritized replay buffer ─────────────────────────────────────────────────
@dataclass
class Transition:
    s:    np.ndarray
    a:    int
    r:    float
    s2:   np.ndarray
    done: bool


class PrioritizedReplay:
    def __init__(
        self,
        capacity:   int   = 100_000,
        alpha:      float = 0.6,
        beta_start: float = 0.4,
        eps:        float = 1e-6,
    ):
        self.tree  = SumTree(capacity)
        self.alpha = alpha
        self.eps   = eps

    def add(self, t: Transition) -> None:
        # New transitions get max priority so they are seen at least once
        priority = (self.tree.max_priority + self.eps) ** self.alpha
        self.tree.add(priority, t)

    def sample(
        self, batch: int, beta: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, List[int]]:
        n       = len(self.tree)
        segment = self.tree.total / batch

        leaf_idxs:   List[int]        = []
        priorities:  List[float]      = []
        transitions: List[Transition] = []

        for i in range(batch):
            # Clamp to (0, total) so we never sample beyond initialised leaves
            lo  = max(0.0, segment * i)
            hi  = min(self.tree.total - 1e-6, segment * (i + 1))
            val = np.random.uniform(lo, hi)
            leaf_idx, priority, data = self.tree.get(val)
            leaf_idxs.append(leaf_idx)
            priorities.append(priority)
            transitions.append(data)

        # IS weights normalised so max weight = 1
        probs      = np.array(priorities, dtype=np.float32) / self.tree.total
        is_weights = (1.0 / (n * probs + 1e-10)) ** beta
        is_weights /= is_weights.max()

        s  = np.stack([t.s  for t in transitions]).astype(np.float32)
        a  = np.array([t.a  for t in transitions], dtype=np.int64)
        r  = np.array([t.r  for t in transitions], dtype=np.float32)
        s2 = np.stack([t.s2 for t in transitions]).astype(np.float32)
        d  = np.array([t.done for t in transitions], dtype=np.float32)
        return s, a, r, s2, d, is_weights.astype(np.float32), leaf_idxs

    def update_priorities(self, leaf_idxs: List[int], td_errors: np.ndarray) -> None:
        for idx, err in zip(leaf_idxs, td_errors):
            self.tree.update(idx, (abs(float(err)) + self.eps) ** self.alpha)

    def __len__(self) -> int:
        return len(self.tree)


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
    ap.add_argument("--episodes",        type=int,   default=2000)
    ap.add_argument("--max_steps",       type=int,   default=1000)
    ap.add_argument("--difficulty",      type=int,   default=0)
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
    # PER hyperparameters
    ap.add_argument("--per_alpha",       type=float, default=0.6,
                    help="Priority exponent (0=uniform, 1=fully prioritised)")
    ap.add_argument("--per_beta_start",  type=float, default=0.4,
                    help="Initial IS-weight exponent beta (annealed to 1.0)")
    ap.add_argument("--per_eps",         type=float, default=1e-6,
                    help="Priority floor to prevent zero-probability transitions")
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
    replay = PrioritizedReplay(
        capacity=args.replay,
        alpha=args.per_alpha,
        beta_start=args.per_beta_start,
        eps=args.per_eps,
    )
    steps       = 0
    total_steps = args.episodes * args.max_steps   # for beta annealing

    recent_returns = deque(maxlen=50)

    def eps_by_step(t: int) -> float:
        if t >= args.eps_decay_steps:
            return args.eps_end
        return args.eps_start + (t / args.eps_decay_steps) * (args.eps_end - args.eps_start)

    def beta_by_step(t: int) -> float:
        """Linearly anneal beta from beta_start to 1.0 over the full run."""
        return min(1.0, args.per_beta_start + (1.0 - args.per_beta_start) * (t / total_steps))

    pbar = tqdm(range(args.episodes), desc="D3QN-PER", unit="ep", dynamic_ncols=True)

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
                beta = beta_by_step(steps)
                sb, ab, rb, s2b, db, is_w, leaf_idxs = replay.sample(args.batch, beta)

                sb_t  = torch.tensor(sb)
                ab_t  = torch.tensor(ab)
                rb_t  = torch.tensor(rb)
                s2b_t = torch.tensor(s2b)
                db_t  = torch.tensor(db)
                w_t   = torch.tensor(is_w)

                # Double DQN target using dueling Q-values
                with torch.no_grad():
                    next_a   = q(s2b_t).argmax(dim=1)
                    next_val = tgt(s2b_t).gather(1, next_a.unsqueeze(1)).squeeze(1)
                    y = rb_t + args.gamma * (1.0 - db_t) * next_val

                pred      = q(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)
                td_errors = (y - pred).detach()

                # IS-weighted loss: scale each sample's error by w_i
                loss      = (w_t * nn.functional.smooth_l1_loss(
                    pred, y, reduction="none"
                )).mean()
                last_loss = loss.item()

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

                # Feed new TD-errors back into the tree
                replay.update_priorities(leaf_idxs, td_errors.cpu().numpy())

                if steps % args.target_sync == 0:
                    tgt.load_state_dict(q.state_dict())

            if done:
                break

        recent_returns.append(ep_ret)
        avg_return = sum(recent_returns) / len(recent_returns)
        pbar.set_postfix({
            "return": f"{ep_ret:>8.1f}",
            "avg50":  f"{avg_return:>8.1f}",
            "loss":   f"{last_loss:.4f}",
            "eps":    f"{eps_by_step(steps):.3f}",
            "beta":   f"{beta_by_step(steps):.3f}",
            "replay": f"{len(replay):>6}",
        })

    pbar.close()
    torch.save(q.state_dict(), args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()