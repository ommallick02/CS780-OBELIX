"""
Training script for D3RQN-PER-Noisy (Dueling Deep Recurrent Q-Network with PER and Noisy Networks) for OBELIX.

Example:
python train_d3rqn_per_noisy.py --obelix_py ../../../obelix.py --wall_obstacles

        ALGORITHM: DUELING DEEP RECURRENT Q-NETWORK + PER + NOISY NETWORKS (D3RQN-PER-NOISY)

D3RQN-PER-Noisy combines ALL improvements:
1. LSTM for temporal memory (like DRQN)
2. Dueling architecture for better value estimation
3. Prioritized Experience Replay for efficient learning
4. Noisy Networks for learned exploration (replaces epsilon-greedy)
5. Double DQN for reduced overestimation bias

Architecture:
    Input (18-dim observation)
        ↓
    LSTM Layer (hidden_size=64)
        ↓
    Value Stream → V(s)        ↘
                               → Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
    Advantage Stream → A(s,a) ↗
    (Both streams use Noisy Linear layers)

Key features:
- LSTM maintains hidden state across time steps within an episode
- Dueling streams separate state value from action advantages
- PER samples important transitions more frequently
- Noisy Linear layers provide learned exploration
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


# ── Noisy Linear Layer ───────────────────────────────────────────────────────
class NoisyLinear(nn.Module):
    """Noisy linear layer with factorized noise."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))

        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)


# ── Dueling LSTM + Noisy Q-Network ───────────────────────────────────────────
class DuelingLSTMNoisyDQN(nn.Module):
    """Dueling DQN with LSTM and Noisy output layers."""

    def __init__(self, in_dim: int = 18, hidden_dim: int = 64, n_actions: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(in_dim, hidden_dim, batch_first=True)

        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, 1)
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, n_actions)
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

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# ── Sum-tree for PER ─────────────────────────────────────────────────────────
class SumTree:
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


def evaluate_agent(q_network, OBELIX, difficulty, wall_obstacles,
                   n_episodes=10, max_steps=1000, seed=99999):
    """Evaluate agent (no noise in eval mode)."""
    q_network.eval()
    rewards = []

    for i in range(n_episodes):
        env = OBELIX(
            scaling_factor=5,
            arena_size=500,
            max_steps=max_steps,
            wall_obstacles=wall_obstacles,
            difficulty=difficulty,
            box_speed=2,
            seed=seed + i
        )

        obs = env.reset(seed=seed + i)
        hidden = q_network.init_hidden(batch_size=1, device="cpu")
        total_reward = 0.0
        done = False

        for _ in range(max_steps):
            with torch.no_grad():
                s_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                q_values, hidden = q_network(s_t, hidden)
                hidden = (hidden[0].detach(), hidden[1].detach())
                a = int(q_values.squeeze(0).argmax().item())

            obs, reward, done = env.step(ACTIONS[a], render=False)
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)

    q_network.train()
    return np.mean(rewards)


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

    ap.add_argument("--per_alpha", type=float, default=0.6)
    ap.add_argument("--per_beta_start", type=float, default=0.4)
    ap.add_argument("--per_eps", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lstm_hidden", type=int, default=64)
    ap.add_argument("--eval_interval", type=int, default=50)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    print(f"Using device: {device}")

    OBELIX = import_obelix(args.obelix_py)

    q = DuelingLSTMNoisyDQN(in_dim=18, hidden_dim=args.lstm_hidden, n_actions=5).to(device)
    tgt = DuelingLSTMNoisyDQN(in_dim=18, hidden_dim=args.lstm_hidden, n_actions=5).to(device)
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

    recent_rewards = []
    best_eval_reward = float("-inf")

    def beta_by_step(t: int) -> float:
        return min(1.0, args.per_beta_start + (1.0 - args.per_beta_start) * (t / total_steps))

    pbar = tqdm(range(args.episodes), desc="D3RQN-PER-Noisy", unit="ep", dynamic_ncols=True)

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
            # Select action (exploration comes from network noise)
            with torch.no_grad():
                s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                q_values, hidden = q(s_t, hidden)
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

                # Reset noise after update
                q.reset_noise()
                tgt.reset_noise()

                replay.update_priorities(leaf_idxs, td_errors.cpu().numpy())

                if steps % args.target_sync == 0:
                    tgt.load_state_dict(q.state_dict())

            if done:
                break

        recent_rewards.append(ep_ret)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)

        pbar.update(1)

        # Evaluation and save best model
        if (ep + 1) % args.eval_interval == 0:
            pbar.set_description("Evaluating...")
            eval_reward = evaluate_agent(q, OBELIX, args.difficulty, args.wall_obstacles,
                                       n_episodes=10, max_steps=args.max_steps,
                                       seed=10000 + ep)
            pbar.set_description("D3RQN-PER-Noisy")

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                torch.save(q.state_dict(), args.out)
                tqdm.write(f"\n[Ep {ep+1}/{args.episodes}] "
                          f"Train: {ep_ret:8.1f} | "
                          f"Eval: {eval_reward:8.1f} | "
                          f"Best: {best_eval_reward:8.1f} | "
                          f"*** NEW BEST - SAVED ***")
            else:
                tqdm.write(f"\n[Ep {ep+1}/{args.episodes}] "
                          f"Train: {ep_ret:8.1f} | "
                          f"Eval: {eval_reward:8.1f} | "
                          f"Best: {best_eval_reward:8.1f}")
        else:
            pbar.set_postfix({
                'reward': f'{ep_ret:.1f}',
                'avg': f'{np.mean(recent_rewards):.1f}',
                'buffer': len(replay),
                'beta': f'{beta_by_step(steps):.3f}'
            })

    pbar.close()
    print(f"Training complete. Best model (eval={best_eval_reward:.1f}) saved at: {args.out}")


if __name__ == "__main__":
    main()
