"""
Training script for D3RQN (Dueling Deep Recurrent Q-Network) with LSTM for OBELIX.

Example:
python train_d3rqn.py --obelix_py ../../../obelix.py --wall_obstacles

                    ALGORITHM: DUELING DEEP RECURRENT Q-NETWORK (D3RQN)

D3RQN combines three improvements:
1. LSTM for temporal memory (like DRQN)
2. Double DQN for reduced overestimation bias
3. Dueling architecture for better value estimation

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
- Double DQN target calculation for stable learning
- Saves best model during training (overwrites previous)
"""

from __future__ import annotations
import argparse, copy, random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


# ── Dueling LSTM Q-Network ───────────────────────────────────────────────────
class DuelingLSTMDQN(nn.Module):
    """
    Dueling DQN with LSTM for temporal memory.

    Architecture:
        LSTM → Value Stream (V(s))
             → Advantage Stream (A(s,a))
        Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
    """
    def __init__(self, in_dim: int = 18, hidden_dim: int = 64, n_actions: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Shared LSTM encoder
        self.lstm = nn.LSTM(in_dim, hidden_dim, batch_first=True)

        # Value stream: estimates state value V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Advantage stream: estimates action advantages A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None):
        """
        Forward pass.

        Args:
            x: Input observation(s), shape (batch, seq_len, in_dim) or (batch, in_dim)
            hidden: Optional (h, c) tuple of LSTM hidden states

        Returns:
            q_values: Q-values for each action, shape (batch, n_actions)
            hidden: Updated (h, c) tuple for next step
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # LSTM forward pass
        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)

        # Take last output in sequence
        features = lstm_out[:, -1, :]

        # Compute value and advantage
        value = self.value_stream(features)  # (batch, 1)
        advantages = self.advantage_stream(features)  # (batch, n_actions)

        # Combine: Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values, hidden

    def init_hidden(self, batch_size: int = 1, device: str = "cpu"):
        """Initialize LSTM hidden state."""
        h = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return (h, c)


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool


class Replay:
    """Experience replay buffer for single-step transitions."""
    def __init__(self, cap: int = 100_000):
        self.buf: Deque[Transition] = deque(maxlen=cap)

    def add(self, t: Transition):
        self.buf.append(t)

    def sample(self, batch: int):
        idx = np.random.choice(len(self.buf), size=batch, replace=False)
        items = [self.buf[i] for i in idx]
        s = np.stack([it.s for it in items]).astype(np.float32)
        a = np.array([it.a for it in items], dtype=np.int64)
        r = np.array([it.r for it in items], dtype=np.float32)
        s2 = np.stack([it.s2 for it in items]).astype(np.float32)
        d = np.array([it.done for it in items], dtype=np.float32)
        return s, a, r, s2, d

    def __len__(self): 
        return len(self.buf)


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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lstm_hidden", type=int, default=64, help="LSTM hidden dimension")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    print(f"Using device: {device}")

    OBELIX = import_obelix(args.obelix_py)

    # Networks
    q = DuelingLSTMDQN(in_dim=18, hidden_dim=args.lstm_hidden, n_actions=5).to(device)
    tgt = DuelingLSTMDQN(in_dim=18, hidden_dim=args.lstm_hidden, n_actions=5).to(device)
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt = optim.Adam(q.parameters(), lr=args.lr)
    replay = Replay(args.replay)
    steps = 0

    recent_returns = deque(maxlen=50)
    best_avg_return = float("-inf")

    def eps_by_step(t):
        if t >= args.eps_decay_steps:
            return args.eps_end
        frac = t / args.eps_decay_steps
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    pbar = tqdm(range(args.episodes), desc="D3RQN", unit="ep", dynamic_ncols=True)

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

        # Initialize LSTM hidden state for this episode
        hidden = q.init_hidden(batch_size=1, device=device)

        for _ in range(args.max_steps):
            eps = eps_by_step(steps)

            # Select action using current hidden state
            with torch.no_grad():
                s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                q_values, hidden = q(s_t, hidden)

                if np.random.rand() < eps:
                    a = np.random.randint(len(ACTIONS))
                else:
                    a = int(q_values.squeeze(0).argmax().item())

            # Detach hidden state to prevent backprop through entire episode
            hidden = (hidden[0].detach(), hidden[1].detach())

            s2, r, done = env.step(ACTIONS[a], render=False)
            ep_ret += float(r)
            replay.add(Transition(s=s, a=a, r=float(r), s2=s2, done=bool(done)))
            s = s2
            steps += 1

            if len(replay) >= max(args.warmup, args.batch):
                sb, ab, rb, s2b, db = replay.sample(args.batch)

                sb_t = torch.tensor(sb, device=device)
                ab_t = torch.tensor(ab, device=device)
                rb_t = torch.tensor(rb, device=device)
                s2b_t = torch.tensor(s2b, device=device)
                db_t = torch.tensor(db, device=device)

                # Double DQN target
                with torch.no_grad():
                    next_q, _ = q(s2b_t.unsqueeze(1))  # Online network selects action
                    next_a = torch.argmax(next_q, dim=1)

                    next_q_tgt, _ = tgt(s2b_t.unsqueeze(1))  # Target network evaluates
                    next_val = next_q_tgt.gather(1, next_a.unsqueeze(1)).squeeze(1)
                    y = rb_t + args.gamma * (1.0 - db_t) * next_val

                pred, _ = q(sb_t.unsqueeze(1))
                pred = pred.gather(1, ab_t.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, y)
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
                "avg50": f"{avg_return:>8.1f}",
                "loss": f"{last_loss:.4f}",
                "eps": f"{eps_by_step(steps):.3f}",
                "replay": f"{len(replay):>6}",
                "saved": "NEW BEST",
            })
        else:
            pbar.set_postfix({
                "return": f"{ep_ret:>8.1f}",
                "avg50": f"{avg_return:>8.1f}",
                "loss": f"{last_loss:.4f}",
                "eps": f"{eps_by_step(steps):.3f}",
                "replay": f"{len(replay):>6}",
            })

    pbar.close()
    print(f"Training complete. Best model (avg50={best_avg_return:.1f}) saved at: {args.out}")


if __name__ == "__main__":
    main()
