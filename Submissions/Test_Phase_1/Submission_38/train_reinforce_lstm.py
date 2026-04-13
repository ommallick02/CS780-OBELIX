"""Training script for REINFORCE with LSTM for OBELIX.

Replaces frame stacking with a recurrent hidden state so the agent can
maintain its own temporal context across steps within an episode.

python train_reinforce_lstm.py --obelix_py ../../../obelix.py --wall_obstacles

                    ALGORITHM: REINFORCE + LSTM

Standard REINFORCE (Monte Carlo Policy Gradient) but the policy network
is an LSTM instead of a flat MLP:

    obs_t  →  Linear encoder  →  LSTM  →  policy_head  →  softmax  →  π(a|h_t)

The LSTM hidden state h_t carries information across timesteps, giving the
agent memory of recent observations — essential for the POMDP setting of
OBELIX where a single 18-bit frame is often ambiguous.

Training (BPTT over the full episode):
  1. Collect episode: store (obs, action, reward) at each step.
  2. Compute discounted Monte Carlo returns G_t.
  3. Re-run the full obs sequence through the LSTM in one vectorised pass
     to obtain log π(a_t | h_t) with gradients intact.
  4. Loss = -Σ_t log π(a_t | h_t) · Ĝ_t   (normalised returns)

Key difference from MLP REINFORCE:
  The update re-runs the trajectory through the network (BPTT) rather than
  using stored log_probs, so gradients flow back through the recurrent
  connections correctly.
"""

import argparse
import copy
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)


# ── Network ───────────────────────────────────────────────────────────────────
class PolicyLSTM(nn.Module):
    """
    obs (18,)  →  encoder (Linear → ReLU)  →  LSTM  →  policy_head  →  softmax

    The LSTM processes one step at a time during rollout (hidden state
    threaded through manually) and the full sequence at once during the
    BPTT update pass.
    """

    def __init__(self, input_dim: int = 18, enc_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder    = nn.Sequential(nn.Linear(input_dim, enc_dim), nn.ReLU())
        self.lstm       = nn.LSTM(enc_dim, hidden_dim, batch_first=False)
        self.policy_head = nn.Linear(hidden_dim, N_ACTIONS)

    def forward(
        self,
        x: torch.Tensor,                              # (seq_len, batch, input_dim)
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        enc = self.encoder(x)                         # (seq_len, batch, enc_dim)
        out, hidden = self.lstm(enc, hidden)          # out: (seq_len, batch, hidden_dim)
        logits = self.policy_head(out)                # (seq_len, batch, N_ACTIONS)
        probs  = torch.softmax(logits, dim=-1)
        return probs, hidden

    def init_hidden(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(1, 1, self.hidden_dim, device=device)
        return (z, z.clone())


# ── Agent ─────────────────────────────────────────────────────────────────────
class REINFORCELSTMAgent:
    def __init__(
        self,
        input_dim:  int   = 18,
        enc_dim:    int   = 64,
        hidden_dim: int   = 128,
        lr:         float = 3e-4,
        gamma:      float = 0.99,
    ):
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma      = gamma
        self.policy     = PolicyLSTM(input_dim, enc_dim, hidden_dim).to(self.device)
        self.optimizer  = optim.Adam(self.policy.parameters(), lr=lr)
        self.episode_count = 0
        self.total_steps   = 0

    # ------------------------------------------------------------------
    def collect_episode(self, env, max_steps: int) -> Tuple[List, float]:
        """
        Run one episode, threading hidden state through the LSTM step by step.
        Returns trajectory = [(obs, action, reward), ...] and episode return.
        """
        obs      = env.reset(seed=self.episode_count)
        hidden   = self.policy.init_hidden(self.device)
        traj     = []
        ep_ret   = 0.0
        done     = False

        for _ in range(max_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(self.device)
            # (1, 1, 18): seq_len=1, batch=1

            with torch.no_grad():
                probs, hidden = self.policy(obs_t, hidden)
                dist   = torch.distributions.Categorical(probs.squeeze())
                action = dist.sample().item()

            next_obs, reward, done = env.step(ACTIONS[action], render=False)
            ep_ret += reward
            traj.append((obs.copy(), action, float(reward)))
            obs = next_obs
            self.total_steps += 1

            if done:
                break

        return traj, ep_ret

    # ------------------------------------------------------------------
    def update(self, traj: List[Tuple]) -> dict:
        """
        BPTT update: re-run the full trajectory through the LSTM to obtain
        differentiable log-probs, then apply REINFORCE loss.
        """
        obs_seq     = np.array([t[0] for t in traj], dtype=np.float32)
        action_seq  = [t[1] for t in traj]
        reward_seq  = [t[2] for t in traj]

        # ── Monte Carlo returns ──────────────────────────────────────
        returns = []
        G = 0.0
        for r in reversed(reward_seq):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns_t = torch.FloatTensor(returns).to(self.device)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # ── BPTT: run full sequence through LSTM in one pass ─────────
        # obs_seq shape: (T, 18) → (T, 1, 18) for LSTM (seq_len, batch, features)
        x_t      = torch.FloatTensor(obs_seq).unsqueeze(1).to(self.device)
        hidden   = self.policy.init_hidden(self.device)
        probs, _ = self.policy(x_t, hidden)           # (T, 1, N_ACTIONS)
        probs    = probs.squeeze(1)                    # (T, N_ACTIONS)

        actions_t  = torch.LongTensor(action_seq).to(self.device)
        dist       = torch.distributions.Categorical(probs)
        log_probs  = dist.log_prob(actions_t)          # (T,)

        loss = -(log_probs * returns_t).sum()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return {"loss": loss.item(), "avg_return": float(returns_t.mean())}


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_agent(agent, OBELIX, difficulty, wall_obstacles,
                   n_episodes: int = 10, max_steps: int = 2000, seed: int = 99999):
    agent.policy.eval()
    rewards = []

    for i in range(n_episodes):
        env = OBELIX(
            scaling_factor=5, arena_size=500, max_steps=max_steps,
            wall_obstacles=wall_obstacles, difficulty=difficulty,
            box_speed=2, seed=seed + i,
        )
        obs    = env.reset(seed=seed + i)
        hidden = agent.policy.init_hidden(agent.device)
        total  = 0.0
        done   = False

        for _ in range(max_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                probs, hidden = agent.policy(obs_t, hidden)
                action = int(probs.squeeze().argmax().item())   # greedy

            obs, reward, done = env.step(ACTIONS[action], render=False)
            total += reward
            if done:
                break

        rewards.append(total)

    agent.policy.train()
    return float(np.mean(rewards)), float(np.std(rewards))


# ── Training loop ─────────────────────────────────────────────────────────────
def import_obelix(path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def train(
    obelix_py:     str,
    episodes:      int   = 5000,
    max_steps:     int   = 2000,
    difficulty:    int   = 3,
    wall_obstacles: bool = False,
    save_path:     str   = "reinforce_lstm.pt",
    eval_interval: int   = 50,
):
    OBELIX = import_obelix(obelix_py)
    agent  = REINFORCELSTMAgent()

    print(f"\n{'='*60}")
    print(f"Training REINFORCE + LSTM")
    print(f"{'='*60}")
    print(f"Episodes:   {episodes}")
    print(f"Difficulty: {difficulty}")
    print(f"Device:     {agent.device}")
    print(f"{'='*60}\n")

    best_eval_reward = float("-inf")
    best_weights     = copy.deepcopy(agent.policy.state_dict())  # in-memory only
    recent_rewards   = deque(maxlen=100)

    for ep in tqdm(range(episodes), desc="Training"):
        env = OBELIX(
            scaling_factor=5, arena_size=500, max_steps=max_steps,
            wall_obstacles=wall_obstacles, difficulty=difficulty,
            box_speed=2, seed=ep,
        )

        traj, ep_ret = agent.collect_episode(env, max_steps)
        agent.update(traj)
        agent.episode_count += 1
        recent_rewards.append(ep_ret)

        if (ep + 1) % eval_interval == 0:
            eval_mean, eval_std = evaluate_agent(
                agent, OBELIX, difficulty, wall_obstacles,
                n_episodes=10, max_steps=max_steps, seed=10000 + ep,
            )
            if eval_mean > best_eval_reward:
                best_eval_reward = eval_mean
                best_weights     = copy.deepcopy(agent.policy.state_dict())
                tqdm.write(
                    f"\n[Ep {ep+1}] Train: {ep_ret:.1f} | "
                    f"Eval: {eval_mean:.1f}±{eval_std:.1f} | "
                    f"Best: {best_eval_reward:.1f} | weights updated in memory"
                )
            else:
                tqdm.write(
                    f"\n[Ep {ep+1}] Train: {ep_ret:.1f} | "
                    f"Eval: {eval_mean:.1f}±{eval_std:.1f} | "
                    f"Best: {best_eval_reward:.1f}"
                )

    # Single save at end — best weights seen during training
    torch.save(best_weights, save_path)
    print(f"\nTraining Complete! Best eval: {best_eval_reward:.1f}")
    print(f"Saved best model to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obelix_py",      type=str,  required=True)
    parser.add_argument("--episodes",       type=int,  default=5000)
    parser.add_argument("--max_steps",      type=int,  default=2000)
    parser.add_argument("--difficulty",     type=int,  default=3)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--save_path",      type=str,  default="reinforce_lstm.pt")
    parser.add_argument("--eval_interval",  type=int,  default=50)
    args = parser.parse_args()
    train(**vars(args))
