"""Training script for VPG (Vanilla Policy Gradient) with LSTM for OBELIX.

python train_vpg_lstm.py --obelix_py ../../../obelix.py --wall_obstacles

                    ALGORITHM: VPG + LSTM

VPG (REINFORCE with a learned value baseline) where both actor and critic
are LSTM networks sharing the same encoder architecture:

    Actor:  obs_t → encoder → LSTM_actor  → policy_head → softmax → π(a|h_t)
    Critic: obs_t → encoder → LSTM_critic → value_head  → V(s_t)

The critic's value estimate V(s_t) is used as a baseline to reduce the
variance of the policy gradient:

    advantage_t = G_t - V(s_t)
    actor_loss  = -Σ_t log π(a_t | h_t) · advantage_t

The critic is trained separately to minimise MSE between V(s_t) and G_t.

Both networks use BPTT over the full episode trajectory:
  - Full obs sequence is run through the LSTM in one vectorised pass
  - Gradients flow back through recurrent connections correctly

The actor and critic have independent LSTM hidden states during rollout,
which avoids gradient interference and allows different learning rates.
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


# ── Networks ──────────────────────────────────────────────────────────────────
class ActorLSTM(nn.Module):
    """
    obs (18,)  →  encoder  →  LSTM  →  policy_head  →  softmax
    """

    def __init__(self, input_dim: int = 18, enc_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.encoder     = nn.Sequential(nn.Linear(input_dim, enc_dim), nn.ReLU())
        self.lstm        = nn.LSTM(enc_dim, hidden_dim, batch_first=False)
        self.policy_head = nn.Linear(hidden_dim, N_ACTIONS)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        enc            = self.encoder(x)
        out, hidden    = self.lstm(enc, hidden)
        probs          = torch.softmax(self.policy_head(out), dim=-1)
        return probs, hidden

    def init_hidden(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(1, 1, self.hidden_dim, device=device)
        return (z, z.clone())


class CriticLSTM(nn.Module):
    """
    obs (18,)  →  encoder  →  LSTM  →  value_head  →  V(s)
    """

    def __init__(self, input_dim: int = 18, enc_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder    = nn.Sequential(nn.Linear(input_dim, enc_dim), nn.ReLU())
        self.lstm       = nn.LSTM(enc_dim, hidden_dim, batch_first=False)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        enc         = self.encoder(x)
        out, hidden = self.lstm(enc, hidden)
        values      = self.value_head(out).squeeze(-1)   # (seq_len, batch)
        return values, hidden

    def init_hidden(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(1, 1, self.hidden_dim, device=device)
        return (z, z.clone())


# ── Agent ─────────────────────────────────────────────────────────────────────
class VPGLSTMAgent:
    def __init__(
        self,
        input_dim:  int   = 18,
        enc_dim:    int   = 64,
        hidden_dim: int   = 128,
        lr_actor:   float = 3e-4,
        lr_critic:  float = 1e-3,
        gamma:      float = 0.99,
    ):
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma     = gamma
        self.actor     = ActorLSTM(input_dim, enc_dim, hidden_dim).to(self.device)
        self.critic    = CriticLSTM(input_dim, enc_dim, hidden_dim).to(self.device)
        self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.episode_count = 0
        self.total_steps   = 0

    # ------------------------------------------------------------------
    def collect_episode(self, env, max_steps: int) -> Tuple[List, float]:
        """
        Run one episode threading actor hidden state through LSTM step by step.
        Returns trajectory = [(obs, action, reward), ...] and episode return.
        """
        obs            = env.reset(seed=self.episode_count)
        actor_hidden   = self.actor.init_hidden(self.device)
        traj           = []
        ep_ret         = 0.0
        done           = False

        for _ in range(max_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(self.device)

            with torch.no_grad():
                probs, actor_hidden = self.actor(obs_t, actor_hidden)
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
        BPTT update for both actor and critic.
        Runs full episode sequence through each LSTM in one vectorised pass.
        """
        obs_seq    = np.array([t[0] for t in traj], dtype=np.float32)
        action_seq = [t[1] for t in traj]
        reward_seq = [t[2] for t in traj]

        # ── Monte Carlo returns ──────────────────────────────────────
        returns = []
        G = 0.0
        for r in reversed(reward_seq):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns_t = torch.FloatTensor(returns).to(self.device)   # (T,)

        # x shape: (T, 1, 18) for LSTM (seq_len, batch, features)
        x_t = torch.FloatTensor(obs_seq).unsqueeze(1).to(self.device)

        # ── Update Critic ────────────────────────────────────────────
        critic_hidden = self.critic.init_hidden(self.device)
        values, _     = self.critic(x_t, critic_hidden)   # (T, 1)
        values        = values.squeeze(1)                  # (T,)
        critic_loss   = nn.functional.mse_loss(values, returns_t)

        self.opt_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_critic.step()

        # ── Compute advantages (no gradient needed through critic here) ──
        with torch.no_grad():
            critic_hidden2 = self.critic.init_hidden(self.device)
            values_det, _  = self.critic(x_t, critic_hidden2)
            values_det     = values_det.squeeze(1)
            advantages     = returns_t - values_det
            advantages     = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── Update Actor (BPTT) ──────────────────────────────────────
        actor_hidden     = self.actor.init_hidden(self.device)
        probs, _         = self.actor(x_t, actor_hidden)   # (T, 1, N_ACTIONS)
        probs            = probs.squeeze(1)                 # (T, N_ACTIONS)
        actions_t        = torch.LongTensor(action_seq).to(self.device)
        dist             = torch.distributions.Categorical(probs)
        log_probs        = dist.log_prob(actions_t)        # (T,)
        actor_loss       = -(log_probs * advantages).mean()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt_actor.step()

        return {
            "actor_loss":  actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "avg_return":  float(returns_t.mean()),
        }


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_agent(agent, OBELIX, difficulty, wall_obstacles,
                   n_episodes: int = 10, max_steps: int = 2000, seed: int = 99999):
    agent.actor.eval()
    rewards = []

    for i in range(n_episodes):
        env = OBELIX(
            scaling_factor=5, arena_size=500, max_steps=max_steps,
            wall_obstacles=wall_obstacles, difficulty=difficulty,
            box_speed=2, seed=seed + i,
        )
        obs    = env.reset(seed=seed + i)
        hidden = agent.actor.init_hidden(agent.device)
        total  = 0.0
        done   = False

        for _ in range(max_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                probs, hidden = agent.actor(obs_t, hidden)
                action = int(probs.squeeze().argmax().item())   # greedy

            obs, reward, done = env.step(ACTIONS[action], render=False)
            total += reward
            if done:
                break

        rewards.append(total)

    agent.actor.train()
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
    save_path:     str   = "vpg_lstm.pt",
    eval_interval: int   = 50,
):
    OBELIX = import_obelix(obelix_py)
    agent  = VPGLSTMAgent()

    print(f"\n{'='*60}")
    print(f"Training VPG + LSTM")
    print(f"{'='*60}")
    print(f"Episodes:   {episodes}")
    print(f"Difficulty: {difficulty}")
    print(f"Device:     {agent.device}")
    print(f"{'='*60}\n")

    best_eval_reward = float("-inf")
    best_weights     = copy.deepcopy(agent.actor.state_dict())  # in-memory only
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
                best_weights     = copy.deepcopy(agent.actor.state_dict())
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

    # Single save at end — best actor weights seen during training
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
    parser.add_argument("--save_path",      type=str,  default="vpg_lstm.pt")
    parser.add_argument("--eval_interval",  type=int,  default=50)
    args = parser.parse_args()
    train(**vars(args))
