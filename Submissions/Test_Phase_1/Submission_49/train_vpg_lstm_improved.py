"""
VPG+LSTM with Curriculum Learning, Reward Shaping, and GAE (λ-return).
- Curriculum: difficulty increases from 1 to 3.
- Reward shaping: potential-based using distances.
- GAE: Generalized Advantage Estimation (λ=0.95).

Usage:
python train_vpg_lstm_improved.py --obelix_py ../../../obelix.py --wall_obstacles
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


# ---------- Networks ----------
class ActorLSTM(nn.Module):
    def __init__(self, input_dim: int = 18, enc_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, enc_dim), nn.ReLU())
        self.lstm = nn.LSTM(enc_dim, hidden_dim, batch_first=False)
        self.policy_head = nn.Linear(hidden_dim, N_ACTIONS)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]):
        enc = self.encoder(x)
        out, hidden = self.lstm(enc, hidden)
        probs = torch.softmax(self.policy_head(out), dim=-1)
        return probs, hidden

    def init_hidden(self, device: torch.device):
        z = torch.zeros(1, 1, self.hidden_dim, device=device)
        return (z, z.clone())


class CriticLSTM(nn.Module):
    def __init__(self, input_dim: int = 18, enc_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, enc_dim), nn.ReLU())
        self.lstm = nn.LSTM(enc_dim, hidden_dim, batch_first=False)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]):
        enc = self.encoder(x)
        out, hidden = self.lstm(enc, hidden)
        values = self.value_head(out).squeeze(-1)
        return values, hidden

    def init_hidden(self, device: torch.device):
        z = torch.zeros(1, 1, self.hidden_dim, device=device)
        return (z, z.clone())


# ---------- Reward Shaping Wrapper ----------
class RewardShapingWrapper:
    def __init__(self, env, gamma=0.99, scale=0.01):
        self.env = env
        self.gamma = gamma
        self.scale = scale
        self.prev_box_dist = None
        self.prev_wedge_dist = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_box_dist = self._get_box_distance(obs)
        self.prev_wedge_dist = self._get_wedge_distance(obs)
        return obs

    def step(self, action, render=False):
        obs, reward, done = self.env.step(action, render)
        curr_box = self._get_box_distance(obs)
        curr_wedge = self._get_wedge_distance(obs)
        shaping = 0.0
        if self.prev_box_dist is not None:
            shaping += self.scale * (self.prev_box_dist - curr_box)
        if self.prev_wedge_dist is not None:
            shaping += self.scale * (self.prev_wedge_dist - curr_wedge)
        shaped_reward = reward + shaping
        self.prev_box_dist = curr_box
        self.prev_wedge_dist = curr_wedge
        return obs, shaped_reward, done

    def _get_box_distance(self, obs):
        return obs[10] if len(obs) > 10 else 1000.0

    def _get_wedge_distance(self, obs):
        return obs[11] if len(obs) > 11 else 1000.0


# ---------- Curriculum Wrapper ----------
class CurriculumWrapper:
    def __init__(self, env_class, start_difficulty=1, end_difficulty=3,
                 total_episodes=5000, **env_kwargs):
        self.env_class = env_class
        self.base_env_kwargs = env_kwargs
        self.start_diff = start_difficulty
        self.end_diff = end_difficulty
        self.total_episodes = total_episodes
        self.current_difficulty = start_difficulty

    def _update_difficulty(self, episode):
        progress = min(1.0, episode / self.total_episodes)
        self.current_difficulty = self.start_diff + int(progress * (self.end_diff - self.start_diff))

    def get_env(self, episode):
        self._update_difficulty(episode)
        kwargs = self.base_env_kwargs.copy()
        kwargs['difficulty'] = self.current_difficulty
        return self.env_class(**kwargs)


# ---------- VPG+LSTM Agent with GAE ----------
class VPGLSTMAgent:
    def __init__(
        self,
        input_dim: int = 18,
        enc_dim: int = 64,
        hidden_dim: int = 128,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        lam: float = 0.95,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.lam = lam
        self.actor = ActorLSTM(input_dim, enc_dim, hidden_dim).to(self.device)
        self.critic = CriticLSTM(input_dim, enc_dim, hidden_dim).to(self.device)
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.episode_count = 0
        self.total_steps = 0

    def collect_episode(self, env, max_steps: int, seed: int) -> Tuple[List, float, bool]:
        obs = env.reset(seed=seed)
        actor_hidden = self.actor.init_hidden(self.device)
        traj = []
        ep_ret = 0.0
        done = False
        for _ in range(max_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                probs, actor_hidden = self.actor(obs_t, actor_hidden)
                dist = torch.distributions.Categorical(probs.squeeze())
                action = dist.sample().item()
            next_obs, reward, done = env.step(ACTIONS[action], render=False)
            ep_ret += reward
            traj.append((obs.copy(), action, float(reward)))
            obs = next_obs
            self.total_steps += 1
            if done:
                break
        return traj, ep_ret, done

    def compute_gae(self, rewards: List[float], values: List[float], done: bool) -> Tuple[np.ndarray, np.ndarray]:
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_val = 0.0 if done else values[t]
                delta = rewards[t] + self.gamma * next_val - values[t]
            else:
                delta = rewards[t] + self.gamma * values[t+1] - values[t]
            gae = delta + self.gamma * self.lam * gae
            advantages[t] = gae
        returns = advantages + np.array(values)
        return advantages, returns

    def update(self, traj: List[Tuple], done: bool) -> dict:
        obs_seq = np.array([t[0] for t in traj], dtype=np.float32)
        action_seq = [t[1] for t in traj]
        reward_seq = [t[2] for t in traj]
        T = len(traj)

        x_t = torch.FloatTensor(obs_seq).unsqueeze(1).to(self.device)  # (T, 1, 18)

        # Get values from critic
        with torch.no_grad():
            critic_hidden = self.critic.init_hidden(self.device)
            values, _ = self.critic(x_t, critic_hidden)  # (T,)
            values = values.squeeze(1).cpu().numpy()

        advantages, returns = self.compute_gae(reward_seq, values, done)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # Critic update
        critic_hidden = self.critic.init_hidden(self.device)
        values_pred, _ = self.critic(x_t, critic_hidden)
        values_pred = values_pred.squeeze(1)
        critic_loss = nn.functional.mse_loss(values_pred, returns_t)

        self.opt_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_critic.step()

        # Actor update
        actor_hidden = self.actor.init_hidden(self.device)
        probs, _ = self.actor(x_t, actor_hidden)
        probs = probs.squeeze(1)  # (T, N_ACTIONS)
        actions_t = torch.LongTensor(action_seq).to(self.device)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions_t)
        actor_loss = -(log_probs * advantages_t).mean()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt_actor.step()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'avg_return': returns_t.mean().item()
        }


# ---------- Evaluation ----------
def evaluate_agent(agent, OBELIX, difficulty, wall_obstacles,
                   n_episodes=10, max_steps=2000, seed=99999):
    agent.actor.eval()
    rewards = []
    for i in range(n_episodes):
        env = OBELIX(
            scaling_factor=5, arena_size=500, max_steps=max_steps,
            wall_obstacles=wall_obstacles, difficulty=difficulty,
            box_speed=2, seed=seed + i
        )
        obs = env.reset(seed=seed + i)
        hidden = agent.actor.init_hidden(agent.device)
        total = 0.0
        done = False
        for _ in range(max_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                probs, hidden = agent.actor(obs_t, hidden)
                action = int(probs.squeeze().argmax().item())
            obs, reward, done = env.step(ACTIONS[action], render=False)
            total += reward
            if done:
                break
        rewards.append(total)
    agent.actor.train()
    return np.mean(rewards), np.std(rewards)


# ---------- Main training ----------
def import_obelix(path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def train(
    obelix_py: str,
    episodes: int = 5000,
    max_steps: int = 2000,
    difficulty_start: int = 1,
    difficulty_end: int = 3,
    wall_obstacles: bool = False,
    save_path: str = "vpg_lstm_improved.pt",
    eval_interval: int = 50,
    gamma: float = 0.99,
    lam: float = 0.95,
    lr_actor: float = 3e-4,
    lr_critic: float = 1e-3,
):
    OBELIX = import_obelix(obelix_py)
    agent = VPGLSTMAgent(gamma=gamma, lam=lam, lr_actor=lr_actor, lr_critic=lr_critic)

    base_env_kwargs = {
        'scaling_factor': 5, 'arena_size': 500, 'max_steps': max_steps,
        'wall_obstacles': wall_obstacles, 'box_speed': 2,
        'difficulty': difficulty_start
    }
    curriculum = CurriculumWrapper(OBELIX,
                                   start_difficulty=difficulty_start,
                                   end_difficulty=difficulty_end,
                                   total_episodes=episodes,
                                   **base_env_kwargs)

    print(f"\n{'='*60}")
    print(f"Training VPG+LSTM + Curriculum + Reward Shaping + GAE(λ={lam})")
    print(f"Difficulty: {difficulty_start} -> {difficulty_end}")
    print(f"γ={gamma}, λ={lam}")
    print(f"Device: {agent.device}")
    print(f"{'='*60}\n")

    best_eval_reward = float('-inf')
    best_weights = copy.deepcopy(agent.actor.state_dict())
    recent_rewards = deque(maxlen=100)

    for ep in tqdm(range(episodes), desc="Training"):
        raw_env = curriculum.get_env(ep)
        shaped_env = RewardShapingWrapper(raw_env, gamma=gamma)
        traj, ep_ret, done = agent.collect_episode(shaped_env, max_steps, seed=ep)
        metrics = agent.update(traj, done)

        agent.episode_count += 1
        recent_rewards.append(ep_ret)

        if (ep + 1) % eval_interval == 0:
            eval_mean, eval_std = evaluate_agent(
                agent, OBELIX, difficulty=curriculum.current_difficulty,
                wall_obstacles=wall_obstacles,
                n_episodes=10, max_steps=max_steps, seed=10000 + ep
            )
            avg_recent = np.mean(recent_rewards)
            if eval_mean > best_eval_reward:
                best_eval_reward = eval_mean
                best_weights = copy.deepcopy(agent.actor.state_dict())
                tqdm.write(f"\n[Ep {ep+1}] Train: {ep_ret:.1f} (avg recent: {avg_recent:.1f}) | "
                           f"Eval: {eval_mean:.1f}±{eval_std:.1f} | Best: {best_eval_reward:.1f} | SAVED")
            else:
                tqdm.write(f"\n[Ep {ep+1}] Train: {ep_ret:.1f} (avg recent: {avg_recent:.1f}) | "
                           f"Eval: {eval_mean:.1f}±{eval_std:.1f} | Best: {best_eval_reward:.1f}")

    torch.save(best_weights, save_path)
    print(f"\nTraining Complete! Best eval: {best_eval_reward:.1f}")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obelix_py", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--difficulty_start", type=int, default=1)
    parser.add_argument("--difficulty_end", type=int, default=3)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--save_path", type=str, default="vpg_lstm_improved.pt")
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--lr_actor", type=float, default=3e-4)
    parser.add_argument("--lr_critic", type=float, default=1e-3)
    args = parser.parse_args()
    train(**vars(args))