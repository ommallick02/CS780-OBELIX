"""PPO Training with GAE and Learning Curve Logging for OBELIX."""

import argparse
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import csv
from typing import List, Tuple

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = 5

class ActorCritic(nn.Module):
    def __init__(self, input_dim=18, hidden=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, N_ACTIONS)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        f = self.encoder(x)
        probs = torch.softmax(self.actor(f), dim=-1)
        value = self.critic(f).squeeze(-1)
        return probs, value

    def get_action_and_value(self, x, action=None):
        probs, value = self.forward(x)
        dist = torch.distributions.Categorical(probs)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

def import_obelix(p):
    import importlib.util
    spec = importlib.util.spec_from_file_location("ob", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def compute_gae(rewards, values, dones, gamma=0.99, lambda_=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0
    next_value = 0

    for t in reversed(range(len(rewards))):
        if dones[t]:
            next_value = 0
            last_gae = 0
        else:
            next_value = values[t + 1] if t + 1 < len(values) else 0

        delta = rewards[t] + gamma * next_value - values[t]
        advantages[t] = last_gae = delta + gamma * lambda_ * last_gae

    returns = advantages + values
    return advantages, returns

def save_learning_curve(log_data: List[Tuple[int, float, float]], log_dir: str, difficulty: int):
    """Save learning curve data to CSV and generate plot."""
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f"ppo_gae_learning_curve_level{difficulty}.csv")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "episode_return", "moving_avg_50"])
        writer.writerows(log_data)

    print(f"Learning curve saved to: {csv_path}")

    # Try to generate plot
    try:
        import matplotlib.pyplot as plt

        episodes = [x[0] for x in log_data]
        returns = [x[1] for x in log_data]
        moving_avgs = [x[2] for x in log_data]

        plt.figure(figsize=(10, 6))
        plt.plot(episodes, returns, alpha=0.3, label="Episode Return")
        plt.plot(episodes, moving_avgs, 'r-', linewidth=2, label="50-episode Moving Average")

        plt.xlabel("Training Episodes")
        plt.ylabel("Cumulative Reward")
        plt.title(f"PPO + GAE Learning Curve - Level {difficulty}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(log_dir, f"ppo_gae_learning_curve_level{difficulty}.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved to: {plot_path}")
        plt.close()
    except ImportError:
        print("matplotlib not available, skipping plot generation")

class PPOAgent:
    def __init__(self, input_dim=18, hidden=128, lr=3e-4, gamma=0.99, lambda_=0.95,
                 clip_eps=0.2, value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.lambda_ = lambda_
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.ac = ActorCritic(input_dim, hidden).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)

        self.reset_buffer()

    def reset_buffer(self):
        self.obs_buf = []
        self.action_buf = []
        self.log_prob_buf = []
        self.reward_buf = []
        self.value_buf = []
        self.done_buf = []

    def select_action(self, obs):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, entropy, value = self.ac.get_action_and_value(obs_t)
        return action.item(), log_prob.item(), value.item()

    def store(self, obs, action, log_prob, reward, value, done):
        self.obs_buf.append(obs)
        self.action_buf.append(action)
        self.log_prob_buf.append(log_prob)
        self.reward_buf.append(reward)
        self.value_buf.append(value)
        self.done_buf.append(done)

    def update(self, n_epochs=4, batch_size=64):
        obs = np.array(self.obs_buf)
        actions = np.array(self.action_buf)
        old_log_probs = np.array(self.log_prob_buf)
        rewards = np.array(self.reward_buf)
        values = np.array(self.value_buf)
        dones = np.array(self.done_buf)

        advantages, returns = compute_gae(rewards, values, dones, self.gamma, self.lambda_)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.FloatTensor(obs).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        dataset_size = len(obs)
        indices = np.arange(dataset_size)

        for epoch in range(n_epochs):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                mb_indices = indices[start:end]

                _, new_log_probs, entropy, new_values = self.ac.get_action_and_value(
                    obs_t[mb_indices], actions_t[mb_indices]
                )

                ratio = torch.exp(new_log_probs - old_log_probs_t[mb_indices])
                surr1 = ratio * advantages_t[mb_indices]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages_t[mb_indices]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(new_values, returns_t[mb_indices])

                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.reset_buffer()

def train(obelix_py, episodes=5000, steps_per_update=2048, difficulty=3, 
          wall_obstacles=False, save_path="ppo_agent.pt", log_dir="./logs"):
    OBELIX = import_obelix(obelix_py)
    agent = PPOAgent()

    best = float("-inf")
    recent = deque(maxlen=50)

    # Learning curve logging
    log_data: List[Tuple[int, float, float]] = []

    for ep in tqdm(range(episodes), desc="PPO"):
        env = OBELIX(scaling_factor=5, difficulty=difficulty, wall_obstacles=wall_obstacles, seed=ep)
        obs = env.reset(seed=ep)
        done = False
        ep_ret = 0

        while not done:
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, done = env.step(ACTIONS[action], render=False)
            ep_ret += reward

            agent.store(obs, action, log_prob, reward, value, done)
            obs = next_obs

            if len(agent.obs_buf) >= steps_per_update:
                agent.update()

        recent.append(ep_ret)
        if len(recent) >= 50 and (ep + 1) % 50 == 0:
            avg = sum(recent) / len(recent)
            if avg > best:
                best = avg
                torch.save(agent.ac.state_dict(), save_path)
                tqdm.write(f"[Ep {ep+1}] New best: {best:.1f}")

        moving_avg = np.mean(list(recent)) if recent else ep_ret
        log_data.append((ep + 1, ep_ret, moving_avg))

    # Save learning curve
    save_learning_curve(log_data, log_dir, difficulty)
    print(f"Best avg: {best:.1f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", required=True)
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--difficulty", type=int, default=3)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--out", default="ppo_agent.pt")
    ap.add_argument("--log_dir", type=str, default="./logs", help="Directory to save learning curves")
    args = ap.parse_args()

    train(args.obelix_py, episodes=args.episodes, difficulty=args.difficulty,
          wall_obstacles=args.wall_obstacles, save_path=args.out, log_dir=args.log_dir)
