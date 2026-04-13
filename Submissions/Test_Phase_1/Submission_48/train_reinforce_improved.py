"""
REINFORCE(λ) with Curriculum Learning and Reward Shaping for OBELIX.
- Curriculum: difficulty increases from 1 to 3 over episodes.
- Reward shaping: potential-based using distance to nearest box and wedge.
- Eligibility traces: λ-return (REINFORCE(λ)) reduces variance.

Usage:
python train_reinforce_improved.py --obelix_py ../../../obelix.py --wall_obstacles
"""

import argparse
from collections import deque
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)


# ---------- Policy Network ----------
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int = 18, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)

    def sample_action(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = self.forward(x)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


# ---------- Reward Shaping Wrapper ----------
class RewardShapingWrapper:
    """Potential-based reward shaping using distances to box and wedge."""
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
        # Adjust indices according to your OBELIX observation layout
        return obs[10] if len(obs) > 10 else 1000.0

    def _get_wedge_distance(self, obs):
        return obs[11] if len(obs) > 11 else 1000.0


# ---------- Curriculum Wrapper ----------
class CurriculumWrapper:
    """Increases environment difficulty over episodes."""
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
        """Return a new environment with the correct difficulty for the episode."""
        self._update_difficulty(episode)
        kwargs = self.base_env_kwargs.copy()
        kwargs['difficulty'] = self.current_difficulty
        return self.env_class(**kwargs)


# ---------- REINFORCE Agent with λ-return ----------
class REINFORCELambdaAgent:
    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.lam = lam
        self.policy = PolicyNetwork(input_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.episode_count = 0
        self.total_steps = 0

    def select_action(self, obs: np.ndarray) -> Tuple[int, float]:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob = self.policy.sample_action(obs_tensor)
        return action.item(), log_prob.item()

    def compute_lambda_returns(self, rewards: List[float]) -> List[float]:
        """Compute λ-returns for a trajectory using forward view."""
        T = len(rewards)
        lambda_returns = np.zeros(T)
        # Precompute discount factors
        gamma_pows = [self.gamma ** i for i in range(T + 1)]
        for t in range(T):
            G_lambda = 0.0
            # Sum over n-step returns
            for n in range(1, T - t + 1):
                G_n = 0.0
                for i in range(n):
                    G_n += gamma_pows[i] * rewards[t + i]
                if n == T - t:
                    weight = self.lam ** (n - 1)
                else:
                    weight = (1 - self.lam) * (self.lam ** (n - 1))
                G_lambda += weight * G_n
            lambda_returns[t] = G_lambda
        return lambda_returns.tolist()

    def update(self, trajectory: List[Tuple]) -> dict:
        """
        trajectory: list of (obs, action, log_prob, reward)
        """
        rewards = [t[3] for t in trajectory]
        lambda_returns = self.compute_lambda_returns(rewards)
        lambda_returns = torch.FloatTensor(lambda_returns).to(self.device)
        # Normalize for stability
        if lambda_returns.std() > 1e-8:
            lambda_returns = (lambda_returns - lambda_returns.mean()) / lambda_returns.std()

        policy_loss = []
        for i, (obs, action, _, _) in enumerate(trajectory):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action_t = torch.LongTensor([action]).to(self.device)
            probs = self.policy(obs_t)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(action_t)
            loss = -log_prob * lambda_returns[i]
            policy_loss.append(loss)

        total_loss = torch.stack(policy_loss).sum()
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return {'loss': total_loss.item(), 'avg_return': lambda_returns.mean().item()}


# ---------- Evaluation ----------
def evaluate_agent(agent, OBELIX, difficulty, wall_obstacles,
                   n_episodes=10, max_steps=1000, seed=99999):
    agent.policy.eval()
    rewards = []
    for i in range(n_episodes):
        env = OBELIX(
            scaling_factor=5, arena_size=500, max_steps=max_steps,
            wall_obstacles=wall_obstacles, difficulty=difficulty,
            box_speed=2, seed=seed + i
        )
        obs = env.reset(seed=seed + i)
        total = 0.0
        done = False
        for _ in range(max_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action = agent.policy.sample_action(obs_t)[0].item()
            obs, reward, done = env.step(ACTIONS[action], render=False)
            total += reward
            if done:
                break
        rewards.append(total)
    agent.policy.train()
    return np.mean(rewards), np.std(rewards)


# ---------- Main training ----------
def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
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
    save_path: str = "reinforce_improved.pt",
    eval_interval: int = 50,
    gamma: float = 0.99,
    lam: float = 0.95,
    lr: float = 3e-4,
):
    OBELIX = import_obelix(obelix_py)
    agent = REINFORCELambdaAgent(gamma=gamma, lam=lam, lr=lr)

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
    print(f"Training REINFORCE(λ) with Curriculum + Reward Shaping")
    print(f"Difficulty: {difficulty_start} -> {difficulty_end} over {episodes} episodes")
    print(f"γ={gamma}, λ={lam}, lr={lr}")
    print(f"Device: {agent.device}")
    print(f"{'='*60}\n")

    best_eval_reward = float('-inf')
    recent_rewards = deque(maxlen=100)

    for ep in tqdm(range(episodes), desc="Training"):
        # Get environment with current difficulty
        raw_env = curriculum.get_env(ep)
        shaped_env = RewardShapingWrapper(raw_env, gamma=gamma)
        obs = shaped_env.reset(seed=ep)

        trajectory = []
        episode_reward = 0.0
        done = False

        for _ in range(max_steps):
            action_idx, log_prob = agent.select_action(obs)
            next_obs, shaped_reward, done = shaped_env.step(ACTIONS[action_idx], render=False)
            episode_reward += shaped_reward
            trajectory.append((obs.copy(), action_idx, log_prob, shaped_reward))
            obs = next_obs
            agent.total_steps += 1
            if done:
                break

        metrics = agent.update(trajectory)
        agent.episode_count += 1
        recent_rewards.append(episode_reward)

        if (ep + 1) % eval_interval == 0:
            eval_mean, eval_std = evaluate_agent(
                agent, OBELIX, difficulty=curriculum.current_difficulty,
                wall_obstacles=wall_obstacles,
                n_episodes=10, max_steps=max_steps, seed=10000 + ep
            )
            avg_recent = np.mean(recent_rewards)
            if eval_mean > best_eval_reward:
                best_eval_reward = eval_mean
                torch.save(agent.policy.state_dict(), save_path)
                tqdm.write(f"\n[Ep {ep+1}] Train: {episode_reward:.1f} (avg recent: {avg_recent:.1f}) | "
                           f"Eval: {eval_mean:.1f}±{eval_std:.1f} | Best: {best_eval_reward:.1f} | SAVED")
            else:
                tqdm.write(f"\n[Ep {ep+1}] Train: {episode_reward:.1f} (avg recent: {avg_recent:.1f}) | "
                           f"Eval: {eval_mean:.1f}±{eval_std:.1f} | Best: {best_eval_reward:.1f}")

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
    parser.add_argument("--save_path", type=str, default="reinforce_improved.pt")
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    train(**vars(args))