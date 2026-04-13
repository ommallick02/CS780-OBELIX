"""
Training script for D3QN (Double Dueling DQN) with Noisy Networks.
No Prioritized Experience Replay - uses uniform sampling.

python train_d3qn_noisy.py --obelix_py ../../../obelix.py --wall_obstacles
"""

import argparse
import copy
import os
from typing import List, Tuple, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import deque

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = len(ACTIONS)


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


class NoisyDuelingDQN(nn.Module):
    """
    Dueling DQN with Noisy Networks.
    Separates value and advantage streams.
    """

    def __init__(self, input_dim: int = 18, hidden_dim: int = 128):
        super().__init__()

        # Shared feature layer
        self.feature = nn.Sequential(
            NoisyLinear(input_dim, hidden_dim),
            nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, N_ACTIONS)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)

        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine: Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values

    def reset_noise(self):
        """Reset noise in all noisy layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class ReplayBuffer:
    """Standard experience replay buffer (uniform sampling)."""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])

        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)


class D3QNNoisyAgent:
    """
    D3QN (Double Dueling DQN) with Noisy Networks.
    Standard uniform replay, no prioritization.
    """

    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 128,
        lr: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update: int = 1000
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        # Networks
        self.q_network = NoisyDuelingDQN(input_dim, hidden_dim).to(self.device)
        self.target_network = NoisyDuelingDQN(input_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.update_count = 0
        self.episode_count = 0
        self.total_steps = 0

    def select_action(self, obs: np.ndarray) -> int:
        """Select action using noisy network."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(obs_tensor)

        return int(q_values.argmax(dim=1).item())

    def reset_noise(self):
        """Reset noise in all layers."""
        self.q_network.reset_noise()
        self.target_network.reset_noise()

    def update(self):
        """Update using Double Dueling DQN (uniform sampling)."""
        if len(self.replay_buffer) < self.batch_size * 10:
            return {}

        # Sample uniformly
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q-values
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # DDQN target
        with torch.no_grad():
            # Select action using online network
            next_q_online = self.q_network(next_states)
            next_actions = next_q_online.argmax(dim=1)

            # Evaluate using target network
            next_q_target = self.target_network(next_states)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            targets = rewards + self.gamma * next_q * (1 - dones)

        # Loss
        loss = nn.functional.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Reset noise
        self.reset_noise()

        self.update_count += 1

        # Update target network
        if self.update_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {'loss': loss.item()}

    def save(self, path: str):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
        }, path)
        print(f"Saved D3QN+Noisy to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.episode_count = checkpoint['episode_count']
        self.total_steps = checkpoint['total_steps']


def evaluate_agent(agent, OBELIX, difficulty, wall_obstacles,
                   n_episodes=10, max_steps=1000, seed=99999):
    """Evaluate agent."""
    agent.q_network.eval()
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
        total_reward = 0.0
        done = False

        for _ in range(max_steps):
            action_idx = agent.select_action(obs)
            action = ACTIONS[action_idx]

            obs, reward, done = env.step(action, render=False)
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)

    agent.q_network.train()
    return np.mean(rewards)


def train(
    obelix_py_path: str,
    episodes: int = 5000,
    max_steps: int = 2000,
    difficulty: int = 3,
    wall_obstacles: bool = False,
    save_path: str = "d3qn_noisy_agent.pt",
    eval_interval: int = 50
):
    """Train D3QN with Noisy Networks (no PER)."""

    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OBELIX = mod.OBELIX

    agent = D3QNNoisyAgent()

    print(f"\n{'='*60}")
    print(f"Training D3QN + Noisy Networks (No PER)")
    print(f"{'='*60}")
    print(f"Episodes: {episodes}")
    print(f"Difficulty: {difficulty}")
    print(f"Device: {agent.device}")
    print(f"{'='*60}\n")

    best_eval_reward = float('-inf')
    recent_rewards = []

    with tqdm(total=episodes, desc="Training", unit="ep") as pbar:
        for episode in range(episodes):
            env = OBELIX(
                scaling_factor=5,
                arena_size=500,
                max_steps=max_steps,
                wall_obstacles=wall_obstacles,
                difficulty=difficulty,
                box_speed=2,
                seed=episode
            )

            obs = env.reset(seed=episode)
            rng = np.random.default_rng(episode)
            episode_reward = 0.0
            done = False

            agent.reset_noise()

            for step in range(max_steps):
                action_idx = agent.select_action(obs)
                action = ACTIONS[action_idx]

                next_obs, reward, done = env.step(action, render=False)
                episode_reward += reward

                agent.replay_buffer.push(obs, action_idx, reward, next_obs, done)
                obs = next_obs
                agent.total_steps += 1

                if agent.total_steps % 4 == 0:
                    metrics = agent.update()

                if done:
                    break

            agent.episode_count += 1
            recent_rewards.append(episode_reward)
            if len(recent_rewards) > 100:
                recent_rewards.pop(0)

            pbar.update(1)

            # Evaluation and save best model
            if (episode + 1) % eval_interval == 0:
                pbar.set_description("Evaluating...")
                eval_reward = evaluate_agent(agent, OBELIX, difficulty, wall_obstacles,
                                           n_episodes=10, max_steps=max_steps,
                                           seed=10000 + episode)
                pbar.set_description("Training")

                # Save best model immediately when new best is found (overwrite previous)
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    agent.save(save_path)
                    tqdm.write(f"\n[Ep {episode+1}/{episodes}] "
                              f"Train: {episode_reward:8.1f} | "
                              f"Eval: {eval_reward:8.1f} | "
                              f"Best: {best_eval_reward:8.1f} | "
                              f"*** NEW BEST - SAVED ***")
                else:
                    tqdm.write(f"\n[Ep {episode+1}/{episodes}] "
                              f"Train: {episode_reward:8.1f} | "
                              f"Eval: {eval_reward:8.1f} | "
                              f"Best: {best_eval_reward:8.1f}")
            else:
                pbar.set_postfix({
                    'reward': f'{episode_reward:.1f}',
                    'avg': f'{np.mean(recent_rewards):.1f}',
                    'buffer': len(agent.replay_buffer)
                })

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Final best model (eval={best_eval_reward:.1f}): {save_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obelix_py", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--save_path", type=str, default="d3qn_noisy_agent.pt")

    args = parser.parse_args()

    train(
        obelix_py_path=args.obelix_py,
        episodes=args.episodes,
        max_steps=args.max_steps,
        difficulty=args.difficulty,
        wall_obstacles=args.wall_obstacles,
        save_path=args.save_path
    )
