"""
Training script for Noisy DDQN with:
- Options Framework (hand-crafted sub-policies: Find, Push, Unwedge)
- Curriculum Learning (difficulty increases over time)
- Reward Shaping (potential-based shaping)

Usage:
python train_ddqn_noisy_options_curriculum_shaping.py --obelix_py ../../../obelix.py --wall_obstacles
"""

import argparse
import copy
import os
from typing import List, Tuple, Sequence, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import deque

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = len(ACTIONS)

# ---------- Options definitions ----------
# Three high-level options:
OPTIONS = ["Find", "Push", "Unwedge"]
N_OPTIONS = len(OPTIONS)

# Option termination conditions (hand-crafted, based on object detection)
# For OBELIX, we assume the observation contains info about boxes and wedge.
# Observation indices (example, adjust if needed):
# 0-3: agent position? We'll use simple heuristics based on distance to box/wedge.
# For simplicity, we define termination when sub-goal is achieved.

def get_option_termination(obs: np.ndarray, option: str) -> bool:
    """
    Return True if the current option should terminate.
    Find terminates when a box is within pushing distance.
    Push terminates when box is near wedge.
    Unwedge terminates when wedge is freed (reward signal or specific state).
    """
    # This is a placeholder. In real implementation, parse the observation.
    # For OBELIX, you'd check distances to nearest box and wedge.
    # Here we assume obs[10] = distance to nearest box, obs[11] = distance to wedge.
    # Modify according to your actual observation layout.
    if option == "Find":
        # Terminate if a box is close (distance < 50)
        return obs[10] < 50 if len(obs) > 10 else False
    elif option == "Push":
        # Terminate if box is near wedge (distance < 50)
        return obs[11] < 50 if len(obs) > 11 else False
    elif option == "Unwedge":
        # Terminate when wedge is free (e.g., reward > threshold)
        # We'll use an external flag; for now always false (done by episode end)
        return False
    return False

def select_option_by_heuristic(obs: np.ndarray) -> str:
    """
    Meta-controller: chooses which option to run based on current observation.
    Heuristic: Find -> Push -> Unwedge in sequence.
    """
    # Simple state machine:
    # If no box is pushed yet -> Find
    # If box is pushed but wedge not freed -> Push
    # Else Unwedge
    # For simplicity, we use a global counter. Better: check environment variables.
    # Placeholder: return "Find" first, then "Push", then "Unwedge".
    # We'll store progress in the training loop.
    # This function will be overridden by a state machine in the training loop.
    return "Find"

# ---------- Noisy Networks (same as before) ----------
class NoisyLinear(nn.Module):
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

class NoisyDQN(nn.Module):
    def __init__(self, input_dim: int = 18, hidden_dim: int = 128, n_actions: int = N_ACTIONS):
        super().__init__()
        self.fc1 = NoisyLinear(input_dim, hidden_dim)
        self.fc2 = NoisyLinear(hidden_dim, hidden_dim)
        self.fc3 = NoisyLinear(hidden_dim, n_actions)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()

# ---------- Replay Buffer ----------
class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, option=None):
        self.buffer.append((state, action, reward, next_state, done, option))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones, options = zip(*samples)
        return (torch.FloatTensor(np.array(states)),
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(np.array(next_states)),
                torch.FloatTensor(dones),
                options)  # options not used for learning here

    def __len__(self):
        return len(self.buffer)

# ---------- Agent with Options ----------
class NoisyDDQNWithOptions:
    def __init__(self, input_dim=18, hidden_dim=128, lr=1e-4, gamma=0.99,
                 buffer_size=100000, batch_size=64, target_update=1000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        self.q_network = NoisyDQN(input_dim, hidden_dim, N_ACTIONS).to(self.device)
        self.target_network = NoisyDQN(input_dim, hidden_dim, N_ACTIONS).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.update_count = 0
        self.total_steps = 0

        # Option state machine
        self.current_option = "Find"
        self.option_steps = 0
        self.max_option_steps = 50  # safety

    def select_action(self, obs: np.ndarray) -> int:
        """Select action using current option's policy (same Q-network, but option meta-controller outside)."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)
        return int(q_values.argmax(dim=1).item())

    def update_option(self, obs: np.ndarray, reward: float, done: bool) -> str:
        """
        Update the current option based on termination condition and meta-controller.
        Returns new option.
        """
        if done:
            self.current_option = "Find"
            self.option_steps = 0
            return self.current_option

        # Check termination
        terminated = get_option_termination(obs, self.current_option)
        if terminated or self.option_steps >= self.max_option_steps:
            # Move to next option in sequence: Find -> Push -> Unwedge
            if self.current_option == "Find":
                self.current_option = "Push"
            elif self.current_option == "Push":
                self.current_option = "Unwedge"
            else:
                self.current_option = "Find"  # reset after Unwedge
            self.option_steps = 0
        else:
            self.option_steps += 1
        return self.current_option

    def reset_noise(self):
        self.q_network.reset_noise()
        self.target_network.reset_noise()

    def update(self):
        if len(self.replay_buffer) < self.batch_size * 10:
            return {}
        states, actions, rewards, next_states, dones, _ = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_online = self.q_network(next_states)
            next_actions = next_q_online.argmax(dim=1)
            next_q_target = self.target_network(next_states)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        self.reset_noise()
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        return {'loss': loss.item()}

    def save(self, path):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.total_steps = checkpoint['total_steps']

# ---------- Environment Wrappers ----------
class RewardShapingWrapper:
    """
    Potential-based reward shaping using distance to nearest box/wedge.
    Assumes observation contains distances.
    """
    def __init__(self, env, gamma=0.99):
        self.env = env
        self.gamma = gamma
        self.prev_distance_box = None
        self.prev_distance_wedge = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # Extract distances (modify indices as per your observation)
        self.prev_distance_box = self._get_box_distance(obs)
        self.prev_distance_wedge = self._get_wedge_distance(obs)
        return obs

    def step(self, action, render=False):
        obs, reward, done = self.env.step(action, render)
        # Shaping: improvement in distance to box or wedge
        curr_box_dist = self._get_box_distance(obs)
        curr_wedge_dist = self._get_wedge_distance(obs)
        shaping = 0.0
        if self.prev_distance_box is not None:
            shaping += (self.prev_distance_box - curr_box_dist) * 0.01  # small scaling
        if self.prev_distance_wedge is not None:
            shaping += (self.prev_distance_wedge - curr_wedge_dist) * 0.01
        shaped_reward = reward + shaping
        self.prev_distance_box = curr_box_dist
        self.prev_distance_wedge = curr_wedge_dist
        return obs, shaped_reward, done

    def _get_box_distance(self, obs):
        # Placeholder: assume index 10
        return obs[10] if len(obs) > 10 else 1000

    def _get_wedge_distance(self, obs):
        # Placeholder: assume index 11
        return obs[11] if len(obs) > 11 else 1000

class CurriculumWrapper:
    """
    Increases environment difficulty over episodes.
    """
    def __init__(self, env_class, start_difficulty=1, end_difficulty=3,
                 total_episodes=5000, **env_kwargs):
        self.env_class = env_class
        self.env_kwargs = env_kwargs
        self.start_diff = start_difficulty
        self.end_diff = end_difficulty
        self.total_episodes = total_episodes
        self.current_difficulty = start_difficulty
        self.env = None

    def _update_difficulty(self, episode):
        progress = min(1.0, episode / self.total_episodes)
        self.current_difficulty = self.start_diff + int(progress * (self.end_diff - self.start_diff))
        self.env_kwargs['difficulty'] = self.current_difficulty

    def reset(self, episode, seed=None):
        self._update_difficulty(episode)
        self.env = self.env_class(**self.env_kwargs)
        return self.env.reset(seed=seed)

    def step(self, action, render=False):
        return self.env.step(action, render)

# ---------- Evaluation ----------
def evaluate_agent(agent, OBELIX, difficulty, wall_obstacles,
                   n_episodes=10, max_steps=1000, seed=99999):
    agent.q_network.eval()
    rewards = []
    for i in range(n_episodes):
        env = OBELIX(scaling_factor=5, arena_size=500, max_steps=max_steps,
                     wall_obstacles=wall_obstacles, difficulty=difficulty,
                     box_speed=2, seed=seed + i)
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

# ---------- Main Training ----------
def train(obelix_py_path: str, episodes: int = 5000, max_steps: int = 2000,
          difficulty_start: int = 1, difficulty_end: int = 3,
          wall_obstacles: bool = False, save_path: str = "ddqn_noisy_options_curriculum.pt",
          eval_interval: int = 50):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OBELIX = mod.OBELIX

    # Curriculum wrapper
    base_env_kwargs = {
        'scaling_factor': 5, 'arena_size': 500, 'max_steps': max_steps,
        'wall_obstacles': wall_obstacles, 'box_speed': 2,
        'difficulty': difficulty_start  # will be overridden
    }
    curriculum_env = CurriculumWrapper(OBELIX, start_difficulty=difficulty_start,
                                       end_difficulty=difficulty_end,
                                       total_episodes=episodes,
                                       **base_env_kwargs)

    agent = NoisyDDQNWithOptions()
    print(f"\n{'='*60}")
    print("Training Noisy DDQN + Options + Curriculum + Reward Shaping")
    print(f"Difficulty: {difficulty_start} -> {difficulty_end} over {episodes} episodes")
    print(f"Device: {agent.device}")
    print(f"{'='*60}\n")

    best_eval_reward = float('-inf')
    recent_rewards = []

    for episode in tqdm(range(episodes), desc="Training"):
        # Reset environment with curriculum
        obs = curriculum_env.reset(episode, seed=episode)
        # Reward shaping wrapper applied per step
        shaping_wrapper = RewardShapingWrapper(curriculum_env.env, gamma=agent.gamma)
        shaping_wrapper.prev_distance_box = None
        shaping_wrapper.prev_distance_wedge = None

        episode_reward = 0.0
        done = False
        agent.reset_noise()
        agent.current_option = "Find"
        agent.option_steps = 0

        for step in range(max_steps):
            # Meta-controller updates option (may switch)
            new_option = agent.update_option(obs, 0.0, done)  # reward not used here
            # Select action using current option (same policy)
            action_idx = agent.select_action(obs)
            action = ACTIONS[action_idx]

            # Step with shaping
            next_obs, shaped_reward, done = shaping_wrapper.step(action, render=False)
            episode_reward += shaped_reward

            # Store transition (option not used for Q update, but kept for completeness)
            agent.replay_buffer.push(obs, action_idx, shaped_reward, next_obs, done, agent.current_option)

            obs = next_obs
            agent.total_steps += 1

            if agent.total_steps % 4 == 0:
                metrics = agent.update()

            if done:
                break

        recent_rewards.append(episode_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)

        if (episode + 1) % eval_interval == 0:
            eval_reward = evaluate_agent(agent, OBELIX, difficulty=curriculum_env.current_difficulty,
                                         wall_obstacles=wall_obstacles,
                                         n_episodes=10, max_steps=max_steps,
                                         seed=10000 + episode)
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save(save_path)
                tqdm.write(f"\n[Ep {episode+1}] Train: {episode_reward:.1f} | "
                           f"Eval: {eval_reward:.1f} | Best: {best_eval_reward:.1f} | *** SAVED ***")
            else:
                tqdm.write(f"\n[Ep {episode+1}] Train: {episode_reward:.1f} | "
                           f"Eval: {eval_reward:.1f} | Best: {best_eval_reward:.1f}")
        else:
            tqdm.write(f"\rEp {episode+1}: reward={episode_reward:.1f}, avg={np.mean(recent_rewards):.1f}", end="")

    print(f"\nTraining complete. Best model saved to {save_path} (eval={best_eval_reward:.1f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obelix_py", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--difficulty_start", type=int, default=1)
    parser.add_argument("--difficulty_end", type=int, default=3)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--save_path", type=str, default="ddqn_noisy_options_curriculum.pt")
    args = parser.parse_args()
    train(obelix_py_path=args.obelix_py,
          episodes=args.episodes,
          max_steps=args.max_steps,
          difficulty_start=args.difficulty_start,
          difficulty_end=args.difficulty_end,
          wall_obstacles=args.wall_obstacles,
          save_path=args.save_path)