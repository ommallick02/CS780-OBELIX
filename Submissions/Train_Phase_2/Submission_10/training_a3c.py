"""
Training script for A3C (Asynchronous Advantage Actor-Critic).
Uses multiple parallel workers to collect experience asynchronously.

python training_a3c.py --obelix_py ./obelix.py --episodes 3000 --n_workers 8 --difficulty 2 --wall_obstacles
"""

import argparse
import os
from typing import List, Tuple, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Process, Queue, Value, Lock

mp.set_start_method('spawn', force=True)

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = len(ACTIONS)


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim: int = 18, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_dim, N_ACTIONS)
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value


def worker_process(
    worker_id: int,
    global_network: ActorCriticNetwork,
    shared_optimizer: optim.Adam,
    lock: Lock,
    obelix_py_path: str,
    queue: Queue,
    stop_flag: Value,
    episodes_per_worker: int,
    max_steps: int,
    difficulty: int,
    wall_obstacles: bool,
    gamma: float,
    n_steps: int,
    entropy_coef: float,
    seed_offset: int
):
    """Worker process that collects experience and computes gradients."""
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OBELIX = mod.OBELIX
    
    # Local network (copy of global)
    local_network = ActorCriticNetwork(input_dim=18, hidden_dim=128)
    local_network.load_state_dict(global_network.state_dict())
    device = "cpu"
    local_network.to(device)
    
    worker_episodes = 0
    total_reward = 0.0
    
    while worker_episodes < episodes_per_worker and not stop_flag.value:
        seed = seed_offset + worker_id * 10000 + worker_episodes
        env = OBELIX(
            scaling_factor=5,
            arena_size=500,
            max_steps=max_steps,
            wall_obstacles=wall_obstacles,
            difficulty=difficulty,
            box_speed=2,
            seed=seed
        )
        
        obs = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        
        # Collect trajectory
        observations = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        
        episode_reward = 0.0
        done = False
        
        for step in range(max_steps):
            # Sync with global periodically
            if step % 20 == 0:
                local_network.load_state_dict(global_network.state_dict())
            
            # Get action (WITH gradients)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            logits, value = local_network(obs_tensor)
            
            # Compute probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Sample action
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action_idx))
            
            action = ACTIONS[action_idx]
            next_obs, reward, done = env.step(action, render=False)
            episode_reward += reward
            
            # Store (detach tensors for storage, we'll recompute for gradients)
            observations.append(obs.copy())
            actions.append(action_idx)
            rewards.append(reward)
            
            # Store tensors that need gradients
            values.append(value.squeeze(0))  # Keep gradient
            log_probs.append(log_prob)  # Keep gradient
            
            obs = next_obs
            
            # Update when batch is full or episode ends
            if len(observations) >= n_steps or done:
                # Compute returns and advantages
                with torch.no_grad():
                    if not done:
                        next_obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                        _, next_value = local_network(next_obs_tensor)
                        R = next_value.item()
                    else:
                        R = 0.0
                
                # Compute returns
                returns = []
                for r in reversed(rewards):
                    R = r + gamma * R
                    returns.insert(0, R)
                returns = torch.FloatTensor(returns).to(device)
                
                # Stack values and log_probs
                values_t = torch.stack(values)
                log_probs_t = torch.stack(log_probs)
                
                # Compute advantages
                advantages = returns - values_t.detach()
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Compute loss
                policy_loss = -(log_probs_t * advantages).mean()
                value_loss = nn.functional.mse_loss(values_t, returns)
                
                # Entropy
                probs_all = torch.softmax(logits, dim=-1)
                entropy = -(probs_all * torch.log(probs_all + 1e-8)).sum(dim=-1).mean()
                
                total_loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
                
                # Compute gradients
                local_network.zero_grad()
                total_loss.backward()
                
                # Apply gradients to global network
                with lock:
                    for global_param, local_param in zip(global_network.parameters(), local_network.parameters()):
                        if global_param.grad is None:
                            global_param.grad = local_param.grad.clone()
                        else:
                            global_param.grad.add_(local_param.grad)
                    
                    shared_optimizer.step()
                    shared_optimizer.zero_grad()
                
                # Sync local with global
                local_network.load_state_dict(global_network.state_dict())
                
                # Clear buffers
                observations = []
                actions = []
                rewards = []
                values = []
                log_probs = []
            
            if done:
                break
        
        total_reward += episode_reward
        worker_episodes += 1
        
        queue.put({
            'worker_id': worker_id,
            'episode': worker_episodes,
            'reward': episode_reward,
            'avg_reward': total_reward / worker_episodes
        })
    
    queue.put({'worker_id': worker_id, 'done': True})


def evaluate_agent(network: ActorCriticNetwork, OBELIX, difficulty, wall_obstacles,
                   n_episodes=10, max_steps=1000, seed=99999):
    """Evaluate global network."""
    network.eval()
    rewards = []
    device = "cpu"
    
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
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = network(obs_tensor)
                probs = torch.softmax(logits, dim=-1)
            
            action_idx = int(torch.argmax(probs).item())
            action = ACTIONS[action_idx]
            
            obs, reward, done = env.step(action, render=False)
            total_reward += reward
            
            if done:
                break
        
        rewards.append(total_reward)
    
    network.train()
    return np.mean(rewards)


def train(
    obelix_py_path: str,
    episodes: int = 2000,
    max_steps: int = 1000,
    difficulty: int = 0,
    wall_obstacles: bool = False,
    save_path: str = "a3c_agent.pt",
    eval_interval: int = 100,
    hidden_dim: int = 128,
    lr: float = 3e-4,
    n_steps: int = 5,
    entropy_coef: float = 0.01,
    n_workers: int = 4
):
    """Main A3C training with parallel workers."""
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OBELIX = mod.OBELIX
    
    print(f"\n{'='*60}")
    print(f"Training A3C Agent")
    print(f"{'='*60}")
    print(f"Total episodes: {episodes}")
    print(f"Workers: {n_workers}")
    print(f"Difficulty: {difficulty}")
    print(f"{'='*60}\n")
    
    # Global network
    global_network = ActorCriticNetwork(input_dim=18, hidden_dim=hidden_dim)
    global_network.train()
    
    # Shared optimizer
    shared_optimizer = optim.Adam(global_network.parameters(), lr=lr)
    lock = Lock()
    
    best_eval_reward = float('-inf')
    
    # Communication queue
    queue = Queue()
    stop_flag = Value('i', 0)
    
    episodes_per_worker = episodes // n_workers
    
    # Spawn workers
    processes = []
    for worker_id in range(n_workers):
        p = Process(
            target=worker_process,
            args=(
                worker_id,
                global_network,
                shared_optimizer,
                lock,
                obelix_py_path,
                queue,
                stop_flag,
                episodes_per_worker,
                max_steps,
                difficulty,
                wall_obstacles,
                0.99,  # gamma
                n_steps,
                entropy_coef,
                0  # seed offset
            )
        )
        p.start()
        processes.append(p)
    
    # Monitor progress
    pbar = tqdm(total=episodes, desc="A3C Training", unit="ep")
    active_workers = n_workers
    completed_episodes = 0
    
    while active_workers > 0:
        try:
            msg = queue.get(timeout=1.0)
            
            if msg.get('done'):
                active_workers -= 1
            else:
                completed_episodes += 1
                reward = msg['reward']
                
                pbar.update(1)
                pbar.set_postfix({
                    'worker': msg['worker_id'],
                    'reward': f'{reward:.1f}'
                })
                
                # Evaluation
                if completed_episodes % eval_interval == 0:
                    eval_reward = evaluate_agent(
                        global_network, OBELIX, difficulty, wall_obstacles,
                        n_episodes=10, max_steps=max_steps,
                        seed=10000 + completed_episodes
                    )
                    tqdm.write(f"\n[Ep {completed_episodes}/{episodes}] "
                              f"Eval: {eval_reward:8.1f} | "
                              f"Best: {max(best_eval_reward, eval_reward):8.1f}")
                    
                    if eval_reward > best_eval_reward:
                        best_eval_reward = eval_reward
                        torch.save({
                            'network_state_dict': global_network.state_dict(),
                            'optimizer_state_dict': shared_optimizer.state_dict(),
                        }, save_path.replace('.pt', '_best.pt'))
                        tqdm.write(f"*** New best model saved! ***")
        
        except Exception as e:
            if all(not p.is_alive() for p in processes):
                break
    
    for p in processes:
        p.join()
    
    pbar.close()
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    torch.save({
        'network_state_dict': global_network.state_dict(),
        'optimizer_state_dict': shared_optimizer.state_dict(),
    }, save_path)
    print(f"Best eval reward: {best_eval_reward:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obelix_py", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--save_path", type=str, default="a3c_agent.pt")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--n_workers", type=int, default=4)
    
    args = parser.parse_args()
    
    train(
        obelix_py_path=args.obelix_py,
        episodes=args.episodes,
        max_steps=args.max_steps,
        difficulty=args.difficulty,
        wall_obstacles=args.wall_obstacles,
        save_path=args.save_path,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        n_steps=args.n_steps,
        entropy_coef=args.entropy_coef,
        n_workers=args.n_workers
    )