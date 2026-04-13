"""
Hybrid MCTS + Q-Network Evaluation and Refinement.

Uses trained Q-network for rollouts, MCTS for action selection.
Can also do "Expert Iteration": Use MCTS to generate better training data.
"""

import argparse
import os
import copy
from typing import List, Tuple
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Must match agent.py
ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = 5
STACK_SIZE = 4
USE_STACKING = True


class DuelingDQN(nn.Module):
    def __init__(self, in_dim=18*STACK_SIZE, n_actions=5, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden, 1)
        self.advantage_head = nn.Linear(hidden, n_actions)
    
    def forward(self, x):
        f = self.encoder(x)
        v = self.value_head(f)
        a = self.advantage_head(f)
        return v + a - a.mean(dim=-1, keepdim=True)


class HybridMCTS:
    """Simplified MCTS for training/evaluation (full env access)."""
    
    def __init__(self, q_network, n_sims=15, max_depth=10, gamma=0.99):
        self.q_network = q_network
        self.q_network.eval()
        self.device = next(q_network.parameters()).device
        
        self.n_sims = n_sims
        self.max_depth = max_depth
        self.gamma = gamma
        
        self.frame_buffer = None
    
    def reset_buffer(self, obs):
        self.frame_buffer = deque(maxlen=STACK_SIZE)
        for _ in range(STACK_SIZE-1):
            self.frame_buffer.append(np.zeros(18, dtype=np.float32))
        self.frame_buffer.append(obs.astype(np.float32).copy())
    
    def get_state_tensor(self):
        if USE_STACKING:
            stacked = np.concatenate(list(self.frame_buffer), axis=0)
        else:
            stacked = self.frame_buffer[-1]
        return torch.FloatTensor(stacked).unsqueeze(0).to(self.device)
    
    def select_action(self, env, obs, rng, training=True):
        """
        MCTS with real environment rollouts (slower but accurate).
        Only use this for evaluation or data generation, not submission.
        """
        import math
        
        # Tree: state -> {action: (visits, value, children)}
        tree = {}
        
        root_state = tuple(obs.round(decimals=1))
        
        for sim in range(self.n_sims):
            # Clone env for simulation
            sim_env = copy.deepcopy(env)
            sim_obs = obs.copy()
            sim_buffer = copy.deepcopy(self.frame_buffer)
            
            path = []  # Track (state, action) pairs
            done = False
            step = 0
            
            # SELECTION & EXPANSION
            current_state = root_state
            while step < self.max_depth and not done:
                if current_state not in tree:
                    # Expand
                    tree[current_state] = {
                        'visits': 0,
                        'value': 0.0,
                        'children': {},
                        'untried': list(range(N_ACTIONS))
                    }
                
                node = tree[current_state]
                
                # Select action (UCB1 or untried)
                if node['untried']:
                    action = node['untried'].pop(rng.integers(len(node['untried'])))
                else:
                    # UCB1
                    best_score = float('-inf')
                    action = 0
                    for a in range(N_ACTIONS):
                        if a in node['children']:
                            child = node['children'][a]
                            if child['visits'] > 0:
                                exploitation = child['value'] / child['visits']
                                exploration = 1.414 * math.sqrt(math.log(node['visits']) / child['visits'])
                                score = exploitation + exploration
                            else:
                                score = float('inf')
                        else:
                            score = float('inf')
                        
                        if score > best_score:
                            best_score = score
                            action = a
                
                # Take action in sim env
                sim_obs, reward, done = sim_env.step(ACTIONS[action], render=False)
                path.append((current_state, action, reward))
                current_state = tuple(sim_obs.round(decimals=1))
                step += 1
                
                # Stop if we found a good reward (early termination)
                if reward > 100:  # Found box
                    break
            
            # ROLLOUT: Use Q-network for remaining steps
            rollout_reward = 0.0
            discount = self.gamma ** step
            
            for r_step in range(self.max_depth - step):
                if done:
                    break
                
                # Use Q-network to select action
                sim_buffer.append(sim_obs.astype(np.float32))
                stacked = np.concatenate(list(sim_buffer), axis=0) if USE_STACKING else sim_buffer[-1]
                state_t = torch.FloatTensor(stacked).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    q_vals = self.q_network(state_t).squeeze(0).cpu().numpy()
                    action = int(np.argmax(q_vals))  # Greedy in rollout
                
                sim_obs, reward, done = sim_env.step(ACTIONS[action], render=False)
                rollout_reward += discount * reward
                discount *= self.gamma
            
            # BACKPROPAGATION
            total_return = sum(r * (self.gamma ** i) for i, (_, _, r) in enumerate(path))
            total_return += rollout_reward
            
            for state, action, _ in path:
                if state in tree:
                    tree[state]['visits'] += 1
                    tree[state]['value'] += total_return
                    if action not in tree[state]['children']:
                        tree[state]['children'][action] = {'visits': 0, 'value': 0.0}
                    tree[state]['children'][action]['visits'] += 1
                    tree[state]['children'][action]['value'] += total_return
        
        # Select best action by visit count
        root = tree.get(root_state, {'children': {}})
        if not root['children']:
            # Fallback to Q-network
            self.frame_buffer.append(obs.astype(np.float32))
            state_t = self.get_state_tensor()
            with torch.no_grad():
                return int(self.q_network(state_t).argmax(dim=1).item())
        
        best_action = max(root['children'].items(), 
                         key=lambda x: x[1]['visits'])[0]
        return best_action


def evaluate_hybrid(obelix_py_path, difficulty=3, wall_obstacles=False, 
                   n_episodes=50, n_sims=15):
    """Evaluate hybrid agent."""
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OBELIX = mod.OBELIX
    
    # Load Q-network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net = DuelingDQN().to(device)
    
    if os.path.exists("weights.pth"):
        ckpt = torch.load("weights.pth", map_location=device)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        q_net.load_state_dict(ckpt)
        print("Loaded weights.pth")
    else:
        print("Warning: No weights found, using random Q-network")
    
    q_net.eval()
    
    # Create hybrid MCTS
    mcts = HybridMCTS(q_net, n_sims=n_sims, max_depth=10)
    
    rewards = []
    successes = 0
    
    print(f"\n{'='*60}")
    print(f"Evaluating Hybrid MCTS + Q-Network")
    print(f"Sims per step: {n_sims}")
    print(f"{'='*60}")
    
    for ep in tqdm(range(n_episodes), desc="Evaluating"):
        env = OBELIX(
            scaling_factor=5,
            arena_size=500,
            max_steps=2000,
            wall_obstacles=wall_obstacles,
            difficulty=difficulty,
            box_speed=2,
            seed=ep
        )
        
        obs = env.reset(seed=ep)
        rng = np.random.default_rng(ep)
        mcts.reset_buffer(obs)
        
        total_reward = 0.0
        done = False
        steps = 0
        
        while not done and steps < 2000:
            # Hybrid action selection
            action_idx = mcts.select_action(env, obs, rng, training=False)
            action = ACTIONS[action_idx]
            
            obs, reward, done = env.step(action, render=False)
            total_reward += reward
            steps += 1
            
            # Update buffer for MCTS
            mcts.frame_buffer.append(obs.astype(np.float32))
            
            if reward >= 2000:  # Success
                successes += 1
        
        rewards.append(total_reward)
    
    print(f"\n{'='*60}")
    print(f"Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Success Rate: {successes}/{n_episodes} ({100*successes/n_episodes:.1f}%)")
    print(f"{'='*60}")
    
    return rewards


def expert_iteration(obelix_py_path, n_iterations=10, episodes_per_iter=100):
    """
    Expert Iteration: Use MCTS to generate better data, refine Q-network.
    This improves the Q-network using MCTS as the expert.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OBELIX = mod.OBELIX
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize or load Q-network
    q_net = DuelingDQN().to(device)
    if os.path.exists("weights.pth"):
        q_net.load_state_dict(torch.load("weights.pth", map_location=device))
    
    optimizer = optim.Adam(q_net.parameters(), lr=1e-4)
    
    mcts = HybridMCTS(q_net, n_sims=10, max_depth=8)  # Faster MCTS for data gen
    
    print(f"\nStarting Expert Iteration...")
    
    for iteration in range(n_iterations):
        trajectories = []  # (state, action, return) tuples
        
        # Generate data using MCTS as expert
        for ep in tqdm(range(episodes_per_iter), desc=f"Iter {iteration+1} Data Gen"):
            env = OBELIX(scaling_factor=5, difficulty=1, wall_obstacles=True, seed=iteration*1000+ep)
            obs = env.reset(seed=iteration*1000+ep)
            rng = np.random.default_rng(iteration*1000+ep)
            mcts.reset_buffer(obs)
            
            episode_data = []
            done = False
            steps = 0
            
            while not done and steps < 500:  # Shorter episodes for data gen
                # Use MCTS to select action (the "expert")
                action_idx = mcts.select_action(env, obs, rng, training=True)
                
                # Store state-action pair
                mcts.frame_buffer.append(obs.astype(np.float32))
                state_tensor = mcts.get_state_tensor().cpu().numpy()
                episode_data.append((state_tensor, action_idx))
                
                obs, reward, done = env.step(ACTIONS[action_idx], render=False)
                steps += 1
            
            # Calculate returns and add to trajectories
            # (Simplified - using uniform weighting here)
            for state, action in episode_data:
                trajectories.append((state, action))
        
        # Train Q-network to mimic MCTS (behavior cloning + Q-learning)
        q_net.train()
        np.random.shuffle(trajectories)
        
        batch_size = 32
        losses = []
        
        for i in range(0, len(trajectories), batch_size):
            batch = trajectories[i:i+batch_size]
            states = torch.FloatTensor(np.array([b[0] for b in batch])).squeeze(1).to(device)
            actions = torch.LongTensor([b[1] for b in batch]).to(device)
            
            q_values = q_net(states)
            target_q = q_values.clone().detach()
            
            # Encourage the selected MCTS action
            # Loss: make Q-value of MCTS action higher than others
            target_q[range(len(actions)), actions] = 10.0  # Arbitrary high value for MCTS choice
            
            loss = nn.functional.mse_loss(q_values, target_q)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        q_net.eval()
        print(f"Iteration {iteration+1}: Avg Loss = {np.mean(losses):.4f}")
        
        # Save improved network
        torch.save(q_net.state_dict(), f"weights_expert_iter_{iteration+1}.pth")
    
    # Save final
    torch.save(q_net.state_dict(), "weights.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obelix_py", type=str, required=True)
    parser.add_argument("--mode", type=str, default="expert_iter", choices=["eval", "expert_iter"])
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--n_sims", type=int, default=15, help="MCTS simulations per step")
    parser.add_argument("--episodes", type=int, default=50)
    
    args = parser.parse_args()
    
    if args.mode == "eval":
        evaluate_hybrid(
            args.obelix_py,
            difficulty=args.difficulty,
            wall_obstacles=args.wall_obstacles,
            n_episodes=args.episodes,
            n_sims=args.n_sims
        )
    else:
        expert_iteration(args.obelix_py)
