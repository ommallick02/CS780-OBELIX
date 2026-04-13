"""
Hierarchical A2C with Options Framework, Eligibility Traces, and Curriculum Learning.
Options: FIND, ALIGN, PUSH, RECOVER
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Sequence, Tuple, Dict, Optional
from collections import deque
from tqdm import tqdm

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = len(ACTIONS)

# Option definitions
OPTION_FIND = 0    # Locate and approach box
OPTION_ALIGN = 1   # Fine-tune orientation (optional intermediate)
OPTION_PUSH = 2    # Push box to boundary
OPTION_RECOVER = 3 # Get unstuck from walls
N_OPTIONS = 4

class IntraOptionNetwork(nn.Module):
    """
    Actor-Critic for a specific option with eligibility trace support.
    Uses separate heads for primitive actions.
    """
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
        self.terminator = nn.Linear(hidden_dim, 1)  # Termination probability
        
        # Initialize for stability
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.orthogonal_(self.critic.weight, 1.0)
        
    def forward(self, x: torch.Tensor):
        f = self.shared(x)
        logits = self.actor(f)
        value = self.critic(f)
        term_prob = torch.sigmoid(self.terminator(f))
        return logits, value.squeeze(-1), term_prob.squeeze(-1)
    
    def get_action(self, x: torch.Tensor, deterministic: bool = False):
        logits, value, term_prob = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
            log_prob = torch.log(probs.gather(-1, action.unsqueeze(-1))).squeeze()
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob, value, term_prob

class OptionCritic(nn.Module):
    """
    High-level policy over options.
    """
    def __init__(self, input_dim: int = 18, hidden_dim: int = 128, n_options: int = N_OPTIONS):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.option_values = nn.Linear(hidden_dim, n_options)  # Q_Ω(s, ω)
        
    def forward(self, x: torch.Tensor):
        f = self.encoder(x)
        q_values = self.option_values(f)
        return q_values

class RewardShaper:
    """
    Potential-based reward shaping for OBELIX.
    Φ(s) encodes domain knowledge without changing optimal policy.
    """
    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
        self.prev_potential = 0.0
        
    def compute_potential(self, obs: np.ndarray, attached: bool) -> float:
        """
        Potential based on:
        - Distance to box (when not attached)
        - Proximity to boundary (when attached)
        """
        obs = obs[:16]  # Sonar sensors only
        
        if not attached:
            # Find which sensors detect box
            near_sensors = obs[1::2]  # Odd indices are near bits
            far_sensors = obs[0::2]   # Even indices are far bits
            
            # Higher potential if box detected by forward sensors (indices 2,3,4,5 in sensor layout)
            forward_activation = (near_sensors[2:6].sum() * 2.0 + 
                                 far_sensors[2:6].sum() * 1.0)
            
            # Bonus for IR sensor (index 16 in original obs, but we sliced)
            # Actually IR is separate, handle outside
            return min(forward_activation * 0.5, 10.0)
        else:
            # When attached, potential based on progress toward boundary
            # Higher if facing toward boundary (simplified: reward forward motion when attached)
            return 5.0  # Constant potential when attached (will be shaped by actual movement)
    
    def shape_reward(self, obs: np.ndarray, next_obs: np.ndarray, 
                     reward: float, attached: bool, stuck: bool) -> float:
        """
        F(s, a, s') = γΦ(s') - Φ(s)
        Additional shaping for specific events.
        """
        curr_pot = self.compute_potential(obs, attached)
        
        # Add IR sensor contribution
        if obs[16] > 0.5:  # IR sensor
            curr_pot += 3.0
            
        self.prev_potential = curr_pot
        
        shaped = reward + 0.5 * curr_pot  # Scaling factor for shaping
        
        # Additional intrinsic rewards for exploration
        if stuck:
            shaped -= 5.0  # Penalty for getting stuck (beyond env -200)
            
        return shaped

class CurriculumManager:
    """
    Manages progressive difficulty increases based on performance.
    """
    def __init__(self):
        self.difficulty_levels = [
            (0, False, 0),      # Static, no walls
            (0, True, 0),       # Static, with walls
            (2, False, 0),      # Blinking, no walls
            (2, True, 0),       # Blinking, with walls
            (3, False, 1),      # Moving slow, no walls
            (3, True, 1),       # Moving slow, with walls
            (3, False, 3),      # Moving fast, no walls
            (3, True, 3),       # Moving fast, with walls (hardest)
        ]
        self.current_level = 0
        self.success_history = deque(maxlen=20)
        self.advance_threshold = 0.75  # 75% success rate to advance
        
    def get_current_config(self):
        difficulty, walls, speed = self.difficulty_levels[self.current_level]
        return {
            'difficulty': difficulty,
            'wall_obstacles': walls,
            'box_speed': speed
        }
    
    def update(self, episode_reward: float):
        """
        Check if should advance/retreat difficulty based on recent performance.
        Success = got the +2000 bonus (approximately, check if reward > 1500)
        """
        success = episode_reward > 1500
        self.success_history.append(success)
        
        if len(self.success_history) >= 20:
            success_rate = sum(self.success_history) / len(self.success_history)
            
            if success_rate > self.advance_threshold and self.current_level < len(self.difficulty_levels) - 1:
                self.current_level += 1
                self.success_history.clear()
                return True  # Advanced
            elif success_rate < 0.2 and self.current_level > 0:
                self.current_level -= 1
                self.success_history.clear()
                return False  # Regressed
                
        return None  # No change

class EligibilityTraceBuffer:
    """
    Online TD(λ) with eligibility traces for each option.
    Enables faster credit assignment in long episodes.
    """
    def __init__(self, lambda_: float = 0.95, gamma: float = 0.99):
        self.lambda_ = lambda_
        self.gamma = gamma
        self.reset()
        
    def reset(self):
        self.traces = {}  # Parameter name -> trace tensor
        self.steps = 0
        
    def init_traces(self, model: nn.Module):
        """Initialize trace tensors matching model parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.traces[name] = torch.zeros_like(param.data)
                
    def update_traces(self, model: nn.Module, log_prob: torch.Tensor, 
                      value: torch.Tensor, td_error: float, 
                      is_terminal: bool = False):
        """
        Update eligibility traces using the accumulating trace method.
        z_t = γλ z_{t-1} + ∇log π(a|s) for actor
        z_t = γλ z_{t-1} + ∇V(s) for critic
        """
        # Compute gradients
        model.zero_grad()
        
        # Combined objective for gradient computation
        # We compute gradients separately for actor and critic components
        loss = -log_prob * td_error  # Policy gradient component
        loss.backward(retain_graph=True)
        
        # Accumulate traces for policy
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if 'actor' in name or 'shared' in name:
                    self.traces[name] = (self.gamma * self.lambda_ * self.traces[name] + 
                                        param.grad)
                    
        model.zero_grad()
        value.backward()
        
        # Accumulate traces for critic
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if 'critic' in name or 'shared' in name:
                    self.traces[name] = (self.gamma * self.lambda_ * self.traces[name] + 
                                        param.grad)
                    
        if is_terminal:
            self.reset()

class HierarchicalA2CAgent:
    """
    Hierarchical Agent with Options and Eligibility Traces.
    """
    def __init__(self, 
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 lambda_: float = 0.95,
                 entropy_coef: float = 0.01,
                 option_termination_prob: float = 0.95):
        
        self.device = "cpu"  # Ensure CPU compatibility
        self.gamma = gamma
        self.lambda_ = lambda_
        self.entropy_coef = entropy_coef
        self.option_term_prob = option_termination_prob
        self.lr = lr
        
        # Networks
        self.intra_option_policies = nn.ModuleList([
            IntraOptionNetwork() for _ in range(N_OPTIONS)
        ]).to(self.device)
        
        self.option_critic = OptionCritic().to(self.device)
        
        # Optimizers
        self.intra_optims = [optim.Adam(net.parameters(), lr=lr) 
                            for net in self.intra_option_policies]
        self.critic_optim = optim.Adam(self.option_critic.parameters(), lr=lr)
        
        # Eligibility traces for each option
        self.trace_buffers = [EligibilityTraceBuffer(lambda_, gamma) 
                             for _ in range(N_OPTIONS)]
        
        # Current state
        self.current_option = None
        self.option_steps = 0
        self.max_option_steps = 100  # Force termination after N steps
        
        # Shaping and curriculum
        self.shaper = RewardShaper(gamma)
        self.curriculum = CurriculumManager()
        
    def select_option(self, obs: np.ndarray, attached: bool, stuck: bool, 
                     rng: np.random.Generator) -> int:
        """
        Hand-crafted initiation sets with learned termination.
        Uses high-level Q-values when multiple options available.
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        q_values = self.option_critic(obs_t).squeeze().detach().cpu().numpy()
        
        # Mask invalid options based on initiation sets
        valid_options = np.ones(N_OPTIONS, dtype=bool)
        
        if attached:
            # Can only PUSH or RECOVER when attached
            valid_options[OPTION_FIND] = False
            valid_options[OPTION_ALIGN] = False
        else:
            # Can only FIND or RECOVER when not attached
            valid_options[OPTION_PUSH] = False
            
        if not stuck:
            valid_options[OPTION_RECOVER] = False
            
        valid_indices = np.where(valid_options)[0]
        
        if len(valid_indices) == 0:
            # Fallback: use RECOVER if stuck, else FIND
            return OPTION_RECOVER if stuck else OPTION_FIND
            
        # ε-greedy over valid options
        if rng.random() < 0.1:
            return int(rng.choice(valid_indices))
        else:
            valid_q = q_values[valid_indices]
            return int(valid_indices[np.argmax(valid_q)])
    
    def should_terminate(self, obs: np.ndarray, option_idx: int, 
                        attached: bool, stuck: bool) -> bool:
        """
        Check if option should terminate based on conditions or learned terminator.
        """
        # Hand-crafted termination conditions
        if option_idx == OPTION_FIND and attached:
            return True
        if option_idx == OPTION_PUSH and not attached:
            return True
        if option_idx == OPTION_RECOVER and not stuck:
            return True
        if self.option_steps >= self.max_option_steps:
            return True
            
        # Learned termination
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        _, _, term_prob = self.intra_option_policies[option_idx](obs_t)
        
        return term_prob.item() > 0.5
    
    def train_episode(self, env, seed: int, max_steps: int, 
                     rng: np.random.Generator) -> Tuple[float, int]:
        """
        Train for one episode using online TD(λ) updates.
        """
        obs = env.reset(seed=seed)
        self.current_option = None
        self.option_steps = 0
        
        total_reward = 0.0
        steps = 0
        
        # Initialize traces for current option
        if self.current_option is not None:
            self.trace_buffers[self.current_option].reset()
            self.trace_buffers[self.current_option].init_traces(
                self.intra_option_policies[self.current_option]
            )
        
        while steps < max_steps:
            attached = env.enable_push
            stuck = bool(obs[17])
            
            # Option selection/termination
            if (self.current_option is None or 
                self.should_terminate(obs, self.current_option, attached, stuck)):
                
                self.current_option = self.select_option(obs, attached, stuck, rng)
                self.option_steps = 0
                self.trace_buffers[self.current_option].reset()
                self.trace_buffers[self.current_option].init_traces(
                    self.intra_option_policies[self.current_option]
                )
            
            # Execute intra-option policy
            option_net = self.intra_option_policies[self.current_option]
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            action_idx, log_prob, value, _ = option_net.get_action(obs_t, deterministic=False)
            action = ACTIONS[action_idx]
            
            # Step environment
            next_obs, reward, done = env.step(action, render=False)
            
            # Reward shaping
            shaped_reward = self.shaper.shape_reward(
                obs, next_obs, reward, attached, stuck
            )
            
            total_reward += reward  # Track original reward
            steps += 1
            self.option_steps += 1
            
            # Compute TD error
            next_obs_t = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, next_value, _ = option_net(next_obs_t)
                if done:
                    next_value = torch.tensor(0.0)
            
            td_error = shaped_reward + self.gamma * next_value.item() - value.item()
            
            # Update eligibility traces and parameters
            trace_buffer = self.trace_buffers[self.current_option]
            
            # Manual parameter update using traces
            option_net.zero_grad()
            value_loss = (value - (value.item() + td_error)) ** 2
            policy_loss = -log_prob * td_error
            
            total_loss = policy_loss + 0.5 * value_loss - self.entropy_coef * torch.distributions.Categorical(
                torch.softmax(option_net(obs_t)[0], dim=-1)
            ).entropy()
            
            total_loss.backward()
            
            # Apply trace-based updates
            with torch.no_grad():
                for name, param in option_net.named_parameters():
                    if param.requires_grad and name in trace_buffer.traces:
                        # Update rule: Δθ = α * δ * z
                        param.data += self.lr * td_error * trace_buffer.traces[name]
            
            # Update high-level critic (option values)
            if done or self.should_terminate(next_obs, self.current_option, 
                                             env.enable_push, bool(next_obs[17])):
                # Update Q_Ω
                self.critic_optim.zero_grad()
                q_values = self.option_critic(obs_t)
                option_q = q_values[0, self.current_option]
                
                # Target: cumulative return from executing option
                target = shaped_reward + self.gamma * torch.max(
                    self.option_critic(next_obs_t)
                ).item() if not done else shaped_reward
                
                critic_loss = (option_q - target) ** 2
                critic_loss.backward()
                self.critic_optim.step()
            
            obs = next_obs
            
            if done:
                break
                
        return total_reward, steps

def train(obelix_py_path: str, episodes: int = 5000, max_steps: int = 2000,
          save_path: str = "hierarchical_a2c.pt", force_walls: bool = False):
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OBELIX = mod.OBELIX
    
    agent = HierarchicalA2CAgent()

    # If force_walls is True, start at curriculum level 1 (walls, static)
    if force_walls:
        agent.curriculum.current_level = 1
        print("Forcing wall obstacles enabled (Level 1)")

    rng = np.random.default_rng(42)
    
    recent_rewards = deque(maxlen=50)
    best_avg = float('-inf')
    
    print(f"Training Hierarchical A2C with:")
    print(f"  - Options Framework ({N_OPTIONS} options)")
    print(f"  - Eligibility Traces (λ={agent.lambda_})")
    print(f"  - Curriculum Learning")
    print(f"  - Reward Shaping")
    
    for ep in tqdm(range(episodes), desc="Training"):
        # Get curriculum config
        config = agent.curriculum.get_current_config()
        
        env = OBELIX(
            scaling_factor=5,
            arena_size=500,
            max_steps=max_steps,
            wall_obstacles=config['wall_obstacles'],
            difficulty=config['difficulty'],
            box_speed=config['box_speed'],
            seed=ep
        )
        
        episode_reward, steps = agent.train_episode(env, ep, max_steps, rng)
        recent_rewards.append(episode_reward)
        
        # Update curriculum
        level_changed = agent.curriculum.update(episode_reward)
        if level_changed is True:
            tqdm.write(f"↑ Advanced to level {agent.curriculum.current_level} at episode {ep}")
        elif level_changed is False:
            tqdm.write(f"↓ Regressed to level {agent.curriculum.current_level} at episode {ep}")
        
        if (ep + 1) % 50 == 0:
            avg_reward = np.mean(recent_rewards)
            config = agent.curriculum.get_current_config()
            tqdm.write(f"[Ep {ep+1}] Reward: {episode_reward:.0f}, "
                      f"Avg50: {avg_reward:.0f}, "
                      f"Level: {agent.curriculum.current_level}, "
                      f"Config: {config}")
            
            if avg_reward > best_avg:
                best_avg = avg_reward
                # Save all intra-option policies
                checkpoint = {
                    'intra_option_states': [net.state_dict() for net in agent.intra_option_policies],
                    'critic_state': agent.option_critic.state_dict(),
                    'curriculum_level': agent.curriculum.current_level,
                    'episode': ep
                }
                torch.save(checkpoint, save_path.replace('.pt', '_best.pt'))
    
    # Final save
    checkpoint = {
        'intra_option_states': [net.state_dict() for net in agent.intra_option_policies],
        'critic_state': agent.option_critic.state_dict(),
        'curriculum_level': agent.curriculum.current_level
    }
    torch.save(checkpoint, save_path)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obelix_py", required=True)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--save_path", default="hierarchical_a2c.pt")
    parser.add_argument("--wall_obstacles", action="store_true", 
                       help="Enable wall obstacles (overrides curriculum initial level)")
    args = parser.parse_args()
    
    # If wall_obstacles specified, start at level 1 (static with walls)
    # otherwise start at level 0
    train(args.obelix_py, args.episodes, args.max_steps, args.save_path, 
          force_walls=args.wall_obstacles)