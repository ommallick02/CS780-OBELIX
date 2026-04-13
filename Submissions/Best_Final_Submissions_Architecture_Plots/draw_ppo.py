import torch
import torch.nn as nn
from torchviz import make_dot

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

if __name__ == "__main__":
    device = torch.device("cpu")
    model = ActorCritic(input_dim=18, hidden=128).to(device)
    
    dummy_input = torch.zeros(1, 18, device=device)
    probs, value = model(dummy_input)
    
    # Summing actor and critic outputs to trace the shared encoder graph seamlessly
    loss_dummy = probs.sum() + value.sum()
    
    dot = make_dot(loss_dummy, params=dict(model.named_parameters()))
    dot.render("ppo_actor_critic_architecture", format="png", cleanup=True)
    print("Saved PPO Actor-Critic architecture diagram to ppo_actor_critic_architecture.png")