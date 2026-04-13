import torch
import torch.nn as nn
from torchviz import make_dot

STACK_SIZE = 4
N_ACTIONS = 5

class Actor(nn.Module):
    def __init__(self, in_dim=18*STACK_SIZE, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
        )
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, in_dim=18*STACK_SIZE, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

if __name__ == "__main__":
    device = torch.device("cpu")
    actor = Actor().to(device)
    critic = Critic().to(device)
    
    # Input accounts for the 4 stacked states of 18 features each
    dummy_input = torch.zeros(1, 18 * STACK_SIZE, device=device)
    
    # Generate Actor Diagram
    out_actor = actor(dummy_input)
    dot_actor = make_dot(out_actor, params=dict(actor.named_parameters()))
    dot_actor.render("vpg_actor_architecture", format="png", cleanup=True)
    
    # Generate Critic Diagram
    out_critic = critic(dummy_input)
    dot_critic = make_dot(out_critic, params=dict(critic.named_parameters()))
    dot_critic.render("vpg_critic_architecture", format="png", cleanup=True)
    
    print("Saved VPG architecture diagrams to vpg_actor_architecture.png and vpg_critic_architecture.png")