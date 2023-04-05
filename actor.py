import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the neural network architecture for the actor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x

    def sample(self, state):
        mean = self.forward(state)
        std = torch.zeros_like(mean).fill_(self.max_action * 0.1)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        action = torch.tanh(action)
        return action * self.max_action, log_prob
