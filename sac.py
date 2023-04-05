import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from actor import Actor
from critic import Critic
from replay_buffer import ReplayBuffer

# Define the Soft Actor-Critic (SAC) class
class SAC:
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate,
        gamma,
        tau,
        buffer_size,
        batch_size,
        device,
        env,
        num_episodes,
        save_interval,
        evaluation_interval,
        save_path,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.env = env
        self.num_episodes = num_episodes
        self.save_interval = save_interval
        self.evaluation_interval = evaluation_interval
        self.save_path = save_path
    
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.target_critic = Critic(state_dim, action_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.max_action = max_action
        self.device = device
        self.alpha = 0.2
        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        
        self.iter = 0

    def train(self, replay_buffer, batch_size=64, discount=0.99, tau=0.005, policy_freq=2):
        # Sample a batch of experiences from the replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        # Compute the target Q-values
        with torch.no_grad():
            next_action, log_prob_next_action = self.actor.sample(next_state)
            target_q1, target_q2 = self.target_critic(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * log_prob_next_action
            target_q = reward + (1 - done) * discount * target_q

        # Update the critic networks
        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update the actor network and the temperature
        if self.iter % policy_freq == 0:
            new_action, log_prob_new_action = self.actor.sample(state)
            q1_new, _ = self.critic(state, new_action)
            actor_loss = (self.alpha * log_prob_new_action - q1_new).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            alpha_loss = (self.log_alpha * (-log_prob_new_action - self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()

            # Update the target critic networks
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        self.iter += 1

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def update(self, buffer, batch_size=64, discount=0.99, tau=0.005, policy_freq=2):
        # Update the SAC components using a batch of data from the replay buffer
        self.train(buffer, batch_size=batch_size, discount=discount, tau=tau, policy_freq=policy_freq)

    def save(self, save_path):
        # Save the model parameters
        torch.save(self.actor.state_dict(), f"{save_path}/actor.pth")
        torch.save(self.critic.state_dict(), f"{save_path}/critic.pth")
        torch.save(self.target_critic.state_dict(), f"{save_path}/target_critic.pth")
