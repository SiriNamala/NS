import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.observations = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__, reverse=True)

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.observations[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, nagents, state_dim, action_dim, has_continuous_action_space, action_std):
        super(Actor, self).__init__()

        size = 256

        self.actor = nn.Sequential(
            nn.Linear(state_dim, size),
            # nn.Tanh(),
            nn.Linear(size, 2 * size),
            # nn.Tanh(),
            nn.Linear(2 * size, size),
            # nn.Tanh(),
            nn.Linear(size, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(self.size, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 1),
        )

    def forwardActor(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution

    def forwardCritic(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value


class ActorCritic:
    def __init__(self, nagents, state_dim, action_dim, lr_actor, lr_critic, gamma, has_continuous_action_space,
                 action_std_init=0.6, device=device):
        self.has_continuous_action_space = has_continuous_action_space

        self.buffer = RolloutBuffer()

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        # self.eps_clip = eps_clip
        # self.K_epochs = K_epochs

        # self.buffer = RolloutBuffer()

        self.pbuffer = dict()  # permanant buffer

        self.policy = ActorCritic(nagents, state_dim, action_dim, has_continuous_action_space, action_std_init)

        # self.optimizer = torch.optim.Adam([
        # {'params': self.policy.actor.parameters(), 'lr': lr_actor},
        # {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        # ], amsgrad=True)  # , weight_decay=1e-4, betas=(0.85, 0.99)

        self.MseLoss = nn.MSELoss()

        self.policy = Actor(nagents, state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy1 = Actor(nagents, state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)

        actor_optimizer = optim.Adam(self.policy.actor.parameters())
        critic_optimizer = optim.Adam(self.policy1.critic.parameters())

        self.policy_old = ActorCritic(nagents, state_dim, action_dim, has_continuous_action_space, action_std_init,
                                      device=device)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def select_action(self, obs, index):
        state = obs[index]

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            obs = torch.FloatTensor(obs.flatten()).to(device)
            action, action_logprob = self.policy_old.act(state)
        self.buffer.observations.append(obs)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.item()

    def return_rewards(self, next_value, rewards, masks, gamma):
        reward = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            reward = rewards[step] + gamma * reward * masks[step]
            returns.insert(0, reward)
        return returns
