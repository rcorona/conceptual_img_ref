import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.categorical import Categorical


class Speaker(nn.Module):

    def __init__(self, n_features, mid_dim, embed_agents, policy_type='epsilon_greedy',
                 epsilon_greedy=0.1, eval_epsilon_greedy=0.0):
        super(Speaker, self).__init__()

        self._embed_agents = embed_agents
        self._epsilon_greedy = epsilon_greedy
        self._eval_epsilon_greedy = eval_epsilon_greedy
        self._n_features = n_features
        self._mid_dim = mid_dim
        self._n_layers = 2
        self._policy_type = policy_type

        # Used to generate agent embedding.
        self._lstm = nn.LSTM(n_features, mid_dim,
                             batch_first=True, num_layers=self._n_layers)

        # Q-learning.
        self._Q = nn.Sequential(
            nn.Linear(n_features + mid_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ELU(),
            nn.Linear(mid_dim, n_features)
        )

        self._h = nn.Parameter(torch.empty(
            (self._n_layers, 1, mid_dim,), requires_grad=True))
        nn.init.uniform_(self._h, -0.1, 0.1)

        self._c = nn.Parameter(torch.empty(
            (self._n_layers, 1, mid_dim,), requires_grad=True))
        nn.init.uniform_(self._c, -0.1, 0.1)

        # Used to embed state.
        self._state = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.BatchNorm1d(n_features),
            nn.ELU(),
            nn.Linear(n_features, n_features),
            nn.BatchNorm1d(n_features),
            nn.ELU()
        )

        # Attribute selection policy.
        self._selection_policy = nn.Sequential(
            nn.Linear(n_features + mid_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ELU(),
            nn.Linear(mid_dim, n_features),
            nn.Softmax(dim=1)
        )

        self._log_probs = None

        # Epsilon-greedy attribute selection policy.
        self._epsilon = Bernoulli(torch.tensor([epsilon_greedy]))
        self._eval_epsilon = Bernoulli(torch.tensor([eval_epsilon_greedy]))

    def init(self, batch_sz):

        # Initialize LSTM.
        h = self._h.repeat(1, batch_sz, 1)
        c = self._c.repeat(1, batch_sz, 1)

        # Prob list for policy training.
        self._log_probs = []

        # Initial hidden state is initial agent embedding.
        return h[-1, :, :], (h, c)

    def concat_feature(self, agent_embedding, img_features):

        # Form single embedding from img features and agent embedding.
        if not self._embed_agents:
            agent_embedding = torch.zeros(
                agent_embedding.shape).to(agent_embedding.device)

        state = torch.cat(
            (agent_embedding, img_features.to(agent_embedding.device)), dim=1)

        return state

    def attr_selection(self, Q=None, state_feature=None, eval=False):

        # Epsilon-greedy or curiosity driven exploration.
        if self._policy_type == 'epsilon_greedy' or eval:
            return self.epsilon_greedy_selection(Q, eval)
        else:
            return self.parameterized_selection(state_feature, eval)

    def parameterized_selection(self, state_feature, eval=False):

        probs = self._selection_policy(state_feature.detach())
        dist = Categorical(probs)
        selection = dist.sample()

        log_probs = dist.log_prob(selection)
        self._log_probs.append(log_probs.view(-1, 1))

        return selection.view(-1, 1)

    def epsilon_greedy_selection(self, Q, eval=False):

        # Random choice
        random = np.random.choice(self._n_features, Q.size(0))
        random = torch.tensor(random).view(-1, 1).to(Q.device)

        # Greedy choice.
        greedy = torch.argmax(Q, 1).view(-1, 1)

        # Epsilon greedy policy.
        if eval:
            epsilon = self._eval_epsilon
        else:
            epsilon = self._epsilon

        choices = torch.cat([greedy, random], 1)
        idx = epsilon.sample((Q.size(0),)).to(Q.device).long()
        selection = choices.gather(1, idx)

        return selection

    def update_agent_embedding(self, one_hot, agent_cell):

        agent_embedding, agent_cell = self._lstm(
            one_hot.unsqueeze(1), agent_cell)

        return agent_embedding.squeeze(), agent_cell

    def loss(self, Q, rewards):  # , Qs, Qa, rewards):

        # Compute Q-function loss over all examples
        q_loss = F.mse_loss(Q, rewards, reduction='none')

        return q_loss

    def active_loss(self, q_loss_eval):

        self._log_probs = torch.cat(self._log_probs, -1)
        active_loss = -self._log_probs * -q_loss_eval.detach()

        loss = active_loss.mean()

        return loss

    def curiosity_loss(self, q_loss):
        self._log_probs = torch.cat(self._log_probs, -1)
        curiosity_loss = -self._log_probs * q_loss.detach()

        loss = curiosity_loss.mean()
        return loss

    def step(self, agent_embedding, agent_cell, features, rewards, eval_true):

        # First embed state.
        state_feature = self._state(features.to(
            agent_embedding.device))  # .detach()

        # Combine agent and state embedding.
        combined_feature = self.concat_feature(agent_embedding, state_feature)

        # First generate Q-values for attributes.
        Q = self._Q(combined_feature)

        # Select attribute to send to listener.
        attrs = self.attr_selection(Q, combined_feature, eval_true).to('cpu')

        # Compute reward for listener's guess.
        guess_reward = rewards.gather(1, attrs)

        # Update agent embedding.
        one_hot = torch.zeros((attrs.size(0), features.size(-1)))
        one_hot[np.arange(attrs.size(0)), attrs.view(-1)
                ] = guess_reward.squeeze()

        agent_embedding, agent_cell = self.update_agent_embedding(
            one_hot.to(Q.device), agent_cell)

        return Q, attrs, guess_reward, agent_embedding, agent_cell
