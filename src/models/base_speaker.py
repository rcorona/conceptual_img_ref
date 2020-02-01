import torch
import torch.nn as nn
import sys


class RandomSpeaker(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self._n_features = n_features

    def init(self, batch_sz):
        return 0, (0, 0)

    def loss(self, Q, rewards):
        q_loss = torch.tensor(0.)
        return q_loss

    def step(self, agent_embedding, agent_cell, features, rewards, eval_true):
        # Select attribute to send to listener.
        attrs = torch.LongTensor(features.shape[0]).random_(
            0, features.shape[1]).view(-1, 1)

        # Compute reward for listener's guess.
        guess_reward = rewards.gather(1, attrs)

        # Dummy results
        Q = torch.zeros_like(features)

        return Q, attrs, guess_reward, 0, (0, 0)


class ReactiveSpeaker(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self._n_features = n_features

    def _update_choices(self, mask=None):
        try:
            new_choice = torch.multinomial(self.seen_attributes, 1)
        except:
            print(self.seen_attributes)
            sys.exit()

        if mask is None:
            self.curr_choice = new_choice
        else:
            self.curr_choice[mask] = new_choice[mask]
        batch_sz = self.seen_attributes.size(0)
        self.seen_attributes[range(batch_sz), self.curr_choice.squeeze()] = 0

        # Re-initialize any game sequences which might have exhausted all attributes.
        exhausted = (self.seen_attributes.sum(1) == 0.0).nonzero()
        if exhausted.size(0) > 0:
            e_idx = exhausted.squeeze()

            new_seen = torch.ones(
                (self.seen_attributes.size(0), self._n_features))
            self.seen_attributes[e_idx, :] = new_seen[e_idx, :]

    def init(self, batch_sz):
        self.seen_attributes = torch.ones((batch_sz, self._n_features))
        self._update_choices()
        return 0, (0, 0)

    def loss(self, Q, rewards):
        q_loss = torch.tensor(0.)
        return q_loss

    def step(self, agent_embedding, agent_cell, features, rewards, eval_true):
        # Select attribute to send to listener.
        attrs = self.curr_choice

        # Compute reward for listener's guess.
        guess_reward = rewards.gather(1, attrs)

        # Update choice
        neg_rewards = guess_reward == -1
        if neg_rewards.sum() > 0:
            self._update_choices(neg_rewards)

        # Dummy results
        Q = torch.zeros_like(self.seen_attributes)

        return Q, attrs, guess_reward, 0, (0, 0)


class ReactiveMaxSpeaker(ReactiveSpeaker):
    def init(self, batch_sz):
        self.seen_attributes = torch.ones((batch_sz, self._n_features))
        return 0, (0, 0)

    def _update_choices(self, mask=None):
        if mask is None:
            self.curr_choice = (
                self._features * self.seen_attributes).argmax(dim=1, keepdim=True)
        else:
            batch_sz = self.seen_attributes.size(0)
            self.seen_attributes[torch.arange(
                batch_sz)[mask.squeeze()], self.curr_choice[mask]] = 0

    def step(self, agent_embedding, agent_cell, features, rewards, eval_true):
        self._features = features.abs()
        self._update_choices()
        return super().step(agent_embedding, agent_cell, features, rewards, eval_true)
