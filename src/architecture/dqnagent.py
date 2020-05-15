import random
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from dqn import *
from goal_sampler import GoalSampler
from replay_buffer import ReplayBuffer, Transition


class Action:
    def __init__(self, description, embedding):
        if description is not None:
            self.description = description
            self.has_description = True
        if embedding is not None:
            self.embedding = embedding


class DQNAgent:
    def __init__(self, model, exploration_params, language_model, replay_buffer_params, params, device,
                 goal_sampler=None):
        self.model = model
        self.steps = 0
        self.exploration_params = exploration_params

        if goal_sampler is None:
            self.goal_sampler = GoalSampler(language_model=language_model)

        self.language_model = language_model
        self.replay_buffer = ReplayBuffer(**replay_buffer_params)

        self.discount_factor = params['discount_factor']
        self.loss = params['loss']
        self.optimizer = params['optimizer'](self.model.parameters(), **params['optimizer_params'])
        self.device = device

    def store_transitions(self, *args):
        self.replay_buffer.store(*args)

    def sample_goal(self, strategy=None):
        return self.goal_sampler.sample_goal(strategy=strategy)

    def _embed_one_action(self, a):
        assert isinstance(a, Action)
        if a.embedding is not None:
            # NL embedding already computed or other types of embedding
            embedding = a.embedding
        elif a.has_description:
            # if action has no embedding already, it should have a description
            # We then use the language model to compute an embedding
            a.embedding = self.language_model(a.description)
            embedding = a.embedding
        else:
            raise NotImplementedError
        return embedding

    # TODO adapt to a model of actions, for now suppose that possible attributes are 'embedding' 'description' 'has_description'
    def embed_actions(self, actions):
        """
        Transform the actions into their standard embedding using a language model for action with description
        or other types of predefined embedding. The embeddings may not have all the same size and will later be
        projected in a common embedding space.
        :param actions: list of torch tensors (possible actions)
        :return:
        """
        assert isinstance(actions, list)
        if isinstance(actions[0], list):
            return [self.embed_actions(a) for a in actions]
        else:
            action_embeddings = [self._embed_one_action(a) for a in actions]
            return torch.cat(action_embeddings, dim=0).to(self.device)

    def select_action(self, state, instruction, actions, hidden_state):
        """
        select an action among a list of actions and update
        :param state:
        :param instruction:
        :param actions: list of action
        :return:
        """
        sample = random.random()
        eps_threshold = self.exploration_params['min_eps'] + (
                self.exploration_params['start_eps'] - self.exploration_params['min_eps']) * np.exp(
            -1. * self.steps / self.exploration_params['eps_decay'])
        self.steps += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                embedded_actions = self.embed_actions(actions)
                Q, normalized_action_embedding = self.model(state.to(self.device),
                                                            instruction.to(self.device),
                                                            embedded_actions,
                                                            hidden_state.to(self.device))
                action_idx = Q.argmax(1).item()
                # hidden_state += hidden_state.view(*self.hidden_state.size())
        else:
            action_idx = random.randint(0, len(actions))

        action = actions[action_idx]
        if isinstance(self.model, FlatCritic):
            hidden_state += normalized_action_embedding.squeeze()[action_idx]  # TODO Check

        return action, hidden_state

    def udpate_qnet(self, update_params):
        self.optimizer.zero_grad()

        transitions = self.replay_buffer.sample(update_params['update_batch_size'])
        transitions = Transition(*zip(*transitions))

        goals = torch.cat([g.goal_embedding for g in transitions.goal])

        states, _ = zip(*transitions.state)
        states = torch.cat(states).to(self.device)

        actions = [[a] for a in transitions.action]
        embedded_actions = self.embed_actions(actions)

        next_states, next_av_actions = zip(*transitions.next_state)
        next_states = torch.cat(next_states).to(self.device)
        next_av_actions_embedded = self.embed_actions(list(next_av_actions))

        done = torch.cat(transitions.done).to(self.device)
        rewards = torch.cat(transitions.reward).to(self.device)
        hidden_states = torch.cat(transitions.hidden_states).to(self.device)

        Q_sa, normalized_action_embedding = self.model(state=states,
                                                       instruction=goals,
                                                       actions=embedded_actions,
                                                       hidden_state=hidden_states)
        next_hidden_states = normalized_action_embedding.squeeze()

        maxQ = torch.zeros(update_params['update_batch_size'], device=self.device)
        maxQ[done], _ = self.model(state=next_states[done],
                                   instruction=goals[done],
                                   actions=[a for a, mask in zip(next_av_actions_embedded, done) if mask],
                                   hidden_states=next_hidden_states[done])
        maxQ = maxQ.max(1)

        expected_value = rewards + self.discount_factor * maxQ
        loss = self.loss(expected_value.detach(), Q_sa)

        loss.backward()
        clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
