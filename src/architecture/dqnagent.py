import random

import torch
import numpy as np

from dqn import FlatQnet
from torch.nn.utils import clip_grad_norm_
from goal_sampler import GoalSampler
from replay_buffer import ReplayBuffer, Transition
from utils import flatten


class DQNAgent:
    def __init__(self, model, language_model, params, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.policy_network = model(**params['model_params'])
        self.policy_network.to(device)

        self.target_network = model(**params['model_params'])
        self.target_network.to(device)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        self.target_network.eval()

        self.steps = 0
        self.exploration_params = params['exploration_params']

        self.goal_sampler = GoalSampler(language_model=language_model, **params['goal_sampler_params'])

        self.language_model = language_model
        self.replay_buffer = ReplayBuffer(**params['replay_buffer_params'])

        self.batch_size = params['batch_size']
        self.discount_factor = params['discount_factor']
        self.loss = params['loss']
        self.optimizer = params['optimizer'](self.policy_network.parameters(), **params['optimizer_params'])
        self.device = device

    def store_transitions(self, *args):
        self.replay_buffer.store(*args)

    def store_final_transitions(self, achieved_goals, failed_goals, state, action, next_state, hidden_state):
        for g in achieved_goals:
            self.store_transitions(g, state, action, next_state, True, 1, hidden_state)
        for g in failed_goals:
            self.store_transitions(g, state, action, next_state, True, 0, hidden_state)

    def sample_goal(self, strategy=None):
        return self.goal_sampler.sample_goal(strategy=strategy)

    def _embed_one_action(self, a):
        # embedding = a.get_node_embedding()
        # if embedding is None and a.has_description:
        #     # if action has no embedding already, it should have a description
        #     # We then use the language policy_network to compute an embedding
        #     a.node_embedding = self.language_model(a.description)
        # else:
        #     raise NotImplementedError
        embedding = torch.tensor(a.get_node_embedding())
        embedding = embedding.view(1, -1)
        return embedding.float().to(self.device)

    # TODO adapt to a policy_network of actions, for now suppose that possible attributes are 'embedding' 'description' 'has_description'
    def embed_actions(self, actions):
        """
        Transform the actions into their standard embedding using a language policy_network for action with description
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
            return action_embeddings
            # return torch.cat(action_embeddings, dim=0).to(self.device)

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
                Q, normalized_action_embedding = self.policy_network(state=state.to(self.device),
                                                                     instruction=instruction.to(self.device),
                                                                     actions=[embedded_actions],
                                                                     hidden_state=hidden_state.to(self.device))
                action_idx = Q.argmax(1).item()
                # hidden_state += hidden_state.view(*self.hidden_state.size())
        else:
            action_idx = random.randint(0, len(actions) - 1)
            embedded_actions = self.embed_actions(actions)
            normalized_action_embedding = self.policy_network.project_action_embedding([embedded_actions])

        action = actions[action_idx]
        # Workaround to isintance(self.policy_network, FlatQnet) that return False for a weird reason
        if self.policy_network._get_name() == 'FlatQnet':
            hidden_state = hidden_state.to(self.device)
            hidden_state += normalized_action_embedding.squeeze()[action_idx]  # TODO Check

        return action, hidden_state

    def udpate_policy_net(self, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        if batch_size > len(self.replay_buffer):
            return

        self.optimizer.zero_grad()

        transitions = self.replay_buffer.sample(batch_size)
        transitions = Transition(*zip(*transitions))

        goals = torch.cat([g.goal_embedding for g in transitions.goal])

        # states = zip(*transitions.state)
        states = torch.cat(transitions.state).to(self.device)

        actions = [[a] for a in transitions.action]
        embedded_actions = self.embed_actions(actions)

        next_states, next_av_actions = zip(*transitions.next_state)
        next_states = torch.cat(next_states).to(self.device)
        next_av_actions_embedded = self.embed_actions(list(next_av_actions))

        done = torch.cat(transitions.done).to(self.device)
        rewards = torch.cat(transitions.reward).to(self.device)
        hidden_states = torch.cat(transitions.hidden_states).to(self.device)

        Q_sa, normalized_action_embedding = self.policy_network(state=states,
                                                                instruction=goals,
                                                                actions=embedded_actions,
                                                                hidden_state=hidden_states)
        next_hidden_states = normalized_action_embedding.squeeze()

        maxQ = torch.zeros(batch_size, device=self.device)
        maxQ[done], _ = self.target_network(state=next_states[done],
                                            instruction=goals[done],
                                            actions=[a for a, mask in zip(next_av_actions_embedded, done) if mask],
                                            hidden_states=next_hidden_states[done])
        maxQ = maxQ.max(1)

        expected_value = rewards + self.discount_factor * maxQ
        loss = self.loss(expected_value.detach(), Q_sa)

        loss.backward()
        clip_grad_norm_(self.policy_network.parameters(), 1)
        self.optimizer.step()

    def update_target_net(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())
