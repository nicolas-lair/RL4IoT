import random

import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_

from logger import rootLogger
from architecture.contextnet import DeepSetStateNet, FlatStateNet, AttentionFlatState
from architecture.goal_sampler import GoalSampler
from architecture.replay_buffer import ReplayBuffer, Transition
from architecture.utils import dict_to_device
from architecture.dqn import FullNet

logger = rootLogger.getChild(__name__)


# logger.setLevel(10)


class DQNAgent:
    def __init__(self, language_model, params):
        self.device = params['device']

        # self.policy_network = model(**params['model_params'])
        self.policy_network = FullNet(**params['model_params'])
        self.policy_network.to(self.device)

        self.target_network = FullNet(**params['model_params'])
        self.target_network.to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_update_freq = params['target_update_frequence']

        self.target_network.eval()

        self.exploration_params = params['exploration_params']
        self.exploration_threshold = self.exploration_params['start_eps']

        self.goal_sampler = GoalSampler(language_model=language_model, **params['goal_sampler_params'])

        self.language_model = language_model.to(self.device)
        self.language_model.device = self.device

        self.replay_buffer = ReplayBuffer(**params['replay_buffer_params'])

        self.batch_size = params['batch_size']
        self.discount_factor = params['discount_factor']

        if params['loss'] == 'mse':
            self.loss = torch.nn.functional.mse_loss
        elif params == 'smooth_l1':
            self.loss = torch.nn.functional.smooth_l1_loss
        else:
            raise NotImplementedError

        self.optimizer = params['optimizer'](
            list(self.policy_network.parameters()) + list(self.language_model.parameters()),
            **params['optimizer_params'])

        self.update_counter = 0

    def store_transitions(self, **kwargs):
        self.replay_buffer.store(**kwargs)

    def store_final_transitions(self, achieved_goals, failed_goals, **kwargs):
        for g in achieved_goals:
            self.store_transitions(goal=g, done=True, reward=1, **kwargs)
        for g in failed_goals:
            self.store_transitions(goal=g, done=True, reward=0, **kwargs)

    def sample_goal(self, strategy=None):
        return self.goal_sampler.sample_goal(strategy=strategy)

    def _embed_one_action(self, a):
        """

        :param a: object of Node Type
        :return:
        """
        # embedding = a.get_node_embedding()
        # if embedding is None and a.has_description:
        #     # if action has no embedding already, it should have a description
        #     # We then use the language policy_network to compute an embedding
        #     a.node_embedding = self.language_model(a.description)
        # else:
        #     raise NotImplementedError
        embedding = a.get_node_embedding()
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding)
        logger.debug(f'View debuggin: {embedding.size()}')
        embedding = embedding.view(1, -1)
        return embedding.float().to(self.device), a.node_type

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
            action_embedding, action_type = zip(*[self.embed_actions(a) for a in actions])
            return action_embedding, action_type
        else:
            action_embeddings = [self._embed_one_action(a) for a in actions]
            action_embedding, action_type = zip(*action_embeddings)

        return action_embedding, action_type
        # return torch.cat(action_embeddings, dim=0).to(self.device)

    def select_action(self, state, instruction, actions, hidden_state, exploration):
        """
        select an action among a list of actions and update
        :param state:
        :param instruction:
        :param actions: list of action
        :return:
        """
        sample = random.random() if exploration else 1
        if sample > self.exploration_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                embedded_actions, action_type = self.embed_actions(actions)

                Q, normalized_action_embedding = self.policy_network(state=dict_to_device(state, self.device),
                                                                     instruction=instruction.to(self.device),
                                                                     actions=([embedded_actions], [action_type]),
                                                                     hidden_state=hidden_state.to(self.device))
                action_idx = Q.argmax(1).item()
                # hidden_state += hidden_state.view(*self.hidden_state.size())
        else:
            action_idx = random.randint(0, len(actions) - 1)
            embedded_actions, action_type = self.embed_actions(actions)
            normalized_action_embedding = self.policy_network.action_projector([embedded_actions],
                                                                               [action_type])

        action = actions[action_idx]
        if isinstance(self.policy_network.context_net, (FlatStateNet, AttentionFlatState, DeepSetStateNet)):
            hidden_state = hidden_state.to(self.device)
            # hidden_state += normalized_action_embedding.squeeze()[action_idx]  # TODO Check
            hidden_state = normalized_action_embedding[:, action_idx]
        else:
            raise NotImplementedError

        return action, hidden_state

    def update_policy_net(self, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        if batch_size > len(self.replay_buffer):
            return

        self.optimizer.zero_grad()

        transitions = self.replay_buffer.sample(batch_size)
        transitions = Transition(*zip(*transitions))

        logger.debug('Computing goal embeddings for update')
        # goals = torch.cat([g.compute_embedding(self.language_model) for g in transitions.goal]).to(self.device)
        goals = self.language_model([g.goal_string for g in transitions.goal])

        states = [dict_to_device(d, self.device) for d in transitions.state]

        actions = [[a] for a in transitions.action]

        logger.debug('Embedding actions')
        embedded_actions = self.embed_actions(actions)

        rewards = torch.tensor(transitions.reward).to(self.device)
        hidden_states = torch.cat(transitions.hidden_state).to(self.device)

        previous_action = [[a] for a in transitions.previous_action]
        logger.debug('Embedding previous action')
        embedded_previous_action = self.embed_actions(previous_action)

        logger.debug('Projecting previous action in a command subspace')
        # TODO try detach the hidden state
        projected_previous_actions = self.policy_network.action_projector(*embedded_previous_action).squeeze()

        logger.debug('Computing Q(s,a)')
        Q_sa, normalized_action_embedding = self.policy_network(state=states,
                                                                instruction=goals,
                                                                actions=embedded_actions,
                                                                hidden_state=projected_previous_actions)
        logger.debug(f'Done: {Q_sa.squeeze()}')
        next_hidden_states = normalized_action_embedding.squeeze()

        next_states, next_av_actions = zip(*transitions.next_state)
        next_states = [dict_to_device(d, self.device) for d in next_states]
        done = torch.tensor(transitions.done)  # .to(self.device)

        logger.debug('Embedding next available actions')
        maxQ = torch.zeros(batch_size, device=self.device)
        if not done.all():
            next_av_actions_embedded = self.embed_actions([a for a, mask in zip(next_av_actions, done) if not mask])

            logger.debug('Computing max(Q(s`, a)) over a')
            with torch.no_grad():
                Q_target_net, _ = self.target_network(state=[s for s, flag in zip(next_states, done.numpy()) if not flag],
                                                  instruction=goals[~done],
                                                  actions=next_av_actions_embedded,
                                                  hidden_state=next_hidden_states[~done])
            maxQ[~done] = Q_target_net.max(1).values.squeeze()
            logger.debug(f'Done: {maxQ}')

        logger.debug('Computing loss')
        expected_value = rewards + self.discount_factor * maxQ
        loss = self.loss(expected_value.detach(), Q_sa.squeeze())
        logger.debug(f'Done: {loss}')

        logger.debug('Backward pass')
        loss.backward()

        logger.debug('Clipping gradient norm')
        # Takes time to clip
        clip_grad_norm_(self.policy_network.parameters(), 1)

        logger.debug('Optimising step')
        self.optimizer.step()

        self.update_counter += 1

    def update_target_net(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def update_exploration_function(self, n):
        """

        :param n: episode number
        :return:
        """
        self.exploration_threshold = self.exploration_params['min_eps'] + (
                self.exploration_params['start_eps'] - self.exploration_params['min_eps']) * np.exp(
            -1. * n / self.exploration_params['eps_decay'])

    def update(self, n):
        """

        :param n: episode number
        :return:
        """
        logger.debug('Update of policy net')
        self.update_policy_net()
        logger.debug('done')
        self.goal_sampler.update_embedding()
        self.update_exploration_function(n=n + 1)
        if n % self.target_update_freq == 0:
            logger.debug('Update of target net')
            self.update_target_net()
            logger.debug('done')
