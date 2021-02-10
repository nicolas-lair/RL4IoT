import random

import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_

from logger import get_logger
from architecture.goal_sampler import GoalSampler
from architecture.replay_buffer import get_replay_buffer, Transition
from architecture.utils import dict_to_device
from architecture.dqn import FullNet
from action_embedder import ActionModel
from architecture.state_embedder import StateEmbedder, get_description_embedder

logger = get_logger(__name__)


class DQNAgent:
    def __init__(self, language_model, params, env_discrete_params):
        self.device = params['device']

        self.policy_network = FullNet(**params['model_params'], discrete_params=env_discrete_params)

        self.target_network = FullNet(**params['model_params'], discrete_params=env_discrete_params)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_update_freq = params['target_update_frequence']
        self.target_network.eval()

        self.policy_network.to(self.device)
        self.target_network.to(self.device)

        self.exploration_params = params['exploration_params']
        self.exploration_threshold = self.exploration_params['start_eps']

        self.goal_sampler = GoalSampler(language_model=language_model, **params['goal_sampler_params'])

        self.action_model = ActionModel(raw_action_size=params['model_params']['raw_action_size'],
                                        out_features=params['model_params']['action_embedding_size'],
                                        env_discrete_params=env_discrete_params)
        self.action_model.to(self.device)

        self.language_model = language_model.to(self.device)
        self.language_model.device = self.device

        self.replay_buffer = get_replay_buffer(**params['replay_buffer_params'])
        self.per = params['replay_buffer_params']['per']

        self.batch_size = params['batch_size']
        self.discount_factor = params['discount_factor']
        self.double_dqn = params['use_double_dqn']

        if params['loss'] == 'mse':
            self.loss = torch.nn.functional.mse_loss
        elif params == 'smooth_l1':
            self.loss = torch.nn.functional.smooth_l1_loss
        else:
            raise NotImplementedError

        self.optimizer = params['optimizer'](
            list(self.policy_network.parameters()) + list(
                self.language_model.parameters()) + list(self.action_model.parameters()),
            **params['optimizer_params'])

        if params['lr_scheduler'] is not None:
            self.lr_scheduler = params['lr_scheduler'](self.optimizer, **params['lr_scheduler_params'])
        else:
            self.lr_scheduler = None

        self.update_counter = 0

    def store_transitions(self, **kwargs):
        self.replay_buffer.store(**kwargs)

    def store_transitions_with_oracle_feedback(self, achieved_goals_str, done, **kwargs):
        failed_goals = self.goal_sampler.get_failed_goals(achieved_goals_str=achieved_goals_str)
        achieved_goals = [self.goal_sampler.discovered_goals[g] for g in achieved_goals_str]
        for g in achieved_goals:
            self.store_transitions(goal=g, done=True, reward=1, **kwargs)
        for g in failed_goals:
            self.store_transitions(goal=g, done=done, reward=0, **kwargs)

    def sample_goal(self, strategy=None):
        return self.goal_sampler.sample_goal(strategy=strategy)

    def _embed_one_action(self, a):
        """

        :param a: object of Node Type
        :return:
        """
        embedding = a.get_node_embedding()
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding)
        logger.debug(f'View debugging: {embedding.size()}')
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
        actions = [actions] if not isinstance(actions[0], list) else actions
        temp = [[self._embed_one_action(a) for a in action_list] for action_list in actions]
        action_embedding, action_type = zip(*[zip(*t) for t in temp])
        embedded_actions = self.action_model(action_embedding, action_type)
        return embedded_actions
        # if isinstance(actions[0], list):
        #     action_embedding, action_type = zip(*[self.embed_actions(a) for a in actions])
        #     return action_embedding, action_type
        # else:
        #     action_embeddings = [self._embed_one_action(a) for a in actions]
        #     action_embedding, action_type = zip(*action_embeddings)

        # return action_embedding, action_type

    def get_greedy_action(self, state, instruction, actions, hidden_state):
        # embedded_actions, action_type = self.embed_actions(actions)
        Q_values = self.policy_network(state=dict_to_device(state, self.device),
                                       instruction=instruction.to(self.device),
                                       # actions=(embedded_actions, action_type),
                                       actions=actions,
                                       hidden_state=hidden_state.to(self.device))
        action_idx = Q_values.argmax(1)
        # action = actions[action_idx]
        # new_hidden_state = actions[:, action_idx]
        return action_idx

    def select_action(self, state, instruction, actions, hidden_state, exploration):
        """
        select an action among a list of actions and update
        :param state:
        :param instruction:
        :param actions: list of action or list of list of actions
        :return:
        """
        sample = random.random() if exploration else 1

        embedded_actions = self.embed_actions(actions)
        if sample >= self.exploration_threshold:
            with torch.no_grad():
                action_idx = self.get_greedy_action(state, instruction, embedded_actions, hidden_state)
                action_idx = action_idx.item()
                # # t.max(1) will return largest column value of each row.
                # # second column on max result is index of where max element was
                # # found, so we pick action with the larger expected reward.
                # embedded_actions, action_type = self.embed_actions(actions)
                #
                # Q, normalized_action_embedding = self.policy_network(state=dict_to_device(state, self.device),
                #                                                      instruction=instruction.to(self.device),
                #                                                      actions=(embedded_actions, action_type),
                #                                                      hidden_state=hidden_state.to(self.device))
                # action_idx = Q.argmax(1).item()
                # # hidden_state += hidden_state.view(*self.hidden_state.size())
        else:
            action_idx = random.randint(0, len(actions) - 1)

        action = actions[action_idx]
        hidden_state = embedded_actions[:, action_idx]
        # embedded_actions, action_type = self.embed_actions(actions)
        # normalized_action_embedding = self.policy_network.action_projector([embedded_actions],
        #                                                                    [action_type])

        # if isinstance(self.policy_network.context_net, (FlatStateNet, AttentionFlatState, DeepSetStateNet)):
        #     hidden_state = hidden_state.to(self.device)
        #     # hidden_state += normalized_action_embedding.squeeze()[action_idx]  # TODO Check
        #     hidden_state = normalized_action_embedding[:, action_idx]
        # else:
        #     raise NotImplementedError

        return action, hidden_state

    def update_policy_net(self, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        if batch_size > len(self.replay_buffer):
            return

        self.optimizer.zero_grad()

        indices, transitions, weights = self.replay_buffer.sample(batch_size)
        transitions = Transition(*zip(*transitions))

        logger.debug('Computing goal embeddings for update')
        goals = self.language_model([g.goal_string for g in transitions.goal])

        logger.debug('Embedding actions')
        actions = [[a] for a in transitions.action]
        embedded_actions = self.embed_actions(actions)

        previous_action = [[a] for a in transitions.previous_action]
        logger.debug('Embedding previous action')
        # TODO try detach the hidden state
        embedded_previous_action = self.embed_actions(previous_action).squeeze()
        # projected_previous_actions = self.policy_network.action_projector(*embedded_previous_action).squeeze()

        logger.debug('Computing Q(s,a)')
        Q_sa = self.policy_network(state=dict_to_device(transitions.state, self.device),
                                   instruction=goals,
                                   actions=embedded_actions,
                                   hidden_state=embedded_previous_action)
        logger.debug(f'Done: {Q_sa.squeeze()}')
        next_hidden_states = embedded_actions.squeeze()

        done = torch.tensor(transitions.done)  # .to(self.device)
        maxQ = torch.zeros(batch_size, device=self.device)
        if not done.all():
            next_states, next_av_actions = zip(*transitions.next_state)
            next_states = dict_to_device([s for s, mask in zip(next_states, done) if not mask], self.device)
            next_av_actions = [a for a, mask in zip(next_av_actions, done) if not mask]
            logger.debug('Embedding next available actions')
            embedded_next_actions = self.embed_actions(next_av_actions)
            if self.double_dqn:
                next_action_idx = self.get_greedy_action(state=next_states,
                                                         instruction=goals[~done],
                                                         actions=embedded_next_actions,
                                                         hidden_state=next_hidden_states[~done])


                embedded_next_actions = embedded_next_actions.gather(dim=1, index=torch.stack(
                    [next_action_idx.view(-1)] * (embedded_next_actions.size(2)), dim=1).unsqueeze(dim=1))

            next_q = self.target_network(state=next_states,
                                         instruction=goals[~done],
                                         actions=embedded_next_actions,
                                         hidden_state=next_hidden_states[~done])

            # next_states = dict_to_device([s for s, mask in zip(next_states, done) if not mask], self.device)
            # next_av_actions_embedded = self.embed_actions([a for a, mask in zip(next_av_actions, done) if not mask])
            #
            # logger.debug('Computing max(Q(s`, a)) over a')
            # with torch.no_grad():
            #     next_q = self.target_network(state=next_states,
            #                                  instruction=goals[~done],
            #                                  actions=next_av_actions_embedded,
            #                                  hidden_state=next_hidden_states[~done])
            maxQ[~done] = next_q.max(1).values.squeeze()
            logger.debug(f'Done: {maxQ}')

        logger.debug('Computing loss')
        rewards = torch.tensor(transitions.reward).to(self.device)
        expected_value = rewards + self.discount_factor * maxQ
        element_wise_loss = self.loss(expected_value.detach(), Q_sa.squeeze(), reduction='none')
        weights = torch.FloatTensor(weights).to(self.device)
        loss = torch.mean(element_wise_loss * weights)
        logger.info(f'Train loss : {loss}')

        logger.debug('Backward pass')
        loss.backward()

        logger.debug('Clipping gradient norm')
        # Takes time to clip
        clip_grad_norm_(self.policy_network.parameters(), 1)

        logger.debug('Optimising step')
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step(loss)

        if self.per:
            # Update PER priorities
            element_wise_loss = element_wise_loss.detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices=indices, priorities=element_wise_loss)
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

    def update(self, episode, max_episodes):
        """

        :param episode: episode number
        :return:
        """
        logger.debug('Update of policy net')
        self.update_policy_net()
        logger.debug('done')
        self.goal_sampler.update_embedding()
        self.update_exploration_function(n=episode + 1)
        if self.per:
            self.replay_buffer.update_beta(episode=episode, max_episodes=max_episodes)
        if episode % self.target_update_freq == 0:
            logger.debug('Update of target net')
            self.update_target_net()
            logger.debug('done')
