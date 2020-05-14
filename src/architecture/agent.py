import random
import numpy as np
import torch


class Agent:
    def __init__(self, model, exploration_params, goal_sampler, language_model):
        self.model = model
        self.steps = 0
        self.hidden_state = None
        self.exploration_params = exploration_params
        self.goal_sampler = goal_sampler
        self.language_model =language_model

    def sample_goal(self, strategy=None):
        return self.goal_sampler.sample_goal(strategy=strategy)

    def embed_actions(self, actions):
        """

        :param actions: list of torch tensors (possible actions)
        :return:
        """
        action_embeddings = []
        for a in actions:
            if a.description_embedding is not None:
                # NL embedding already computed or other types of embedding
                action_embeddings.append(a.description_embedding)
            elif a.has_description:
                # if action has no embedding already, it should have a description
                # We then use the language model to compute an embedding
                a.description_embedding = self.language_model(a.description)
                action_embeddings.append(a.description_embedding)
            else:
               raise NotImplementedError
        return torch.cat(action_embeddings, dim=0)


    def select_action(self, state, instruction, actions):
        """
        select an action among a list of actions and update
        :param state:
        :param instruction:
        :param actions:
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
                Q, hidden_state = self.model(state, instruction, actions, self.hidden_state).argmax(
                    1)  # TODO check
                hidden_state = hidden_state.view(*self.hidden_state.size())
        else:
            action_idx = torch.tensor([random.randint(0, actions.size(0))])  # TODO check
            hidden_state = actions[action_idx].view(*self.hidden_state.size())  # TODO check

        self.hidden_state += hidden_state
        return action_idx
