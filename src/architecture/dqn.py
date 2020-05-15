import itertools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from instruction_pointer_network import InstructedPointerNetwork


class FlatCritic(nn.Module):
    def __init__(self, instruction_embedding, state_embedding, action_embedding, n_step, net_params,
                 action_description_embedding, action_standard_embedding):
        super().__init__()
        self.in_features = instruction_embedding + state_embedding + 2 * action_embedding + n_step
        self.action_embedding_layers = {
            'action_with_description': nn.Linear(in_features=action_description_embedding,
                                                 out_features=action_embedding),
            'action_without_description': nn.Linear(in_features=action_standard_embedding,
                                                    out_features=action_embedding)
        }
        self.q_network = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=net_params['hidden1_out']),
            nn.ReLU(),
            nn.Linear(in_features=net_params['hidden2_out'], out_features=net_params['hidden2_out']),
            nn.ReLU(),
            nn.Linear(in_features=net_params['hidden2_out'], out_features=1),
            nn.ReLU()
        )

    def project_action_embedding(self, actions):
        """

        :param actions: list of size BATCH_SIZE [
                            list of size N_ACTION [
                                tuple(string, torch.tensor(1, ACTION_EMBEDDING))
                            ]
                        ]
        :return: torch tensor of size (BATCH_SIZE, longest_sequence, PROJECTED_ACTION_EMBEDDING)
        """
        # Project each action to a common embedding space and create a tensor for each sublist of action
        # (corresponding to the possible action in one state)
        embedded_actions = [torch.cat([self.action_embedding_layers[a[0]](a[1]) for a in seq]) for seq in actions]
        # Create a unique tensor that contains all the possible actions for the different element in the batch
        # Need to pad in case some element in the batch have different number of possible actions
        embedded_actions = rnn_utils.pad_sequence(embedded_actions, batch_first=True)

        return embedded_actions

    def forward(self, instruction, state, hidden_state, actions):
        """

        :param instruction: torch tensor (BATCH_SIZE, INSTRUCTION_EMBEDDING)
        :param state: torch tensor (BATCH_SIZE, STATE_EMBEDDING)
        :param hidden_state: torch.tensor(BATCH_SIZE, ACTION_EMBEDDING)
                    string can be "action_with_description" or "action_without_description"
        :param actions: torch tensor of size (BATCH_SIZE, longest_sequence, PROJECTED_ACTION_EMBEDDING)

        :return: torch tensor of size (BATCH_SIZE, longest_sequence, 1), hidden_state (actions here)
        """
        embedded_actions = self.project_action_embedding(actions)

        context = torch.cat([instruction, state, hidden_state], dim=1)
        context = context.unsqueeze(1).repeat_interleave(repeats=embedded_actions.size(1), dim=1)
        x = torch.cat([context, embedded_actions], dim=2)
        x = self.q_network(x)
        return x, embedded_actions


class DeepSetActionCritic(nn.Module):
    def __init__(self, instruction_embedding, state_embedding, action_embedding, n_step, net_params):
        super().__init__()
        self.action_projector = nn.Linear(in_features=action_embedding, out_features=net_params['action_projection'],
                                          bias=False)
        self.in_features = instruction_embedding + state_embedding + action_embedding + n_step + net_params[
            'action_projection']
        self.hidden1 = nn.Linear(in_features=self.in_features, out_features=net_params['hidden1_out'])
        self.hidden2 = nn.Linear(in_features=net_params['hidden2_out'], out_features=net_params['hidden2_out'])
        self.final_layer = nn.Linear(in_features=net_params['hidden2_out'], out_features=1)

    # TODO check with gated attention on instruction
    def forward(self, instruction, state, previous_action, actions, step):
        projected_action = self.action_projector(actions)
        action_context = projected_action.sum(dim=0)

        context = torch.cat([instruction, state, previous_action, step, action_context])
        c = context.repeat_interleave(actions.size(0), dim=0)

        x = torch.cat([c, actions], dim=1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.final_layer(x)
        x = x.view(-1)
        action_distribution = F.softmax(x)
        return action_distribution


class AttentionCritic(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, instruction, state, previous_action, actions, step):
        pass


class SelfAttentionCritic(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, instruction, state, previous_action, actions, step):
        pass


class PointerCritic(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, instruction, state, previous_action, actions, step):
        pass


class DifferentPtrNetCritic(nn.Module):
    def __init__(self, language_model, thing_selector_params, channel_selector_params, action_selector_params):
        super().__init__()
        self.language_model = language_model
        self.thing_selector = InstructedPointerNetwork(**thing_selector_params)
        self.channel_selector = InstructedPointerNetwork(**channel_selector_params)
        self.action_selector = InstructedPointerNetwork(**action_selector_params)

    def forward(self, instruction, state, previous_action, actions, step):
        embedded_instruction = self.language_model(instruction)
        condition = torch.cat([embedded_instruction, state, previous_action], dim=1)

        # TODO use action sampled from action_proba rather than argmax for better exploration during learning phase
        if step == 'thing':
            action_proba, best_action = self.thing_selector(sequence=actions, condition_embedding=condition)
        elif step == "channel":
            action_proba, best_action = self.channel_selector(sequence=actions, condition_embedding=condition)
        elif step == "action":
            action_proba, best_action = self.action_selector(sequence=actions, condition_embedding=condition)
        elif step == "params":
            selector = None
        elif step == "go":
            selector = None
        else:
            raise NotImplementedError
