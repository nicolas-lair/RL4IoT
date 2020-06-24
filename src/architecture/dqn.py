import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from architecture.instruction_pointer_network import InstructedPointerNetwork
from architecture.utils import flatten


class BasicQnet(nn.Module):
    def __init__(self, in_features, qnet_params):
        super().__init__()
        self.q_network = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=qnet_params['hidden1_out']),
            nn.ReLU(),
            nn.Linear(in_features=qnet_params['hidden1_out'], out_features=qnet_params['hidden2_out']),
            nn.ReLU(),
            nn.Linear(in_features=qnet_params['hidden2_out'], out_features=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.q_network(x)


class ActionProjector(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.raw_action_size = in_features
        self.action_embedding_size = out_features
        # Projection layer for action with description embedding
        self.description_embedding_projector = nn.Linear(in_features=self.raw_action_size,
                                                         out_features=self.action_embedding_size)

        # Projection layer for action with their own embedding
        self.openhab_action_projector = nn.Linear(in_features=self.raw_action_size,
                                                  out_features=self.action_embedding_size)

        self.color_projector = nn.Linear(in_features=self.raw_action_size,
                                         out_features=self.action_embedding_size)

        self.level_projector = nn.Linear(in_features=self.raw_action_size,
                                         out_features=self.action_embedding_size)

        # Dictionnary of projection layer for normalizing the action embedding. The projector should be defined as
        # attributes of the current module to ensure that they are passed to cuda
        self.action_embedding_layers = {
            'root': None,
            'description_node': self.description_embedding_projector,
            'openHAB_action': self.openhab_action_projector,
            'color_params': self.color_projector,
            'level_params': self.level_projector
        }

    def forward(self, actions, action_type):
        """

        :param actions: list of size BATCH_SIZE [
                            list of size N_ACTION [
                                torch.tensor(1, ACTION_EMBEDDING)
                            ]
                        ]
        :return: torch tensor of size (BATCH_SIZE, longest_sequence, PROJECTED_ACTION_EMBEDDING)
        """
        with torch.autograd.set_detect_anomaly(False):
            with torch.no_grad():
                action_type_list = list(self.action_embedding_layers.keys())
                action_type_indices = [torch.tensor([action_type_list.index(t) for t in seq]) for seq in action_type]
                action_type_indices = rnn_utils.pad_sequence(action_type_indices, batch_first=True)
                action_type_mask = torch.zeros(*action_type_indices.size(), len(action_type_list))
                action_type_mask = action_type_mask.scatter_(-1, action_type_indices.unsqueeze(dim=-1), 1.).unsqueeze(
                    -1).detach()

                actions = [torch.cat([F.pad(a, pad=(0, self.raw_action_size - a.size(1), 0, 0)) for a in seq], dim=0)
                           for seq in actions]
                actions = rnn_utils.pad_sequence(actions, batch_first=True).detach()

            projection_des = self.description_embedding_projector(actions)
            projection_openhab = self.openhab_action_projector(actions)
            projection_color = self.color_projector(actions)
            projection_level = self.level_projector(actions)
            projection = [torch.zeros(actions.size(0), actions.size(1), self.action_embedding_size).to(actions.device),
                          projection_des, projection_openhab, projection_color, projection_level]

            p = torch.stack(projection, dim=-2)
            embedded_actions = (action_type_mask.to(p.device) * p).sum(dim=-2, keepdim=False)

            # Project each action to a common embedding space and create a tensor for each sublist of action
            # (corresponding to the possible action in one user_state)

            # Method 3
            # embedded_actions = []
            # for seq in actions:
            #     embedded_actions.append(self._project_one_action(torch.cat(seq)))
            # Method 2
            # embedded_actions = []
            # for seq in actions:
            #     sequence = []
            #     for a in seq:
            #         sequence.append(self._project_one_action(a))
            #     embedded_actions.append(torch.cat(sequence))

            # Method 1
            # embedded_actions = [torch.cat([self.action_embedding_layers[a_type](a) for a, a_type in zip(seq, seq_type)])
            #                     for seq, seq_type in zip(actions, action_type)]
            #
            # # Create a unique tensor that contains all the possible actions for the different element in the batch
            # # Need to pad in case some element in the batch have different number of possible actions
            # embedded_actions = rnn_utils.pad_sequence(embedded_actions, batch_first=True)

            return embedded_actions


def flatten_state(state):
    if isinstance(state, dict):
        state = [state]
    flatten_states = [flatten(s) for s in state]
    state = torch.stack([torch.stack(list(s.values())) for s in flatten_states])
    return state


class NoAttentionFlatQnet(nn.Module):
    def __init__(self, instruction_embedding, state_embedding, action_embedding_size, net_params, raw_action_size):
        super().__init__()
        self.in_features = instruction_embedding + state_embedding + 2 * action_embedding_size
        self.q_network = BasicQnet(self.in_features, qnet_params=net_params['q_network'])
        self.action_projector = ActionProjector(in_features=max(raw_action_size.values()),
                                                out_features=action_embedding_size)

    def preprocess_state(self, state, **kwargs):
        state = flatten_state(state)
        state = state.float().mean(1)
        # state = state.view(len(state), -1)
        return state

    def forward(self, instruction, state, actions, hidden_state):
        """

        :param instruction: torch tensor (BATCH_SIZE, INSTRUCTION_EMBEDDING)
        :param state: torch tensor (BATCH_SIZE, STATE_EMBEDDING)
        :param hidden_state: torch.tensor(BATCH_SIZE, ACTION_EMBEDDING)
                    string can be "action_with_description" or "action_without_description"
        :param actions: tuple containing actions pre-embedding and action type
        :return: torch tensor of size (BATCH_SIZE, longest_sequence, 1), hidden_state (actions here)
        """
        with torch.autograd.set_detect_anomaly(False):
            embedded_actions = self.action_projector(*actions)
            state = self.preprocess_state(state, instruction=instruction)

            context = torch.cat([instruction, state, hidden_state], dim=1)
            context = context.unsqueeze(1).repeat_interleave(repeats=embedded_actions.size(1), dim=1)
            x = torch.cat([context, embedded_actions], dim=2)
            x = self.q_network(x)
            return x, embedded_actions


class AttentionFlatQnet(NoAttentionFlatQnet):
    def __init__(self, instruction_embedding, state_embedding, action_embedding_size, net_params, raw_action_size):
        super().__init__(instruction_embedding, state_embedding, action_embedding_size, net_params, raw_action_size)
        self.state_attention_layer = nn.Sequential(nn.Linear(instruction_embedding, state_embedding),
                                                   nn.Sigmoid())

    def preprocess_state(self, state, **kwargs):
        attention_vector = self.state_attention_layer(kwargs['instruction'])
        state = flatten_state(state)
        state = (attention_vector.unsqueeze(1) * state).float().mean(1)
        return state


class DeepSetQnet(nn.Module):
    def __init__(self, instruction_embedding, state_embedding, action_embedding_size, net_params, raw_action_size):
        super().__init__()
        self.in_features = instruction_embedding + net_params['scaler_layer']['latent_out'] + action_embedding_size

        self.action_projector = ActionProjector(in_features=max(raw_action_size.values()),
                                                out_features=action_embedding_size)
        self.state_attention_layer = nn.Sequential(
            nn.Linear(instruction_embedding, state_embedding + action_embedding_size),
            nn.Sigmoid()
        )
        self.scaler_layer = nn.Sequential(
            nn.Linear(state_embedding, net_params['scaler_layer']['hidden1_out']),
            nn.ReLU(),
            nn.Linear(net_params['scaler_layer']['hidden1_out'], net_params['scaler_layer']['latent_out']),
            nn.ReLU()
        )

        self.q_network = BasicQnet(self.in_features, qnet_params=net_params['q_network'])

        self.in_features = 512 + instruction_embedding + 2 * action_embedding_size
        self.state_attention_layer = nn.Sequential(nn.Linear(instruction_embedding, state_embedding),
                                                   nn.Sigmoid())
        super().__init__(instruction_embedding, 512, action_embedding_size, net_params, raw_action_size)
        self.type = 'DeepSetQnet'

    def forward(self, instruction, state, actions, hidden_state):
        """

        :param instruction: torch tensor (BATCH_SIZE, INSTRUCTION_EMBEDDING)
        :param state: torch tensor (BATCH_SIZE, STATE_EMBEDDING)
        :param hidden_state: torch.tensor(BATCH_SIZE, ACTION_EMBEDDING)
                    string can be "action_with_description" or "action_without_description"
        :param actions: tuple containing actions pre-embedding and action type
        :return: torch tensor of size (BATCH_SIZE, longest_sequence, 1), hidden_state (actions here)
        """
        with torch.autograd.set_detect_anomaly(False):
            embedded_actions = self.project_action_embedding(*actions)

            flatten_state = flatten_state(state).float()
            hidden_state = hidden_state.unsqueeze(1).repeat_interleave(repeats=flatten_state.size(1), dim=1)
            full_state = torch.cat([flatten_state, hidden_state])
            attention_vector = self.state_attention_layer(instruction)
            full_state = attention_vector.unsqueeze(1) * full_state
            full_state = self.scaler_layer(full_state).mean(1, keepdim=True)

            full_state = full_state.repeat_interleave(repeats=flatten_state.size(1), dim=1)
            x = torch.cat([full_state, embedded_actions], dim=2)
            x = self.q_network(x)
            return x, embedded_actions


class DoubleDeepSetQnet(DeepSetQnet):
    def __init__(self, instruction_embedding, state_embedding, action_embedding_size, net_params, raw_action_size):
        super().__init__(instruction_embedding, state_embedding, action_embedding_size, net_params, raw_action_size)
        self.type = 'DeepSetQnet'


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
