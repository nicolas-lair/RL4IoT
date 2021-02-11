import torch
from torch import nn as nn

from architecture.utils import flatten_state, differentiable_or


class Net(nn.Module):
    def __init__(self, n_inputs):
        super(Net, self).__init__()
        self.shared_encoding = nn.Sequential(nn.Linear(1, 100),
                                             nn.ReLU(),
                                             nn.Linear(100, n_inputs))
        self.out_layer = nn.Sequential(nn.Linear(n_inputs, 100),
                                       nn.ReLU(),
                                       nn.Linear(100, 1),
                                       nn.Sigmoid())

    def forward(self, x):
        latent = self.shared_encoding(x.unsqueeze(dim=2))
        latent = latent.sum(dim=1)
        out = self.out_layer(latent)
        return out


def build_scaler_layer(input_size, hidden_size, output_size, last_activation):
    if last_activation == 'relu':
        last_activation_layer = nn.ReLU()
    elif last_activation == 'sigmoid':
        last_activation_layer = nn.Sigmoid()
    else:
        raise NotImplementedError(
            f'last_scaler_activation should be one of relu or sigmoid, not {last_activation}')

    scaler_layer = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
        last_activation_layer
    )

    return scaler_layer


class DeepSetStateNet(nn.Module):
    def __init__(self, instruction_embedding, state_embedding, scaler_layer_params, hidden_state_size=0,
                 aggregate='mean'):
        super().__init__()
        self.goal_attention_layer = nn.Sequential(
            nn.Linear(instruction_embedding, state_embedding + hidden_state_size),
            nn.Sigmoid()
        )

        self.scaler_layer = build_scaler_layer(state_embedding + hidden_state_size, **scaler_layer_params)

        self.out_features = scaler_layer_params['output_size']

        assert aggregate in ['mean', 'sum', 'diff_or',
                             None], f'aggregate should be one of mean, sum or None, not {aggregate}'
        self.aggregation = aggregate
        if self.aggregation == 'diff_or':
            self.diff_or = torch.load('/home/nicolas/PycharmProjects/imagineIoT/model/params_or.pk')
            # self.diff_or = Net(n_inputs=3)
            # self.diff_or.load_state_dict(torch.load('/home/nicolas/PycharmProjects/imagineIoT/model/params_or.pk'))
            self.diff_or.eval()
            for param in self.diff_or.parameters():
                param.requires_grad = False

    def _aggregate(self, full_state):
        if self.aggregate == 'mean':
            full_state = full_state.mean(1)
        elif self.aggregate == 'sum':
            full_state = full_state.sum(1)
        elif self.aggregate == 'diff_or':
            full_state = self.diff_or(full_state)
        elif self.aggregate is None:
            pass
        else:
            raise NotImplementedError
        return full_state

    @staticmethod
    def format_state(state):
        # full_state = flatten_state(state).float()
        if isinstance(state, dict):
            state = [state]
        full_state = [torch.cat(list(s.values())) for s in state]  # cat all channels from state
        full_state = torch.stack(full_state)  # stack batch state
        return full_state

    def forward(self, state, instruction, hidden_state=None):
        full_state = self.format_state(state)
        if hidden_state is not None:
            hidden_state = hidden_state.unsqueeze(1).repeat_interleave(repeats=full_state.size(1), dim=1)
            full_state = torch.cat([full_state, hidden_state], dim=2)
        attention_vector = self.goal_attention_layer(instruction)
        full_state = attention_vector.unsqueeze(1) * full_state
        full_state = self.scaler_layer(full_state)
        full_state = self._aggregate(full_state)
        # print(full_state.size())
        return full_state


class DoubleAttDeepSet(DeepSetStateNet):
    def __init__(self, instruction_embedding, state_embedding, scaler_layer_params, hidden_state_size=0,
                 aggregate='mean'):
        super(DoubleAttDeepSet, self).__init__(instruction_embedding, state_embedding, scaler_layer_params,
                                               hidden_state_size=0, aggregate=aggregate)
        self.hidden_state_attention_layer = nn.Sequential(
            nn.Linear(hidden_state_size, state_embedding),
            nn.Sigmoid()
        )

    def forward(self, state, instruction, hidden_state):
        # full_state = flatten_state(state).float()
        full_state = self.format_state(state)
        goal_attention_vector = self.goal_attention_layer(instruction).unsqueeze(1)
        hidden_state_attention_vector = self.hidden_state_attention_layer(hidden_state).unsqueeze(1)
        full_state = hidden_state_attention_vector * goal_attention_vector * full_state
        full_state = self.scaler_layer(full_state)
        full_state = self._aggregate(full_state)
        # print(full_state.size())
        return full_state


class HierarchicalDeepSet(DeepSetStateNet):
    def __init__(self, instruction_embedding, state_embedding, scaler_layer_params, hidden_state_size=0,
                 aggregate='mean'):
        super().__init__(instruction_embedding, state_embedding, scaler_layer_params, hidden_state_size, aggregate)
        first_scaler_params = scaler_layer_params.copy()
        first_scaler_params.update(dict(output_size=scaler_layer_params['hidden_size']))
        self.scaler_layer = build_scaler_layer(state_embedding + hidden_state_size, **first_scaler_params)

        self.second_scaler = build_scaler_layer(scaler_layer_params['hidden_size'], **scaler_layer_params)
        self.second_attention_layer = nn.Sequential(
            nn.Linear(instruction_embedding, scaler_layer_params['hidden_size']),
            nn.Sigmoid()
        )

    def aggregate_by_thing(self, batch_vector, n_channels):
        begin = 0
        aggregated_vectors = []
        for i in n_channels:
            aggregated_vectors.append(self._aggregate(batch_vector[:, begin:(begin+i), :]))
            begin = i
        return torch.stack(aggregated_vectors, dim=1)

    @staticmethod
    def get_number_of_channels(state):
        if isinstance(state, list):
            state = state[0]
        return [thing.size(0) for thing in state.values()]

    def forward(self, state, instruction, hidden_state=None):
        full_state = self.format_state(state)
        n_channels = self.get_number_of_channels(state)
        assert sum(n_channels, 0) == full_state.size(1)

        if hidden_state is not None:
            hidden_state = hidden_state.unsqueeze(1).repeat_interleave(repeats=full_state.size(1), dim=1)
            full_state = torch.cat([full_state, hidden_state], dim=2)
        attention_vector = self.goal_attention_layer(instruction)
        full_state = attention_vector.unsqueeze(1) * full_state
        full_state = self.scaler_layer(full_state)
        full_state = self.aggregate_by_thing(full_state, n_channels)

        second_attention_vector = self.second_attention_layer(instruction)
        full_state = second_attention_vector.unsqueeze(1) * full_state
        full_state = self.second_scaler(full_state)
        full_state = self._aggregate(full_state)
        return full_state


class FlatStateNet(nn.Module):
    def __init__(self, instruction_embedding, state_embedding, hidden_state_size=0):
        super().__init__()
        self.out_features = instruction_embedding + state_embedding + hidden_state_size

    def forward(self, state, instruction, hidden_state=None):
        """

        :param state: dict
        :param instruction: torch tensor
        :param hidden_state: torch tensor
        :return: torch tensor of size (batch_size, 1, self.out_features)
        """
        state = flatten_state(state)
        state = state.float().mean(1)
        context = [instruction, state]
        if hidden_state is not None:
            context.append(hidden_state)
        context_tensor = torch.cat(context, dim=1)
        return context_tensor


class AttentionFlatState(nn.Module):
    def __init__(self, instruction_embedding, state_embedding, hidden_state_size=0):
        super().__init__()
        self.state_attention_layer = nn.Sequential(nn.Linear(instruction_embedding, state_embedding),
                                                   nn.Sigmoid())
        self.out_features = instruction_embedding + state_embedding + hidden_state_size

    def forward(self, state, instruction, hidden_state=None):
        attention_vector = self.state_attention_layer(instruction)
        state = flatten_state(state)
        state = (attention_vector.unsqueeze(1) * state).float().mean(1)
        context = [instruction, state]
        if hidden_state is not None:
            context.append(hidden_state)
        context_tensor = torch.cat(context, dim=1)
        return context_tensor
