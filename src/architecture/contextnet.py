import torch
from torch import nn as nn

from utils import flatten_state


class DeepSetStateNet(nn.Module):
    def __init__(self, instruction_embedding, state_embedding, scaler_layer_params, hidden_state_size=0):
        super().__init__()
        self.state_attention_layer = nn.Sequential(
            nn.Linear(instruction_embedding, state_embedding + hidden_state_size),
            nn.Sigmoid()
        )
        self.scaler_layer = nn.Sequential(
            nn.Linear(state_embedding + hidden_state_size, scaler_layer_params['hidden1_out']),
            nn.ReLU(),
            nn.Linear(scaler_layer_params['hidden1_out'], scaler_layer_params['latent_out']),
            nn.ReLU()
        )
        self.out_features = scaler_layer_params['latent_out']

    def forward(self, state, instruction, hidden_state=None):
        full_state = flatten_state(state).float()
        if hidden_state is not None:
            hidden_state = hidden_state.unsqueeze(1).repeat_interleave(repeats=full_state.size(1), dim=1)
            full_state = torch.cat([full_state, hidden_state], dim=2)
        attention_vector = self.state_attention_layer(instruction)
        full_state = attention_vector.unsqueeze(1) * full_state
        full_state = self.scaler_layer(full_state).mean(1)
        print(full_state.size())
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
        # state = state.view(len(state), -1)
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