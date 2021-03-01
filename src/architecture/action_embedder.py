import torch
from torch import nn as nn
from torch.nn.utils import rnn as rnn_utils


def get_action_model(use_attention, **kwargs):
    if use_attention:
        return ActionModelWithAttention(**kwargs)
    else:
        return ActionModel(**kwargs)


class ActionModel(nn.Module):
    def __init__(self, raw_action_size, out_features, merge_thing_action_embedding):
        super().__init__()
        self.in_features = raw_action_size
        self.action_embedding_size = out_features

        # Dictionnary of projection layer for normalizing the action embedding. The projector should be defined as
        # attributes of the current module to ensure that they are passed to cuda
        self.action_embedding_layers = nn.ModuleDict({
            k: nn.Linear(in_features=v, out_features=self.action_embedding_size) for k, v in self.in_features.items()
        })
        self.action_embedding_layers.update(nn.ModuleDict({'root': nn.Identity()}))

        if merge_thing_action_embedding:
            self.action_embedding_layers['channel'] = self.action_embedding_layers['thing']

    def forward(self, actions, action_type, **kwargs):
        """

        :param actions: list of size BATCH_SIZE [
                            list of size N_ACTION [
                                torch.tensor(1, ACTION_EMBEDDING)
                            ]
                        ]
        :return: torch tensor of size (BATCH_SIZE, longest_sequence, PROJECTED_ACTION_EMBEDDING)
        """
        batch = []
        for action_batch, action_type_batch in zip(actions, action_type):
            embedded_batch = [self.action_embedding_layers[t](a) for a, t in zip(action_batch, action_type_batch)]
            batch.append(torch.cat(embedded_batch))
        return rnn_utils.pad_sequence(batch, batch_first=True)


class ActionModelWithAttention(ActionModel):
    def __init__(self, lm_embedding_size, raw_action_size, out_features, merge_thing_action_embedding):
        super().__init__(raw_action_size, out_features, merge_thing_action_embedding)
        self.attention_layers = nn.ModuleDict({
            k: nn.Linear(in_features=lm_embedding_size, out_features=v) for k, v in self.in_features.items()
        })
        self.attention_layers.update(nn.ModuleDict(dict(root=nn.Linear(in_features=lm_embedding_size,
                                                                       out_features=self.action_embedding_size))))

        if merge_thing_action_embedding:
            self.attention_layers['channel'] = self.attention_layers['thing']

    def forward(self, actions, action_type, instruction):
        batch = []
        for action_batch, action_type_batch, ins in zip(actions, action_type, instruction):
            embedded_batch = [self.action_embedding_layers[t](a * self.attention_layers[t](ins))
                              for a, t in zip(action_batch, action_type_batch)]
            batch.append(torch.cat(embedded_batch))
        return rnn_utils.pad_sequence(batch, batch_first=True)
