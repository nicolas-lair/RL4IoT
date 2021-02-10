import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import rnn as rnn_utils


class ActionModel(nn.Module):
    def __init__(self, raw_action_size, out_features, env_discrete_params):
        super().__init__()
        self.in_features = max(raw_action_size.values())
        self.action_embedding_size = out_features

        # Dictionnary of projection layer for normalizing the action embedding. The projector should be defined as
        # attributes of the current module to ensure that they are passed to cuda
        self.action_embedding_layers = nn.ModuleDict({
            'root': nn.Identity(),
            'description_node': nn.Linear(in_features=self.in_features, out_features=self.action_embedding_size),
            'openHAB_action': nn.Linear(in_features=self.in_features, out_features=self.action_embedding_size),
        })
        self.action_embedding_layers.update(
            nn.ModuleDict(
                {k + '_params': nn.Linear(in_features=self.in_features, out_features=self.action_embedding_size)
                 for k in env_discrete_params})
        )

    def forward(self, actions, action_type):
        """

        :param actions: list of size BATCH_SIZE [
                            list of size N_ACTION [
                                torch.tensor(1, ACTION_EMBEDDING)
                            ]
                        ]
        :return: torch tensor of size (BATCH_SIZE, longest_sequence, PROJECTED_ACTION_EMBEDDING)
        """
        with torch.no_grad():
            action_type_list = list(self.action_embedding_layers.keys())
            action_type_indices = [torch.tensor([action_type_list.index(t) for t in seq]) for seq in action_type]
            action_type_indices = rnn_utils.pad_sequence(action_type_indices, batch_first=True)
            action_type_mask = torch.zeros(*action_type_indices.size(), len(action_type_list))
            action_type_mask = action_type_mask.scatter_(-1, action_type_indices.unsqueeze(dim=-1), 1.).unsqueeze(
                -1).detach()

            actions = [torch.cat([F.pad(a, pad=(0, self.in_features - a.size(1), 0, 0)) for a in seq], dim=0)
                       for seq in actions]
            actions = rnn_utils.pad_sequence(actions, batch_first=True).detach()

        projection = [v(actions) for v in self.action_embedding_layers.values()]
        p = torch.stack(projection, dim=-2)
        embedded_actions = (action_type_mask.to(p.device) * p).sum(dim=-2, keepdim=False)
        return embedded_actions