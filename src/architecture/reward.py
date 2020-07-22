import numpy as np
import joblib
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from architecture.utils import differentiable_or


class LearnedReward(nn.Module):
    def __init__(self, context_model, language_model, reward_params):
        super().__init__()
        self.context_net = context_model(**reward_params)

        self.language_model = language_model

        # self.linear_attn = nn.Linear(in_features=self.language_model.out_features, out_features=observation_size)
        # self.hidden_layer = nn.Linear(in_features=observation_size, out_features=hidden_layer_size)

        self.reward_layer = nn.Linear(in_features=self.context_net.out_features, out_features=1)

    def forward(self, state, instructions):
        """

        :param state:
        :param instructions: list of string instructions
        :return:
        """
        instruction_embedding = self.language_model(instructions).view(len(instructions), -1)
        context = self.context_net(state=state, instruction=instruction_embedding)
        x = self.reward_layer(context)
        reward = torch.sigmoid(x)
        # reward = differentiable_or(x.view(-1))
        return reward


class StateDataset(Dataset):
    def __init__(self, data, data_type, raw_state_transformer=None, max_size=None, keep_ind=None):
        if data_type == 'file':
            if isinstance(data, str):
                data = [data]
            if isinstance(data, list):
                data = data
                self.state_record = []
                for f in data:
                    self.state_record.extend(joblib.load(f))
                    if max_size and len(self) > max_size:
                        break
        elif data_type == 'list':
            self.state_record = data
        else:
            raise NotImplementedError('data_type should be one of "file" or "list". '
                                      'Prefer class method from_files and from_list')

        if max_size:
            self.state_record = self.state_record[:max_size]

        if keep_ind:
            self.state_record = [self.state_record[i] for i in keep_ind]

        self.transform = raw_state_transformer

    def __len__(self):
        return len(self.state_record)

    def __getitem__(self, item):
        state = self.state_record[item]
        state = state._asdict()

        if self.transform:
            state['state'] = self.transform(state['state'])
        return state

    def split(self, train_test_ratio=None, train_size=None, test_size=None, max_train=None, max_test=None):
        n = len(self.state_record)
        if train_test_ratio:
            train_size = int(train_test_ratio * n)
        elif test_size:
            train_size = n - test_size
        elif train_size is None:
            raise NotImplementedError

        train_idx = list(np.random.choice(range(n), size=train_size))
        test_idx = list(set(range(n)).difference(set(train_idx)))

        train_dts = StateDataset.from_list(data=self.state_record, raw_state_transformer=self.transform,
                                           keep_ind=train_idx,
                                           max_size=max_train)
        test_dts = StateDataset.from_list(data=self.state_record, raw_state_transformer=self.transform,
                                          keep_ind=test_idx,
                                          max_size=max_test)
        return train_dts, test_dts

    @classmethod
    def from_files(cls, data, raw_state_transformer=None, max_size=None, keep_ind=None):
        return cls(data=data, data_type='file', raw_state_transformer=raw_state_transformer, max_size=max_size,
                   keep_ind=keep_ind)

    @classmethod
    def from_list(cls, data, raw_state_transformer=None, max_size=None, keep_ind=None):
        return cls(data=data, data_type='list', raw_state_transformer=raw_state_transformer, max_size=max_size,
                   keep_ind=keep_ind)


if __name__ == "__main__":
    pass
