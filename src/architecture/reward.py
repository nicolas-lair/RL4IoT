import numpy as np
import pandas as pd
import joblib
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import f1_score, recall_score, precision_score


class Reward:
    def __init__(self, context_model, language_model, net_params, fit_params, device):
        self.reward_model = RewardModel(context_model=context_model, language_model=language_model,
                                        reward_params=net_params)
        self.device = device
        self.reward_model.to(self.device)

        self.optimizer = fit_params['optimizer'](self.reward_model.parameters())
        self.loss_function = fit_params['loss']()
        self.batch_size = fit_params['batch_size']
        self.n_epoch = fit_params['n_epoch']
        self.sampler_params = fit_params['sampler_params']

        self.metrics = None

    def fit(self, train_dts, eval=False):
        if self.reward_model.language_model.frozen:
            self.reward_model.language_model.unfreeze()

        sampler = ImbalancedDatasetSampler(train_dts, **self.sampler_params)
        train_loader = DataLoader(dataset=train_dts, sampler=sampler, batch_size=self.batch_size)

        for epoch in range(self.n_epoch):
            for i, batch in tqdm(enumerate(train_loader)):
                self.optimizer.zero_grad()
                reward = self.reward_model(state=batch['state'], instructions=batch['instruction']).view(-1)
                loss = self.loss_function(reward, batch['reward'].float().to(params['device']))
                loss.backward()
                clip_grad_norm_(reward_function.parameters(), 1)
                self.optimizer.step()

    def eval(self, test_batch, epoch, data_stats=False):
        def compute_stats(g):
            count = len(g)
            pred_1 = g.pred.sum()
            pred_0 = (1 - g.pred).sum()
            true_1 = g.true.sum()
            true_0 = (1 - g.true).sum()
            stats = [count, true_0, pred_0, true_1, pred_1]
            stats_name = ['count', 'true_0', 'pred_0', 'true_1', 'pred_1']
            result = pd.DataFrame(stats).T
            result.columns = stats_name
            return result

        def compute_classification_metrics(g):
            # accuracy = accuracy_score(g.true, g.pred)
            precision = precision_score(g.true, g.pred)
            recall = recall_score(g.true, g.pred)
            f1 = f1_score(g.true, g.pred)
            # scores = [accuracy, precision, recall, f1]
            scores = [precision, recall, f1]
            score_name = ['accuracy', 'precision', 'recall', 'f1_score']
            result = pd.DataFrame(scores).T
            result.columns = score_name
            return result

        def compute_metrics(g, data_stats=False):
            df_score = compute_classification_metrics(g)
            df_stats = compute_stats(g) if data_stats else pd.DataFrame()
            result = pd.concat([df_stats, df_score], axis=1)
            return result

        test_reward = reward_function(state=test_batch['state'], instructions=test_batch['instruction'])#.view(-1)
        torch.cat([test_reward.cpu(), test_batch['reward'].float().view(-1, 1)], dim=1)
        pred_true_tensor = torch.cat([(test_reward >0.5).float().cpu(), test_batch['reward'].float().view(-1, 1)], dim=1)
        df = pd.DataFrame(pred_true_tensor.detach().numpy(), columns=['pred', 'true'])
        df['instruction'] = test_batch['instruction']
        metrics = df.groupby('instruction').apply(compute_metrics, data_stats=data_stats)
        metrics = metrics.reset_index(level=1, drop=True)

        metrics_overall = compute_metrics(df, data_stats=False)
        metrics_overall.index = 'model'
        metrics.append(metrics_overall)
        metrics.reset_index(inplace=True)

        metrics['epoch'] = epoch

        if metrics is None:
            self.metrics = metrics
        else:
            self.metrics = self.metrics.append(metrics, ignore_index=True)

        return metrics



class RewardModel(nn.Module):
    def __init__(self, context_model, language_model, reward_params):
        super().__init__()
        self.context_net = context_model(**reward_params)

        self.language_model = language_model

        # self.linear_attn = nn.Linear(in_features=self.language_model.out_features, out_features=observation_size)
        # self.hidden_layer = nn.Linear(in_features=observation_size, out_features=hidden_layer_size)

        self.prelearned_or = (self.context_net.aggregate == 'diff_or')

        if not self.prelearned_or:
            self.reward_layer = nn.Sequential(
                nn.Linear(in_features=self.context_net.out_features, out_features=128),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=1),
                nn.Sigmoid()
            )

    def forward(self, state, instructions):
        """

        :param state:
        :param instructions: list of string instructions
        :return:
        """
        instruction_embedding = self.language_model(instructions).view(len(instructions), -1)
        r = self.context_net(state=state, instruction=instruction_embedding)
        if not self.prelearned_or:
            r = self.reward_layer(r)
        return r


class StateDataset(Dataset):
    def __init__(self, data, data_type, raw_state_transformer=None, max_size=np.inf):
        if data_type == 'file':
            if isinstance(data, str):
                data = [data]
            if isinstance(data, list):
                pass
            else:
                raise NotImplementedError('data should be a file name or list of file name')

            self.positive_record = []
            self.negative_record = []
            for f in data:
                pos_batch, neg_batch = joblib.load(f)
                self.positive_record = pos_batch
                self.negative_record = neg_batch
                if len(self.positive_record) + len(self.negative_record) > max_size:
                    break

        elif data_type == 'tuple':
            self.positive_record, self.negative_record = data
        else:
            raise NotImplementedError('data_type should be one of "file" or "tuple". '
                                      'Prefer class method from_files and from_tuple')

        self.build_record_index()

        n = len(self)
        if max_size < n:
            # pos_ind, neg_ind = self.build_index_subset(max_size)
            index = np.random.choice(range(len(self)), max_size, replace=False)
            self.positive_record, self.negative_record = self.build_record_subset_from_index(index)
            self.build_record_index()

        self.transform = raw_state_transformer
        try:
            self.device = self.transform.keywords['device']
        except (AttributeError, KeyError) as e:
            print(e)
            self.device = None

    def __len__(self):
        return len(self.record_index)

    def get_n_pos_examples(self):
        return sum([len(v) for v in self.positive_record.values()])

    def build_record_index(self):
        record_index = {}
        current_index = 0
        for k, v in self.positive_record.items():
            record_index.update(
                {index: ('pos', k, j) for index, j in zip(range(current_index, current_index + len(v)), range(len(v)))})
            current_index = current_index + len(v)
        for k, v in self.negative_record.items():
            record_index.update(
                {index: ('neg', k, j) for index, j in zip(range(current_index, current_index + len(v)), range(len(v)))})
            current_index = current_index + len(v)
        self.record_index = record_index

    # def build_index_subset(self, size):
    #     n_pos = self.get_n_pos_examples()
    #     ind = np.random.choice(range(len(self)), size, replace=False)
    #
    #     pos_ind = [i for i in ind if i < n_pos]
    #     neg_ind = [i for i in ind if i >= n_pos]
    #     return pos_ind, neg_ind

    def build_record_subset_from_index(self, ind):
        instruction_set = self.positive_record.keys()
        positive_record = {i: [] for i in instruction_set}
        negative_record = {i: [] for i in instruction_set}
        for i in ind:
            (sign, instruction, _) = self.record_index[i]
            record = positive_record if sign == 'pos' else negative_record
            record[instruction].append(self._get(i, as_dict=False))

        # for ind in pos_ind:
        #     (_, instruction, j) = self.record_index[ind]
        #     positive_record[instruction].append(self.positive_record[instruction][j])
        #
        # for (_, instruction, j) in [self.record_index[ind] for ind in neg_ind]:
        #     negative_record[instruction].append(self.negative_record[instruction][j])

        return positive_record, negative_record

    def _get(self, item, as_dict=True):
        (sign, instruction, ind) = self.record_index[item]
        if sign == 'pos':
            d = self.positive_record
        elif sign == 'neg':
            d = self.negative_record
        else:
            raise ValueError('There is an issue with the dataset when building the record index')

        state = d[instruction][ind]
        if as_dict:
            state = state._asdict()
        return state

    def __getitem__(self, item):
        state = self._get(item)
        if self.transform:
            state['state'] = self.transform(state['state'])
        return state

    def split(self, train_test_ratio=None, train_size=None, test_size=None, max_train=np.inf, max_test=np.inf):
        n = len(self)
        if train_test_ratio:
            train_size = round(train_test_ratio * n)
        elif test_size:
            train_size = n - test_size
        elif train_size is None:
            raise NotImplementedError

        # pos_train_ind, neg_train_ind = self.build_index_subset(train_size)
        train_index = np.random.choice(range(len(self)), train_size, replace=False)
        pos_train_records, neg_train_records = self.build_record_subset_from_index(train_index)

        test_index = list(set(self.record_index).difference(set(train_index)))
        # pos_test_ind = list(set(range(self.get_n_pos_examples())).difference(set(pos_train_ind)))
        # neg_test_ind = list(set(range(self.get_n_pos_examples(), len(self))).difference(set(neg_train_ind)))
        pos_test_records, neg_test_records = self.build_record_subset_from_index(test_index)

        train_dts = self.__class__.from_tuple(data=(pos_train_records, neg_train_records),
                                              raw_state_transformer=self.transform,
                                              max_size=max_train)
        test_dts = self.__class__.from_tuple(data=(pos_test_records, neg_test_records),
                                             raw_state_transformer=self.transform,
                                             max_size=max_test)
        return train_dts, test_dts

    @classmethod
    def from_files(cls, data, raw_state_transformer=None, max_size=np.inf):
        return cls(data=data, data_type='file', raw_state_transformer=raw_state_transformer, max_size=max_size)

    @classmethod
    def from_tuple(cls, data, raw_state_transformer=None, max_size=np.inf):
        return cls(data=data, data_type='tuple', raw_state_transformer=raw_state_transformer, max_size=max_size)

    def get_stats(self):
        import pandas as pd
        positive_count = [len(v) for v in self.positive_record.values()]
        negative_count = [len(v) for v in self.negative_record.values()]
        df = pd.DataFrame([positive_count, negative_count], columns=['pos', 'neg']).T
        df.index = list(self.positive_record)
        df['%pos'] = df.pos / (df.pos + df.neg)
        return df


class EpisodeDataset(StateDataset):
    def __getitem__(self, item):
        state = self._get(item)

        if self.transform:
            # TODO fix hack
            raw_state_size = self.transform.keywords['raw_state_size']
            state['state'] = self.transform(state['final_state'])
            for thing_name, thing in state['initial_state'].items():
                for channel_name, channel in thing.items():
                    if channel['item_type'] == 'goal_string':
                        raise NotImplementedError
                    state_embedding = np.zeros(raw_state_size)
                    state_embedding[:len(channel['state'])] = channel['state']
                    state_embedding = torch.tensor(state_embedding, device=self.device)

                    full_state = state['state'][thing_name][channel_name]
                    diff_state_embedding = full_state[-raw_state_size:] - state_embedding
                    state['state'][thing_name][channel_name] = torch.cat([full_state, diff_state_embedding])
                    assert full_state.size(0) + 3 == state['state'][thing_name][channel_name].size(0)

        del state['initial_state']
        del state['final_state']
        return state


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, pos_weight=0.2):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # weight for each sample
        pos_samples = [len(v) for v in dataset.positive_record.values()]
        neg_samples = [len(v) for v in dataset.negative_record.values()]
        weights = sum([[pos_weight / (n_samples * len(pos_samples))] * n_samples for n_samples in pos_samples], [])
        weights += sum([[(1 - pos_weight) / (n_samples * len(neg_samples))] * n_samples for n_samples in neg_samples],
                       [])
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        b = (self.indices[ind] for ind in torch.multinomial(
            self.weights, self.num_samples, replacement=False))
        return b

    def __len__(self):
        return self.num_samples


if __name__ == "__main__":
    from collections import namedtuple
    from sklearn.preprocessing import OneHotEncoder

    from config import generate_params
    from architecture.language_model import LanguageModel
    from simulator.description_embedder import Description_embedder
    from simulator.Environment import preprocess_raw_observation
    from simulator.Items import ITEM_TYPE

    StateRecord = namedtuple('StateRecord', ('state', 'instruction', 'reward'))
    EpisodeRecord = namedtuple('EpisodeRecord', ('initial_state', 'final_state', 'instruction', 'reward'))

    params = generate_params(save_path=False)
    state_path = '/home/nicolas/PycharmProjects/imagineIoT/results/state_records.jbl'
    episode_path = '/home/nicolas/PycharmProjects/imagineIoT/results/episodes_records3.jbl'

    description_embedder = Description_embedder(**params['env_params']['description_embedder_params'])

    item_type_embedder = OneHotEncoder(sparse=False)
    item_type_embedder.fit(np.array(ITEM_TYPE).reshape(-1, 1))

    from functools import partial

    transformer = partial(preprocess_raw_observation, description_embedder=description_embedder,
                          item_type_embedder=item_type_embedder, raw_state_size=3,
                          pytorch=True, device=params['device'])

    # dts = StateDataset.from_files(state_path, raw_state_transformer=transformer, max_size=5000)
    # dts = EpisodeDataset.from_files(episode_path, raw_state_transformer=transformer)
    # dts_loader = DataLoader(dts, batch_size=len(dts))
    # train_dts, test_dts = dts.split(train_test_ratio = 0.85)
    train_episode = '/home/nicolas/PycharmProjects/imagineIoT/results/episodes_records4.jbl'
    test_episode = '/home/nicolas/PycharmProjects/imagineIoT/results/episodes_records3.jbl'
    # train_dts = EpisodeDataset.from_files(train_episode, raw_state_transformer=transformer)
    test_dts = EpisodeDataset.from_files(test_episode, raw_state_transformer=transformer, max_size=5000)

    # sampler = ImbalancedDatasetSampler(train_dts, num_samples=20000, pos_weight=0.2)
    # train_loader = DataLoader(dataset=train_dts, sampler=sampler, batch_size=128)
    test_loader = DataLoader(test_dts, batch_size=len(test_dts))
    test_batch = next(iter(test_loader))

    language_model = LanguageModel(**params['language_model_params'])
    reward_function = RewardModel(context_model=params['model_params']['context_model'],
                                  language_model=language_model, reward_params=params['reward_model_params'])
    reward_function.to(params['device'])

    from torch import optim

    loss_func = nn.BCELoss()
    optimizer = optim.Adam(reward_function.parameters())

    from tqdm import tqdm

    for epoch in range(20):
        for i, batch in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            reward = reward_function(state=batch['state'], instructions=batch['instruction']).view(-1)
            loss = loss_func(reward, batch['reward'].float().to(params['device']))
            print(f'{epoch} {i} loss: {loss.item()}')
            loss.backward()
            clip_grad_norm_(reward_function.parameters(), 1)
            optimizer.step()
