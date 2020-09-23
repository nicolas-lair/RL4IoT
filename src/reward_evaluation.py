from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from architecture.reward import EpisodeDataset, RewardModel, ImbalancedDatasetSampler
from config import get_reward_training_params
from logger import set_logger_handler, rootLogger

params = get_reward_training_params()

set_logger_handler(rootLogger, **params['logger'])
logger = rootLogger.getChild(__name__)
logger.setLevel(10)

class Net(nn.Module):
    """
    Cedric Or module
    """
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

class RewardTrainer:
    pass
class RewardComparision:
    def __init__(self, reward_list, fit_params, device):
        self.device = device
        self.model_list = reward_list
        for model in reward_list:
            model.to(self.device)

        self.optimizers_list = [fit_params['optimizer'](model.parameters()) for model in self.model_list]
        self.loss_function = fit_params['loss']()
        self.batch_size = fit_params['batch_size']
        self.n_epoch = fit_params['n_epoch']
        self.sampler_params = fit_params['sampler_params']

        self.metrics = None

    def zero_grad(self):
        for optim in self.optimizers_list:
            optim.zero_grad()

    def compute_reward(self, state, instructions):
        reward_list = []
        for model in self.model_list:
            reward_list.append(model(state, instructions).view(-1))
        return reward_list

    def compute_loss(self, reward_list, true_reward):
        true_reward = true_reward.float().to(self.device)
        loss_list = []
        for reward in reward_list:
            loss = self.loss_function(reward, true_reward)
            loss.backward()
            loss_list.append(loss)
        return np.array(loss_list)

    def clipping_grad_norm(self):
        for model in self.model_list:
            clip_grad_norm_(model.parameters(), 1)

    def step_optimizers(self):
        for optim in self.optimizers_list:
            optim.step()

    def fit_epoch(self, train_loader):
        loss = []
        for i, batch in enumerate(train_loader):
            self.zero_grad()
            reward_list = self.compute_reward(state=batch['state'], instructions=batch['instruction'])
            loss.append(self.compute_loss(reward_list, batch['reward']))
            self.clipping_grad_norm()
            self.step_optimizers()

        loss = np.stack(loss, axis=0)
        loss = loss.mean(axis=0)
        return loss

    def run(self, train_dts, test_batch, data_stats):
        sampler = ImbalancedDatasetSampler(train_dts, **self.sampler_params)
        train_loader = DataLoader(dataset=train_dts, sampler=sampler, batch_size=self.batch_size)

        loss_list = []
        metrics = {i: [] for i in range(len(self.model_list))}
        self.metrics = loss_list, metrics
        for epoch in range(self.n_epoch):
            loss_list.append(self.fit_epoch(train_loader))
            print(f'Epoch: {epoch}: loss: {[l.item() for l in loss_list[-1]]}')
            for i in range(len(self.model_list)):
                metrics[i].append(self.eval(model=self.model_list[i],
                                            test_batch=test_batch,
                                            epoch=epoch,
                                            data_stats=data_stats)
                                  )
            self.metrics = loss_list, metrics
        metrics = [pd.concat(metrics[i], axis=0, ignore_index=True) for i in range(len(self.model_list))]
        return metrics, loss_list

    @staticmethod
    def eval(model, test_batch, epoch, data_stats=False):
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
            precision = precision_score(g.true, g.pred, zero_division=0)
            recall = recall_score(g.true, g.pred, zero_division=0)
            f1 = f1_score(g.true, g.pred, zero_division=0)
            # scores = [accuracy, precision, recall, f1]
            # score_name = ['accuracy', 'precision', 'recall', 'f1_score']
            scores = [precision, recall, f1]
            score_name = ['precision', 'recall', 'f1_score']
            result = pd.DataFrame(scores).T
            result.columns = score_name
            return result

        def compute_metrics(g, data_stats=False):
            df_score = compute_classification_metrics(g)
            df_stats = compute_stats(g) if data_stats else pd.DataFrame()
            result = pd.concat([df_stats, df_score], axis=1)
            return result

        test_reward = model(state=test_batch['state'], instructions=test_batch['instruction'])  # .view(-1)
        pred_true_tensor = torch.cat(
            [(test_reward > 0.5).float().cpu().view(-1, 1), test_batch['reward'].float().view(-1, 1)],
            dim=1)
        df = pd.DataFrame(pred_true_tensor.detach().numpy(), columns=['pred', 'true'])
        df['instruction'] = test_batch['instruction']
        metrics = df.groupby('instruction').apply(compute_metrics, data_stats=data_stats)
        metrics = metrics.reset_index(level=1, drop=True)

        metrics_overall = compute_metrics(df, data_stats=False)
        metrics_overall.index = ['model']
        metrics = metrics.append(metrics_overall)
        metrics.reset_index(inplace=True)

        metrics['epoch'] = epoch
        return metrics

    def save_language_model(self):
        torch.save(self.model_list)


def get_transformer_function(params):
    from simulator.Items import ITEM_TYPE
    from simulator.description_embedder import Description_embedder
    from simulator.Environment import preprocess_raw_observation

    description_embedder = Description_embedder(**params['description_embedder_params'])

    item_type_embedder = OneHotEncoder(sparse=False)
    item_type_embedder.fit(np.array(ITEM_TYPE).reshape(-1, 1))

    transformer = partial(preprocess_raw_observation, description_embedder=description_embedder,
                          item_type_embedder=item_type_embedder, raw_state_size=3,
                          pytorch=True, device=params['device'])
    return transformer


if __name__ == "__main__":
    from collections import namedtuple
    from sklearn.preprocessing import OneHotEncoder

    from architecture.language_model import LanguageModel

    StateRecord = namedtuple('StateRecord', ('state', 'instruction', 'reward'))
    EpisodeRecord = namedtuple('EpisodeRecord', ('initial_state', 'final_state', 'instruction', 'reward'))

    episode_path = '/home/nicolas/PycharmProjects/imagineIoT/results/episodes_records3.jbl'

    transformer = get_transformer_function(params)

    # dts = StateDataset.from_files(state_path, raw_state_transformer=transformer, max_size=5000)
    # dts = EpisodeDataset.from_files(episode_path, raw_state_transformer=transformer)
    # dts_loader = DataLoader(dts, batch_size=len(dts))
    # train_dts, test_dts = dts.split(train_test_ratio = 0.85)
    train_episode = '/home/nicolas/PycharmProjects/imagineIoT/results/episodes_records4.jbl'
    test_episode = '/home/nicolas/PycharmProjects/imagineIoT/results/episodes_records3.jbl'

    train_dts = EpisodeDataset.from_files(train_episode, raw_state_transformer=transformer)

    test_dts = EpisodeDataset.from_files(test_episode, raw_state_transformer=transformer, max_size=10000)
    test_loader = DataLoader(test_dts, batch_size=len(test_dts))
    test_batch = next(iter(test_loader))

    language_model = LanguageModel(**params['language_model_params'])
    reward = RewardModel(context_model=params['reward_params']['context_model'], language_model=language_model,
                         reward_params=params['reward_params']['net_params'])

    evaluator = RewardComparision([reward], fit_params=params['reward_params']['fit_params'],
                                  device=params['device'])

    metrics, loss_list = evaluator.run(train_dts, test_batch, data_stats=False)
    torch.save(language_model.state_dict(), params['save_dir'])

    # reward_net_params = params['reward_params']['net_params'].copy()
    # reward_net_params.update(aggregate='diff_or',
    #                          scaler_layer_params=dict(hidden1_out=256, latent_out=1, last_activation='sigmoid')
    #                          )
    # prelearned_or_reward = RewardModel(context_model=params['model_params']['context_model'],
    #                                    language_model=language_model,
    #                                    reward_params=reward_net_params)

    # evaluator = RewardComparision([reward, prelearned_or_reward], fit_params=params['reward_params']['fit_params'],
    #                               device=params['device'])



