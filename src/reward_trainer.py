from functools import partial
from pprint import pformat

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from architecture.reward import EpisodeDataset, RewardModel, ImbalancedDatasetSampler
from config import get_reward_training_params, format_config
from logger import set_logger_handler, rootLogger

params = get_reward_training_params(name='do_nothing_test')

set_logger_handler(rootLogger, **params['logger'])
logger = rootLogger.getChild(__name__)
logger.setLevel(10)


class RewardTrainer:
    def __init__(self, model, optimizer, fit_params, device):
        self.device = device
        self.model = model
        self.model.to(self.device)

        self.optimizer = optimizer(model.parameters())
        self.loss_function = fit_params['loss']()
        self.batch_size = fit_params['batch_size']
        self.n_epoch = fit_params['n_epoch']
        self.sampler_params = fit_params['sampler_params']

        self.metrics = None

    def compute_loss(self, predicted_y, true_y):
        true_y = true_y.float().to(self.device)
        loss = self.loss_function(predicted_y, true_y)
        loss.backward()
        return loss

    def fit_epoch(self, train_loader):
        loss = []
        for i, batch in enumerate(train_loader):
            self.optimizer.zero_grad()
            predicted_y = self.model(state=batch['state'], instructions=batch['instruction']).view(-1)
            loss.append(self.compute_loss(predicted_y, batch['reward']))
            clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()

        loss = torch.stack(loss).mean()
        return loss

    def run(self, train_dts, test_batch, data_stats):
        sampler = ImbalancedDatasetSampler(train_dts, **self.sampler_params)
        train_loader = DataLoader(dataset=train_dts, sampler=sampler, batch_size=self.batch_size)

        loss_list = []
        metrics = []
        self.metrics = loss_list, metrics
        for epoch in range(self.n_epoch):
            loss_list.append(self.fit_epoch(train_loader))
            logger.info(f'Epoch: {epoch}: loss: {loss_list[-1].item()}')

            metrics.append(self.eval(model=self.model,
                                     test_batch=test_batch,
                                     epoch=epoch,
                                     data_stats=data_stats)
                           )
            self.metrics = loss_list, metrics
        metrics = pd.concat(metrics, axis=0, ignore_index=True)
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

    def save_reward_model(self, path):
        torch.save(self.model.state_dict(), path)

    def save_language_model(self, path):
        torch.save(self.model.language_model.state_dict(), path)


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

    logger.info(f'Simulation params:\n {pformat(format_config(params))}')

    EpisodeRecord = namedtuple('EpisodeRecord', ('initial_state', 'final_state', 'instruction', 'reward'))

    episode_path = '/home/nicolas/PycharmProjects/RL4IoT/results/episodes_records_data_collection_do_nothing_test.jbl'

    transformer = get_transformer_function(params)

    logger.info('Loading datasets')
    dts = EpisodeDataset.from_files(episode_path, raw_state_transformer=transformer)

    logger.info('Splitting between train and test')
    train_dts, test_dts = dts.split(train_test_ratio=0.85, max_test=5000)

    # train_episode = '/home/nicolas/PycharmProjects/RL4IoT/results/episodes_records4.jbl'
    # test_episode = '/home/nicolas/PycharmProjects/RL4IoT/results/episodes_records3.jbl'
    # train_dts = EpisodeDataset.from_files(train_episode, raw_state_transformer=transformer)
    # test_dts = EpisodeDataset.from_files(test_episode, raw_state_transformer=transformer, max_size=10000)

    logger.info('Creating test batch')
    test_loader = DataLoader(test_dts, batch_size=len(test_dts))
    test_batch = next(iter(test_loader))

    language_model = LanguageModel(**params['language_model_params'])
    reward = RewardModel(context_model=params['reward_params']['context_model'], language_model=language_model,
                         reward_params=params['reward_params']['net_params'])
    reward.load_state_dict(torch.load('/home/nicolas/PycharmProjects/RL4IoT/results/reward_do_nothing_test.pth'))

    evaluator = RewardTrainer(model=reward,
                              optimizer=params['reward_params']['fit_params']['optimizer'],
                              fit_params=params['reward_params']['fit_params'],
                              device=params['device'])

    logger.info('Run training')
    metrics, loss_list = evaluator.run(train_dts, test_batch, data_stats=False)

    logger.info('Saving language model')
    evaluator.save_language_model(params['lm_save_path'])


# instruction with no positive reward
# ['You set the color of intermediate light bulb to pink',
#  'You set the color of level one light bulb to yellow',
#  'the light temperature of level one light bulb is now warm',
#  'the light temperature of level one light bulb is now cold',
#  'You set the color of intermediate light bulb to yellow',
#  'You made the light of level one light bulb colder',
#  'You set the color of intermediate light bulb to green',
#  'You set the color of level one light bulb to orange',
#  'You made the light of level one light bulb warmer',
#  'You set the color of intermediate light bulb to red',
#  'You set the color of level one light bulb to red',
#  'You set the color of intermediate light bulb to purple',
#  'You set the color of intermediate light bulb to blue',
#  'You set the color of level one light bulb to green',
#  'the light temperature of level one light bulb is now very warm',
#  'the light temperature of level one light bulb is now average',
#  'You set the color of level one light bulb to blue',
#  'the light temperature of level one light bulb is now very cold',
#  'You set the color of intermediate light bulb to orange',
#  'You set the color of level one light bulb to purple',
#  'You set the color of level one light bulb to pink']