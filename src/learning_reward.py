import sys

from collections import namedtuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import joblib
from torch import nn
from sklearn.preprocessing import OneHotEncoder

from architecture.reward import LearnedReward, StateDataset
from architecture.language_model import LanguageModel
from simulator.description_embedder import Description_embedder
from simulator.Environment import preprocess_raw_observation
from simulator.Items import ITEM_TYPE
from config import generate_params

params = generate_params(save_path=False)

EpisodeRecord = namedtuple('EpisodeRecord', ('initial_state', 'final_state', 'instruction', 'reward'))
episode_path = 'results/episodes_records.jbl'
episodes = joblib.load(episode_path)

StateRecord = namedtuple('StateRecord', ('state', 'instruction', 'reward'))
state_path = 'results/state_records.jbl'
states = joblib.load(state_path)

description_embedder = Description_embedder(**params['env_params']['description_embedder_params'])

item_type_embedder = OneHotEncoder(sparse=False)
item_type_embedder.fit(np.array(ITEM_TYPE).reshape(-1, 1))

from functools import partial

transformer = partial(preprocess_raw_observation, description_embedder=description_embedder,
                      item_type_embedder=item_type_embedder, raw_state_size=3, pytorch=True, device='cuda:1')

dts = StateDataset.from_files(state_path, raw_state_transformer=transformer)
train_dts, test_dts = dts.split(train_test_ratio=0.7)
train_loader = DataLoader(train_dts, batch_size=5, shuffle=True)

language_model = LanguageModel(**params['language_model_params'])
reward = LearnedReward(context_model=params['model_params']['context_model'], language_model=language_model,
                       reward_params=params['reward_model_params'])
reward.to('cuda:1')

if __name__ == "__main__":
    for i, batch in enumerate(train_loader):
        # print(i)
        # print(len(batch['instruction']))
        # print(len(batch['reward']))
        # print(batch['state']['first plug']['switch_binary'].size())
        print(reward(state=batch['state'], instructions=batch['instruction']))

        if i == 0:
            break
