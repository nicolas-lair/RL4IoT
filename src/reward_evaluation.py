from functools import partial

import numpy as np
from torch.utils.data import DataLoader
from architecture.reward import EpisodeDataset, RewardModel, Reward


def get_transformer_function(params):
    from simulator.Items import ITEM_TYPE
    from simulator.description_embedder import Description_embedder
    from simulator.Environment import preprocess_raw_observation

    description_embedder = Description_embedder(**params['env_params']['description_embedder_params'])

    item_type_embedder = OneHotEncoder(sparse=False)
    item_type_embedder.fit(np.array(ITEM_TYPE).reshape(-1, 1))

    transformer = partial(preprocess_raw_observation, description_embedder=description_embedder,
                          item_type_embedder=item_type_embedder, raw_state_size=3,
                          pytorch=True, device=params['device'])
    return transformer


if __name__ == "__main__":
    from collections import namedtuple
    from sklearn.preprocessing import OneHotEncoder

    from config import generate_params
    from architecture.language_model import LanguageModel

    StateRecord = namedtuple('StateRecord', ('state', 'instruction', 'reward'))
    EpisodeRecord = namedtuple('EpisodeRecord', ('initial_state', 'final_state', 'instruction', 'reward'))

    params = generate_params(save_path=False)
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
    reward = Reward(context_model=params['model_params']['context_model'], language_model=language_model,
                    device=params['device'], **params['reward_params'])
    reward.fit(train_dts)

