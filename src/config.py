import torch
import torch.nn as nn
from torch import optim

import torchtext
from simulator.Items import ITEM_TYPE
from simulator.Action import ACTION_SPACE
from simulator.utils import color_list, percent_level

instruction_embedding = 100
description_embedding = 100
state_encoding_size = 3  # size of the vector in which is encoded the value of a channel
state_embedding_size = state_encoding_size + description_embedding + len(ITEM_TYPE)
action_embedding = 67

device = 'cuda' if torch.cuda.is_available() else 'cpu'

params = {
    'env_params': {
        'state_encoding_size': state_encoding_size,
        'description_embedder_params': {
            'embedding': 'glove',
            'dimension': description_embedding,
            'reduction': 'mean',
            'authorize_cache': True
        }
    },
    'model_params': {
        'instruction_embedding': instruction_embedding,
        'state_embedding': state_embedding_size,  # TODO
        'action_embedding_size': action_embedding,  # TODO
        'raw_action_size': {
            'action_description_embedding': description_embedding,
            'action_standard_embedding': len(ACTION_SPACE),
            'color': len(color_list),
            'level': len(percent_level)
        },
        'net_params': {
            'hidden1_out': 512,
            'hidden2_out': 256,
        }
    },
    'goal_sampler_params': {
        'goal_sampling_stategy': 'random',
        'oracle_strategy': 'exhaustive_feedback'
    },
    'exploration_params': {
        'start_eps': 0.9,
        'min_eps': 0.05,
        'eps_decay': 200
    },
    'replay_buffer_params': {
        'max_size': 10000,
    },
    'discount_factor': 0.999,
    'batch_size': 128,
    # 'loss': nn.SmoothL1Loss(reduction='mean'),
    'loss': nn.functional.smooth_l1_loss,
    'optimizer': optim.Adam,
    'optimizer_params': {},
    'language_model_params': {
        'type': 'lstm',
        'embedding_size': 100,
        'linear1_out': 256,
        'out_features': instruction_embedding,
        'vocab': torchtext.vocab.GloVe(name='6B', dim=100),
        'vocab_size': 500,
        'device': device
    },
    'target_update_frequence': 10,
    'device': device,
}
