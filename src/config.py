import os
from datetime import datetime
from collections import namedtuple
import torch
import torch.nn as nn
from torch import optim

import torchtext
from simulator.Items import ITEM_TYPE
from simulator.Action import ACTION_SPACE
from simulator.utils import color_list, percent_level
from architecture.dqn import NoAttentionFlatQnet, AttentionFlatQnet, DeepSetQnet
from simulator.Thing import PlugSwitch, LightBulb

word_embedding_size = 100
instruction_embedding = 100
description_embedding = 100
state_encoding_size = 3  # size of the vector in which is encoded the value of a channel
state_embedding_size = state_encoding_size + description_embedding + len(ITEM_TYPE)
action_embedding = 67

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

ThingParam = namedtuple('ThingParam', ('Class', 'Params'))

params = dict(
    env_params=dict(
        state_encoding_size=state_encoding_size,
        description_embedder_params=dict(
            embedding='glove',
            dimension=description_embedding,
            reduction='mean',
            authorize_cache=True
        ),
        thing_params=[
            ThingParam(PlugSwitch,
                       dict(name='first plug',
                            description='This is a plug',
                            is_visible=True,
                            init_type='random',
                            init_params=dict())
                       ),
            ThingParam(LightBulb,
                       dict(name='first light bulb',
                            description='This is a light bulb',
                            is_visible=True,
                            init_type='random',
                            init_params=dict())
                       )
        ],
    ),
    model_params=dict(
        instruction_embedding=instruction_embedding,
        state_embedding=state_embedding_size,  # TODO
        action_embedding_size=action_embedding,  # TODO
        raw_action_size=dict(
            description_node=description_embedding,
            openHAB_action=len(ACTION_SPACE),
            color_params=len(color_list),
            level_params=len(percent_level)
        ),
        net_params=dict(
            q_network=dict(
                hidden1_out=512,
                hidden2_out=256
            ),
            # Scaler layer for DeepSet models
            scaler_layer=dict(
                hidden1_out=256,
                latent_out=512

            ),
        )
    ),
    goal_sampler_params=dict(
        goal_sampling_stategy='random',
        oracle_strategy='exhaustive_feedback'
    ),
    exploration_params=dict(
        start_eps=0.9,
        min_eps=0.05,
        eps_decay=200
    ),
    replay_buffer_params=dict(
        max_size=10000,
    ),
    discount_factor=0.999,
    batch_size=128,
    loss=nn.functional.smooth_l1_loss,
    optimizer=optim.Adam,
    optimizer_params={},
    language_model_params=dict(
        type='lstm',
        embedding_size=word_embedding_size,
        linear1_out=256,
        out_features=instruction_embedding,
        vocab=torchtext.vocab.GloVe(name='6B', dim=word_embedding_size),
        vocab_size=500,
        device=device
    ),
    dqn_architecture=DeepSetQnet,
    n_episode=20000,
    target_update_frequence=100,
    device=device,
    episode_reset=True,
    test_frequence=100,
    n_iter_test=30,
    verbose=True

)


def save_config(config):
    def aux(d, out):
        for k, v in d.items():
            if isinstance(v, dict):
                aux(v, out)
            elif isinstance(v, (str, int, bool)):
                out[k] = v
            else:
                out[k] = str(v)

    out = {}
    aux(config, out)
    return out
