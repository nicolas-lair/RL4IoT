import os
import glob
import logging
from collections import namedtuple

import joblib
import torch
import torchtext
from torch import optim
import torch.nn as nn

from src.logger import update_log_file_path
from simulator.Items import ITEM_TYPE
from simulator.Action import ACTION_SPACE
from simulator.utils import color_list, percent_level
from architecture.contextnet import DeepSetStateNet, FlatStateNet, AttentionFlatState
from simulator.Thing import PlugSwitch, LightBulb

ThingParam = namedtuple('ThingParam', ('Class', 'Params'))


def prepare_simulation(simulation_name):
    if simulation_name == '':
        base_path = '../results/simulation_'
    else:
        base_path = f'../results/simulation_{simulation_name}_'

    l = glob.glob(base_path + '*')
    l = [name.split('_')[-1] for name in l if name.split('_')[-1].isdigit()]
    if l:
        sim_id = max([int(id) for id in l]) + 1
        # sim_id = max([int(name.split('_')[-1]) for name in l]) + 1
    else:
        sim_id = 0
    path_dir = base_path + f'{sim_id}/'
    os.mkdir(path_dir)
    return path_dir


def generate_params(save_path=True):
    word_embedding_size = 50
    instruction_embedding = 40
    description_embedding = 50
    state_encoding_size = 3  # size of the vector in which is encoded the value of a channel
    state_embedding_size = state_encoding_size + description_embedding + len(ITEM_TYPE)
    action_embedding = 50

    vector_cache = '/home/nicolas/PycharmProjects/imagineIoT/.vector_cache'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    policy_context_archi = DeepSetStateNet
    simulation_name = str(policy_context_archi).split("'")[-2].split('.')[-1]
    if save_path:
        path_dir = prepare_simulation(simulation_name)
    else:
        path_dir = None

    # Build context net parameters
    context_net_params = dict(instruction_embedding=instruction_embedding,
                              state_embedding=state_embedding_size,
                              hidden_state_size=action_embedding,
                              aggregate='mean')
    if simulation_name == 'DeepSetStateNet':
        scaler_layer_params = dict(hidden1_out=256, latent_out=512, last_activation='relu')
        context_net_params.update(scaler_layer_params)
    elif simulation_name in ['FlatStateNet', 'AttentionFlatState']:
        pass
    else:
        raise NotImplementedError

    reward_net_params = dict(instruction_embedding=instruction_embedding,
                             hidden_state_size=0,
                             state_embedding=state_embedding_size + state_encoding_size,
                             aggregate='mean')
    if simulation_name == 'DeepSetStateNet':
        reward_net_params.update(scaler_layer_params=dict(hidden1_out=256, latent_out=512, last_activation='relu'))

    reward_fit_params = dict(
        optimizer=optim.Adam,
        loss=nn.BCELoss,
        batch_size=128,
        n_epoch=100,
        sampler_params=dict(num_samples=20000, pos_weight=0.2),
    )
    # Instantiate the param dict
    params = dict(
        simulation_name=simulation_name,
        env_params=dict(
            state_encoding_size=state_encoding_size,
            description_embedder_params=dict(
                embedding='glove',
                word_embedding_params=dict(
                    name='6B',
                    dim=str(description_embedding),
                    cache=vector_cache
                ),
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
            context_model=policy_context_archi,
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
                context_net=context_net_params,
            )
        ),
        reward_params=dict(net_params=reward_net_params,
                           fit_params=reward_fit_params),
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
        discount_factor=0.9,
        batch_size=128,
        loss=nn.functional.smooth_l1_loss,
        optimizer=optim.Adam,
        optimizer_params=dict(),  # TODO optimize
        language_model_params=dict(
            type='lstm',
            embedding_size=word_embedding_size,
            linear1_out=256,
            out_features=instruction_embedding,
            vocab=torchtext.vocab.GloVe(name='6B', dim=word_embedding_size,
                                        cache=vector_cache
                                        ),
            vocab_size=500,
            device=device
        ),
        logger=dict(
            level=logging.INFO,
            console=True,
            log_file=True,
        ),
        n_episode=30000,
        target_update_frequence=100,
        device=device,
        episode_reset=True,
        test_frequence=300,
        n_iter_test=30,
        tqdm=False,
        save_directory=path_dir,
    )

    return params


def format_config(config):
    def aux(d, out):
        for k, v in d.items():
            if isinstance(v, dict):
                out[k] = aux(v, {})
            elif isinstance(v, (str, int, bool)):
                out[k] = v
            else:
                out[k] = str(v)
        return out

    out = {}
    aux(config, out)
    return out


def save_config(config, file_name='simulation_params.jbl'):
    out = format_config(config)
    joblib.dump(out, os.path.join(out["save_directory"], file_name))


def setup_new_simulation(logger, params):
    path_dir = prepare_simulation(params['simulation_name'])
    params['save_directory'] = path_dir
    update_log_file_path(logger, log_path=path_dir)
