import os
import glob
import logging
from collections import namedtuple

import joblib
import torchtext
from torch import optim
import torch.nn as nn

from logger import update_log_file_path
from simulator.Items import ITEM_TYPE
from simulator.Action import ACTION_SPACE
from simulator.utils import color_list, N_LEVELS
from architecture.contextnet import DeepSetStateNet, FlatStateNet, AttentionFlatState
from simulator.Thing import PlugSwitch, LGTV
from simulator.lighting_things import AdorneLightBulb, BigAssFanLightBulb, HueLightBulb

ThingParam = namedtuple('ThingParam', ('Class', 'Params'))

word_embedding_size = 50
instruction_embedding = 40
description_embedding = 50
state_encoding_size = 3  # size of the vector in which is encoded the value of a channel
state_embedding_size = state_encoding_size + description_embedding + len(ITEM_TYPE)
action_embedding = 50

vector_cache = '/home/nicolas/PycharmProjects/imagineIoT/.vector_cache'


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


def generate_env_params():
    env_params = dict(
        max_episode_length=2,
        ignore_exec_action=True,
        allow_do_nothing=True,
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
                       ),
            ThingParam(LGTV, dict(name='television',
                                  description='This is a television',
                                  is_visible=True,
                                  init_type='random',
                                  init_params=dict())
                       )
        ],
    )
    return env_params


def generate_language_model_params(device='cuda', use_pretrained_model=False):
    if use_pretrained_model:
        pretrained_model_path = '/home/nicolas/PycharmProjects/imagineIoT/results/learned_language_model.pth'
    else:
        pretrained_model_path = None

    language_model_params = dict(
        type='lstm',
        embedding_size=word_embedding_size,
        linear1_out=256,
        out_features=instruction_embedding,
        vocab=torchtext.vocab.GloVe(name='6B', dim=word_embedding_size,
                                    cache=vector_cache
                                    ),
        vocab_size=500,
        device=device,
        pretrained_model=pretrained_model_path
    )
    return language_model_params


def generate_reward_params(archi=DeepSetStateNet):
    reward_net_params = dict(instruction_embedding=instruction_embedding,
                             hidden_state_size=0,
                             state_embedding=state_embedding_size + state_encoding_size,
                             aggregate='mean')
    if 'DeepSetStateNet' in str(archi):
        reward_net_params.update(scaler_layer_params=dict(hidden1_out=256, latent_out=512, last_activation='relu'))

    reward_fit_params = dict(
        optimizer=optim.Adam,
        loss=nn.BCELoss,
        batch_size=128,
        n_epoch=250,
        sampler_params=dict(num_samples=8000, pos_weight=0.2),
    )
    reward_params = dict(context_model=archi,
                         net_params=reward_net_params,
                         fit_params=reward_fit_params)
    return reward_params


def get_data_collection_params(name='data_collection_'):
    from datetime import datetime
    env_params = generate_env_params()
    env_params['allow_do_nothing'] = False
    params = dict(
        name=name + str(datetime.now()).split('.')[0],
        env_params=env_params,
        logger=dict(
            level=logging.INFO,
            console=True,
            log_file=False,
        ),
    )
    return params


def get_reward_training_params(name=None, device='cuda'):
    context_archi = DeepSetStateNet

    env_params = generate_env_params()
    reward_params = generate_reward_params(archi=context_archi)
    language_model_params = generate_language_model_params(device=device, use_pretrained_model=False)
    params = dict(
        language_model_params=language_model_params,
        description_embedder_params=env_params['description_embedder_params'],
        context_archi=context_archi,
        reward_params=reward_params,
        logger=dict(
            level=logging.INFO,
            console=True,
            log_file=False,
        ),
        device=device
    )
    return params


def generate_params(simulation_name='default_simulation', use_pretrained_language_model=False, save_path=True,
                    device='cuda', dqn_loss='mse'):
    env_params = generate_env_params()
    language_model_params = generate_language_model_params(device=device,
                                                           use_pretrained_model=use_pretrained_language_model)

    device = device
    # device = 'cpu'

    policy_context_archi = DeepSetStateNet
    reward_params = generate_reward_params(archi=policy_context_archi)

    simulation_name = simulation_name
    if save_path:
        path_dir = prepare_simulation(simulation_name)
    else:
        path_dir = None

    # Build context net parameters
    context_net_params = dict(instruction_embedding=instruction_embedding,
                              state_embedding=state_embedding_size,
                              hidden_state_size=action_embedding,
                              aggregate='mean')
    if 'DeepSetStateNet' in str(policy_context_archi):
        scaler_layer_params = dict(hidden1_out=256, latent_out=512, last_activation='relu')
        context_net_params.update(scaler_layer_params=scaler_layer_params)
    elif ('FlatStateNet' in str(policy_context_archi)) or ('AttentionFlatState' in str(policy_context_archi)):
        pass
    else:
        raise NotImplementedError

    # Instantiate the param dict
    params = dict(
        simulation_name=simulation_name,
        env_params=env_params,
        model_params=dict(
            context_model=policy_context_archi,
            action_embedding_size=action_embedding,  # TODO
            raw_action_size=dict(
                description_node=description_embedding,
                openHAB_action=len(ACTION_SPACE),
                color_params=len(color_list),
                level_params=N_LEVELS,
            ),
            net_params=dict(
                q_network=dict(
                    hidden1_out=512,
                    hidden2_out=256
                ),
                context_net=context_net_params,
            )
        ),
        reward_params=reward_params,
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
        loss=dqn_loss,
        optimizer=optim.Adam,
        optimizer_params=dict(),  # TODO optimize
        language_model_params=language_model_params,
        logger=dict(
            level=logging.INFO,
            console=True,
            log_file=True,
        ),
        n_episode=30000,
        target_update_frequence=250,
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
