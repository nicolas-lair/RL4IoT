import os
import re
import logging
from collections import namedtuple

import joblib
import torchtext
from torch import optim
import torch.nn as nn

from logger import update_logger
from simulator.Items import ITEM_TYPE
from simulator.Action import ACTION_SPACE
from simulator.discrete_parameters import color_list, N_LEVELS, TVchannels_list
from architecture.contextnet import DeepSetStateNet, FlatStateNet, AttentionFlatState, DoubleAttDeepSet
from architecture.dqn import FullNet, FullNetWithGatedAttention
from simulator.Thing import PlugSwitch
from TV_thing import LGTV
from simulator.lighting_things import AdorneLight, BigAssFan, HueLightBulb, SimpleLight, \
    StructuredHueLight

ThingParam = namedtuple('ThingParam', ('Class', 'Params'))
vector_cache = '/home/nicolas/PycharmProjects/RL4IoT/.vector_cache'

word_embedding_size = 50
instruction_embedding = 50
description_embedding = 25
value_encoding_size = 3  # size of the vector in which is encoded the value of a channel
action_embedding = 30

vocab_for_word_embedding = torchtext.vocab.GloVe(name='6B', dim=word_embedding_size, cache=vector_cache)

description_embedder_type = 'projection'
if description_embedder_type == 'glove_mean':
    description_embedding = word_embedding_size

state_embedding_size = value_encoding_size + 2 * description_embedding + len(ITEM_TYPE)

policy_context_archi = DeepSetStateNet
model_archi = FullNet
filter_state_during_episode = True


def prepare_simulation(simulation_name):
    base_folder = '../results/'
    base_name = f'simulation_{simulation_name}' + (len(simulation_name) > 0) * '_'

    simulation_list = [name for name in os.listdir(base_folder) if re.match(base_name + r"[0-9]+", name)]
    if simulation_list:
        simulation_list = [int(name.split('_')[-1]) for name in simulation_list]
        sim_id = max([int(id) for id in simulation_list]) + 1

    else:
        sim_id = 0
    simulation_id = f'{simulation_name}_{sim_id}'
    path_dir = os.path.join(base_folder, base_name + f'{sim_id}/')
    os.mkdir(path_dir)
    return path_dir, simulation_id


env_params = dict(
    max_episode_length=2,
    ignore_exec_action=True,
    allow_do_nothing=True,
    filter_state_during_episode=filter_state_during_episode,
    thing_params=[
        # ThingParam(SimpleLight, dict(name='plug', simple=True)),
        # ThingParam(SimpleLight, dict(name='switch', simple=True)),
        # ThingParam(AdorneLightBulb, dict(name="light", simple=True, always_on=False)),
        ThingParam(BigAssFanLightBulb, dict(name="light", simple=True, always_on=False)),
        # ThingParam(StructuredHueLight, dict(name="colored light", simple=True, always_on=False)),

        # ThingParam(PlugSwitch,
        #            dict(name='first plug',
        #                 description='This is a plug',
        #                 is_visible=True,
        #                 init_type='random',
        #                 init_params=dict())
        #            ),
        # ThingParam(LightBulb,
        #            dict(name='first light bulb',
        #                 description='This is a light bulb',
        #                 is_visible=True,
        #                 init_type='random',
        #                 init_params=dict())
        #            ),
        # ThingParam(LGTV, dict(name='television',
        #                       description='This is a television',
        #                       is_visible=True,
        #                       init_type='random',
        #                       init_params=dict())
        #            )
    ],
)
def generate_env_params():
    return env_params


def generate_description_embedder_params(description_embedder_type=description_embedder_type):
    if description_embedder_type == 'glove_mean':
        description_embedder_params = dict(
            type=description_embedder_type,
            vocab=vocab_for_word_embedding,
            reduction='mean',
        )
    elif description_embedder_type == 'projection':
        description_embedder_params = dict(
            type=description_embedder_type,
            vocab=vocab_for_word_embedding,
            embedding_size=description_embedding,
        )
    elif description_embedder_type == 'learned_lm':
        description_embedder_params = dict(
            type=description_embedder_type,
            embedding_size=description_embedding,
        )
    else:
        raise NotImplementedError
    return description_embedder_params


def generate_state_embedder_params():
    state_embedder_params = dict(
        item_type=ITEM_TYPE,
        value_encoding_size=value_encoding_size,
        use_cache=True
    )
    return state_embedder_params


def generate_language_model_params(device='cuda', use_pretrained_model=False):
    if use_pretrained_model:
        # pretrained_model_path = '/home/nicolas/PycharmProjects/RL4IoT/results/learned_language_model.pth'
        pretrained_model_path = '/home/nicolas/PycharmProjects/RL4IoT/results/lm_do_nothing_test.pth'
    else:
        pretrained_model_path = None

    language_model_params = dict(
        type='lstm',
        embedding_size=word_embedding_size,
        linear1_out=256,
        out_features=instruction_embedding,
        vocab=vocab_for_word_embedding,
        # vocab=False,
        vocab_size=500,
        device=device,
        pretrained_model=pretrained_model_path,
        freq_update=1  # (one of every X episode : 1 means always)
    )
    return language_model_params


def generate_reward_params(archi=DeepSetStateNet):
    reward_net_params = dict(instruction_embedding=instruction_embedding,
                             hidden_state_size=0,
                             state_embedding=state_embedding_size + value_encoding_size,
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


def generate_logger_params(simulation_id):
    return dict(
        level=logging.INFO,
        console=True,
        log_file=True,
        simulation_id=simulation_id
    )


def get_data_collection_params(name='data_collection_'):
    from datetime import datetime
    env_params = generate_env_params()
    env_params['allow_do_nothing'] = True
    _, simulation_id = prepare_simulation(name)
    params = dict(
        name=name + str(datetime.now()).split('.')[0],
        env_params=env_params,
        logger=dict(
            level=logging.INFO,
            console=True,
            log_file=False,
            simulation_id=simulation_id,
        ),
    )
    return params


def get_reward_training_params(name=None, device='cuda'):
    context_archi = DeepSetStateNet
    _, simulation_id = prepare_simulation(name)
    params = dict(
        language_model_params=generate_language_model_params(device=device, use_pretrained_model=False),
        state_embedder_params=generate_state_embedder_params(),
        description_embedder_params=generate_description_embedder_params(),
        context_archi=context_archi,
        reward_params=generate_reward_params(archi=context_archi),
        logger=dict(
            level=logging.INFO,
            console=True,
            log_file=False,
            simulation_id=simulation_id,
        ),
        device=device,
        lm_save_path=f'/home/nicolas/PycharmProjects/RL4IoT/results/lm_{name}.pth'
    )
    return params


def generate_params(simulation_name='default_simulation', use_pretrained_language_model=False, save_path=True,
                    device='cuda', dqn_loss='mse'):
    env_params = generate_env_params()
    language_model_params = generate_language_model_params(device=device,
                                                           use_pretrained_model=use_pretrained_language_model)

    device = device
    reward_params = generate_reward_params(archi=policy_context_archi)
    simulation_name = simulation_name
    if save_path:
        path_dir, simulation_id = prepare_simulation(simulation_name)
    else:
        path_dir, simulation_id = simulation_name

    # Build context net parameters
    context_net_params = dict(instruction_embedding=instruction_embedding,
                              state_embedding=state_embedding_size,
                              hidden_state_size=action_embedding,
                              aggregate='mean')
    if issubclass(policy_context_archi, DeepSetStateNet):
        scaler_layer_params = dict(hidden_size=256, output_size=512, last_activation='relu')
        context_net_params.update(scaler_layer_params=scaler_layer_params)
    elif ('FlatStateNet' in str(policy_context_archi)) or ('AttentionFlatState' in str(policy_context_archi)):
        pass
    else:
        raise NotImplementedError()

    # Instantiate the param dict
    params = dict(
        simulation_name=simulation_name,
        env_params=env_params,
        model_archi=model_archi,
        model_params=dict(
            context_model=policy_context_archi,
            action_embedding_size=action_embedding,  # TODO
            raw_action_size=dict(
                description_node=description_embedding,
                openHAB_action=len(ACTION_SPACE),
                setPercent_params=N_LEVELS,
                setHSB_params=len(color_list),
                setString_params=len(TVchannels_list),
            ),
            net_params=dict(
                q_network=dict(
                    hidden1_out=512,
                    hidden2_out=256
                ),
                context_net=context_net_params,
            ),
        ),
        state_embedder_params=generate_state_embedder_params(),
        description_embedder_params=generate_description_embedder_params(),
        reward_params=reward_params,
        goal_sampler_params=dict(
            goal_sampling_stategy='random',
            oracle_strategy='exhaustive_feedback'
        ),
        exploration_params=dict(
            start_eps=0.9,
            min_eps=0.05,
            eps_decay=500
        ),
        replay_buffer_params=dict(
            per=True,
            max_size=20000,
            alpha=0.5,
            beta=0.6,
            prior_eps=1e-6
        ),
        use_double_dqn=True,
        discount_factor=0.95,
        batch_size=128,
        loss=dqn_loss,
        optimizer=optim.Adam,
        optimizer_params=dict(),  # TODO optimize
        # lr_scheduler=optim.lr_scheduler.ReduceLROnPlateau,
        lr_scheduler=None,
        lr_scheduler_params=dict(mode='min'),
        language_model_params=language_model_params,
        logger=generate_logger_params(simulation_id),
        n_episode=15000,
        target_update_frequence=20,
        device=device,
        episode_reset=True,
        test_frequence=250,
        n_iter_test=25,
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


def setup_new_simulation(params):
    path_dir, simulation_id = prepare_simulation(params['simulation_name'])
    params['save_directory'] = path_dir
    update_logger(log_path=path_dir, simulation_id=simulation_id)
    # adapter = update_log_file_path(log_path=path_dir, simulation_id=simulation_id)
