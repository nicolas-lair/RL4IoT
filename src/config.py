import logging
from pathlib import Path
from collections import namedtuple

import joblib
import torchtext
from torch import optim
import torch.nn as nn

from logger import update_logger
from simulator.Items import ITEM_TYPE
from simulator.Action import ACTION_SPACE
from simulator.discrete_parameters import color_list, N_LEVELS, TVchannels_list
from architecture.contextnet import DeepSetStateNet
from architecture.dqn import FullNet
from utils import extend_dict

ThingParam = namedtuple('ThingParam', ('Class', 'Params'))
vector_cache = '/home/nicolas/PycharmProjects/RL4IoT/.vector_cache'

word_embedding_size = 50
instruction_embedding = 50
thing_description_embedding = 25
channel_description_embedding = 25

value_encoding_size = 3  # size of the vector in which is encoded the value of a channel
action_embedding = 30

one_hot_thing_size = 7
one_hot_channel_size = 7

vocab_for_word_embedding = torchtext.vocab.GloVe(name='6B', dim=word_embedding_size, cache=vector_cache)

description_embedder_type = 'one_hot'
if description_embedder_type == 'glove_mean':
    thing_description_embedding = channel_description_embedding = word_embedding_size
elif description_embedder_type == 'one_hot':
    thing_description_embedding, channel_description_embedding = one_hot_thing_size, one_hot_channel_size

state_embedding_size = value_encoding_size + thing_description_embedding + channel_description_embedding + len(
    ITEM_TYPE)

policy_context_archi = DeepSetStateNet
model_archi = FullNet
change_focus_during_episode = True


def prepare_simulation(simulation_name='default'):
    base_folder = Path('../results/').joinpath(simulation_name)
    try:
        existing_simulation = [int(idx.name) for idx in base_folder.iterdir()]
        max_idx = max(existing_simulation) if existing_simulation else -1
        next_idx = max_idx + 1
    except FileNotFoundError:
        next_idx = 0

    path_dir = base_folder.joinpath(str(next_idx))
    path_dir.mkdir(parents=True)
    simulation_id = f'{simulation_name}_{next_idx}'
    return path_dir, simulation_id


def generate_env_params(thing):
    env_params = dict(
        max_episode_length=2,
        ignore_exec_action=True,
        allow_do_nothing=True,
        filter_state_during_episode=change_focus_during_episode,
        thing_params=thing,
        episode_reset=True
    )
    return env_params


def generate_description_embedder_params(desc_embedder=description_embedder_type):
    description_embedder_params = dict(type=desc_embedder)

    if desc_embedder == 'one_hot':
        description_embedder_params.update(
            thing_list=[],  # TODO
            channel_list=[],
            max_thing=thing_description_embedding,
            max_channel=channel_description_embedding
        )
    elif desc_embedder == 'glove_mean':
        description_embedder_params.update(
            vocab=vocab_for_word_embedding,
            reduction='mean',
        )
    elif desc_embedder == 'projection':
        description_embedder_params.update(
            vocab=vocab_for_word_embedding,
            embedding_size=thing_description_embedding,
        )
    elif desc_embedder == 'learned_lm':
        description_embedder_params.update(
            embedding_size=thing_description_embedding,
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


def generate_model_params(context_architeture=policy_context_archi):
    # Build context net parameters
    context_net_params = dict(instruction_embedding=instruction_embedding,
                              state_embedding=state_embedding_size,
                              hidden_state_size=action_embedding,
                              aggregate='mean')
    if issubclass(context_architeture, DeepSetStateNet):
        scaler_layer_params = dict(hidden_size=64, output_size=128, last_activation='relu')
        # scaler_layer_params = dict(hidden_size=128, output_size=256, last_activation='relu')
        context_net_params.update(scaler_layer_params=scaler_layer_params)
    elif ('FlatStateNet' in str(context_architeture)) or ('AttentionFlatState' in str(context_architeture)):
        pass
    else:
        raise NotImplementedError()

    model_params = dict(
        context_model=context_architeture,
        action_embedding_size=action_embedding,
        net_params=dict(
            q_network=dict(
                hidden1_out=128,
                hidden2_out=128
                # hidden1_out=256,
                # hidden2_out=256
            ),
            context_net=context_net_params,
        )
    )

    return model_params


def generate_action_model_params(desc_embedder=description_embedder_type):
    merge_thing_action_embedding = (desc_embedder != 'one_hot')
    action_model_params = dict(
        use_attention=True,
        raw_action_size=dict(
            thing=thing_description_embedding,
            channel=channel_description_embedding,
            openHAB_action=len(ACTION_SPACE),
            setPercent_params=N_LEVELS,
            setHSB_params=len(color_list),
            setString_params=len(TVchannels_list),
        ),
        out_features=action_embedding,  # TODO
        merge_thing_action_embedding=merge_thing_action_embedding
    )
    if action_model_params['use_attention']:
        action_model_params.update(lm_embedding_size=instruction_embedding)
    return action_model_params


def generate_reward_params(context_architecture=DeepSetStateNet):
    reward_net_params = dict(instruction_embedding=instruction_embedding,
                             hidden_state_size=0,
                             state_embedding=state_embedding_size + value_encoding_size,
                             aggregate='mean')
    if 'DeepSetStateNet' in str(context_architecture):
        reward_net_params.update(scaler_layer_params=dict(hidden1_out=256, latent_out=512, last_activation='relu'))

    reward_fit_params = dict(
        optimizer=optim.Adam,
        loss=nn.BCELoss,
        batch_size=128,
        n_epoch=250,
        sampler_params=dict(num_samples=8000, pos_weight=0.2),
    )
    reward_params = dict(context_model=context_architecture,
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
        reward_params=generate_reward_params(context_architecture=context_archi),
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


def generate_trainer_params(things_list, simulation_name='default_simulation', use_pretrained_language_model=False,
                            save_path=True, device='cuda', dqn_loss='mse', context_architecture=policy_context_archi,
                            **kwargs):
    path_dir, simulation_id = prepare_simulation(simulation_name) if save_path else (None, simulation_name)

    # Instantiate the param dict
    params = dict(
        simulation_name=simulation_name,
        env_params=generate_env_params(thing=things_list),
        model_archi=model_archi,
        model_params=generate_model_params(context_architeture=context_architecture),
        action_model_params=generate_action_model_params(description_embedder_type),
        state_embedder_params=generate_state_embedder_params(),
        description_embedder_params=generate_description_embedder_params(desc_embedder=description_embedder_type),
        reward_params=generate_reward_params(context_architecture=context_architecture),
        language_model_params=generate_language_model_params(device=device,
                                                             use_pretrained_model=use_pretrained_language_model),
        goal_sampler_params=dict(
            goal_sampling_stategy='random',
            oracle_strategy='exhaustive_feedback'
        ),
        exploration_params=dict(
            start_eps=0.9,
            min_eps=0.05,
            eps_decay=300 # 500
        ),
        replay_buffer_params=dict(
            per=True,
            max_size=20000,
            alpha=0.5,
            beta=0.6,
            prior_eps=1e-6
        ),
        oracle_params=dict(
            absolute_instruction=True,
            relative_instruction=True
        ),
        use_double_dqn=True,
        discount_factor=0.95,
        batch_size=128,
        loss=dqn_loss,
        optimizer=optim.Adam,
        optimizer_params=dict(),  # TODO optimize
        # lr_scheduler=optim.lr_scheduler.ReduceLROnPlateau,
        lr_scheduler=None,
        lr_scheduler_params=dict(mode='min', patience=300, min_lr=1e-8, factor=0.5),
        logger=generate_logger_params(simulation_id),
        n_episode=10000,
        target_update_frequence=20,
        device=device,
        test_frequence=100,
        n_iter_test=25,
        tqdm=False,
        save_directory=path_dir,

    )
    for k, v in kwargs.items():
        extend_dict(params, k, v)
    return params


def generate_proc_gen_eval_params(things_list, simulation_name='default_simulation',
                                  use_pretrained_language_model=False, save_path=True, device='cuda', dqn_loss='mse',
                                  context_architecture=policy_context_archi, **kwargs):
    params = generate_trainer_params(things_list=things_list, simulation_name=simulation_name,
                                     use_pretrained_language_model=use_pretrained_language_model, save_path=save_path,
                                     device=device, dqn_loss=dqn_loss, context_architecture=context_architecture,
                                     **kwargs)
    params['new_objects_threshold'] = (5, 0.95)
    return params


def format_config(config):
    def aux(d, out_):
        for k, v in d.items():
            if isinstance(v, dict):
                out_[k] = aux(v, {})
            elif isinstance(v, (str, int, bool)):
                out_[k] = v
            else:
                out_[k] = str(v)
        return out_

    out = {}
    aux(config, out)
    return out


def save_config(config, file_name='simulation_params.jbl'):
    out = format_config(config)
    joblib.dump(out, config["save_directory"].joinpath(file_name))


def setup_new_simulation(params):
    path_dir, simulation_id = prepare_simulation(params['simulation_name'])
    params['save_directory'] = path_dir
    update_logger(log_path=path_dir, simulation_id=simulation_id)
    # adapter = update_log_file_path(log_path=path_dir, simulation_id=simulation_id)
