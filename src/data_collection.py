from collections import namedtuple
import random

import numpy as np

import joblib

from src.logger import rootLogger, set_logger_handler
from src.config import get_data_collection_params
from simulator.Environment import IoTEnv4ML
from simulator.oracle import Oracle
from architecture.reward import EpisodeDataset

EpisodeRecord = namedtuple('EpisodeRecord', ('initial_state', 'final_state', 'instruction', 'reward'))
StateRecord = namedtuple('StateRecord', ('state', 'instruction', 'reward'))

params = get_data_collection_params()

set_logger_handler(rootLogger, **params['logger'])
logger = rootLogger.getChild(__name__)
logger.setLevel(10)


def run_episode(env, weights):
    _, available_actions = env.get_state_and_action()
    # Choose thing first to weight according to the number of instruction related to the thing
    action = np.random.choice(available_actions, p=[weights[t.name] for t in available_actions])
    logger.debug(f'action: {action.name}')
    (_, available_actions), _, done, _ = env.step(action=action)

    while not done:
        action_idx = random.randint(0, len(available_actions) - 1)
        action = available_actions[action_idx]
        logger.debug(f'action: {action.name}')
        (_, available_actions), _, done, _ = env.step(action=action)


def remove_key(state, key='embedding'):
    s = state.copy()
    if isinstance(s, dict):
        try:
            s.pop(key)
        except KeyError:
            pass

        for k, v in s.items():
            if isinstance(v, dict):
                s[k] = remove_key(v, key)
    return s


def save_episodes():
    joblib.dump(episodes_records, f'../results/episodes_records_{params["name"]}.jbl')
    # joblib.dump(state_records, '../results/state_records4.jbl')
    pass


if __name__ == "__main__":

    env = IoTEnv4ML(**params['env_params'])
    oracle = Oracle(thing_list=env.get_thing_list())
    num_episodes = 60000

    instruction_by_thing = oracle.str_instructions
    instructions_set = set(sum([list(v) for v in instruction_by_thing.values()], []))
    thing_weights = {k: len(v) / len(instructions_set) for k, v in instruction_by_thing.items()}
    if env.allow_do_nothing:
        do_nothing_weight = 0.05
        thing_weights = {k: v * (1 - do_nothing_weight) for k, v in thing_weights.items()}
        thing_weights.update({'do_nothing': 0.05})

    pos_episodes = {i: [] for i in instructions_set}
    neg_episodes = {i: [] for i in instructions_set}
    pos_states = {i: [] for i in instructions_set}
    neg_states = {i: [] for i in instructions_set}


    def store_record(obj, pos_storage, neg_storage, achieved_set):
        for i in achieved_set:
            pos_storage[i].append(obj(instruction=i, reward=1))
        for i in instructions_set.difference(achieved_set):
            if random.random() < 0.025:
                neg_storage[i].append(obj(instruction=i, reward=0))


    for i in range(num_episodes):
        logger.info('%' * 5 + f' Episode {i} ' + '%' * 5)
        env.reset()
        run_episode(env=env, weights=thing_weights)
        achieved_goals_str = oracle.get_state_change(env.previous_user_state, env.user_state)
        previous_state_descriptions = oracle.get_state_description(env.previous_user_state, as_string=True)
        state_descriptions = oracle.get_state_description(env.user_state, as_string=True)

        logger.info(achieved_goals_str)
        logger.info(previous_state_descriptions)
        logger.info(state_descriptions)

        previous_state = remove_key(env.previous_user_state)
        state = remove_key(env.user_state)

        from functools import partial

        store_record(obj=partial(EpisodeRecord, initial_state=previous_state, final_state=state),
                     pos_storage=pos_episodes,
                     neg_storage=neg_episodes, achieved_set=achieved_goals_str)

        # store_record(obj=partial(StateRecord, state=state), pos_storage=pos_states,
        #              neg_storage=neg_states, achieved_set=state_descriptions)
        #
        # store_record(obj=partial(StateRecord, state=previous_state), pos_storage=pos_states,
        #              neg_storage=neg_states, achieved_set=previous_state_descriptions)

        # if i > 0 and i % 5000 == 0:
        #     episodes_records = (pos_episodes, neg_episodes)
        #     state_records = (pos_states, neg_states)
        #     save_episodes()

    episodes_records = (pos_episodes, neg_episodes)
    # state_records = (pos_states, neg_states)
    save_episodes()

    dts = EpisodeDataset.from_tuple(data=(pos_episodes, neg_episodes))
    print(dts.get_stats())
