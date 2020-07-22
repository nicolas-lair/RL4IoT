from collections import namedtuple
from datetime import datetime
import random

import joblib

from src.logger import rootLogger, set_logger_handler
from src.config import generate_params, ThingParam
from simulator.Environment import IoTEnv4ML
from simulator.oracle import Oracle
from simulator.Thing import PlugSwitch, LightBulb

EpisodeRecord = namedtuple('EpisodeRecord', ('initial_state', 'final_state', 'instruction', 'reward'))
StateRecord = namedtuple('StateRecord', ('state', 'instruction', 'reward'))

params = generate_params()
set_logger_handler(rootLogger, **params['logger'], log_path=params['save_directory'])
logger = rootLogger.getChild(__name__)
logger.setLevel(10)


def run_episode(env):
    _, available_actions = env.get_state_and_action()
    done = False
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
    joblib.dump(episodes_records, '../results/episodes_records.jbl')
    joblib.dump(state_records, '../results/state_records.jbl')


if __name__ == "__main__":

    env = IoTEnv4ML(params=params['env_params'])
    oracle = Oracle(env=env)
    num_episodes = 1000
    episodes_records = []
    state_records = []
    instructions_set = set(sum([list(v) for v in oracle.str_instructions.values()], []))

    for i in range(num_episodes):
        logger.info('%' * 5 + f' Episode {i} ' + '%' * 5)
        env.reset()
        run_episode(env=env)
        achieved_goals_str = oracle.get_state_change(env.previous_user_state, env.user_state)
        previous_state_descriptions = oracle.get_state_descriptions(env.previous_user_state, as_string=True)
        state_descriptions = oracle.get_state_descriptions(env.user_state, as_string=True)

        logger.info(achieved_goals_str)
        logger.info(previous_state_descriptions)
        logger.info(state_descriptions)

        for instruction in instructions_set:
            previous_state = remove_key(env.previous_user_state)
            state = remove_key(env.user_state)
            episodes_records.append(
                EpisodeRecord(initial_state=previous_state,
                              final_state=state,
                              instruction=instruction,
                              reward=instruction in achieved_goals_str)
            )
            state_records += [
                StateRecord(state=state,
                            instruction=instruction,
                            reward=instruction in state_descriptions),
                StateRecord(state=previous_state,
                            instruction=instruction,
                            reward=instruction in previous_state_descriptions)
            ]

            if i > 0 and i % 5000 == 0:
                save_episodes()

    save_episodes()
