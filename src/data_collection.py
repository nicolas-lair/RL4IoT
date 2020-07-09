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


def save_episodes():
    joblib.dump(episodes_records, '../results/episodes_records.jbl')


if __name__ == "__main__":

    env = IoTEnv4ML(params=params['env_params'])
    oracle = Oracle(env=env)
    num_episodes = 10000
    episodes_records = []
    instructions_set = set(sum([list(v) for v in oracle.instructions.values()], []))

    for i in range(num_episodes):
        logger.info('%' * 5 + f' Episode {i} ' + '%' * 5)
        env.reset()
        run_episode(env=env)
        achieved_goals_str = oracle.get_state_change(env.previous_user_state, env.user_state)
        if not achieved_goals_str:
            print('stop')
        logger.info(achieved_goals_str)

        for instruction in instructions_set:
            episodes_records.append(
                EpisodeRecord(initial_state=env.previous_user_state,
                              final_state=env.user_state,
                              instruction=instruction,
                              reward=instruction in achieved_goals_str)
            )

            if i > 0 and i % 500 == 0:
                save_episodes()
