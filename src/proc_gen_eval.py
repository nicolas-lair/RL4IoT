from pprint import pformat
import logging

from records import Records
from utils import run_episode, test_agent, load_agent_state_dict
from logger import get_logger, set_logger_handler
from config import generate_proc_gen_eval_params, save_config, format_config, setup_new_simulation, ThingParam
from simulator.lighting_things import BigAssFan, SimpleLight, AdorneLight
from simulator.Environment import IoTEnv4ML
from simulator.oracle import Oracle
from architecture.dqnagent import DQNAgent
from architecture.language_model import LanguageModel

simulation_name = 'three_lights_alwayson_onehot'
device = 'cuda:1'
N_SIMULATION = 2
use_pretrained_language_model = False
optim_loss = 'mse'

n_episode = 20000
test_frequence = 300

load_agent = False
load_agent_path = '../results/debug/0/agent'


thing = [
    ThingParam(BigAssFan, dict(name="bulb", simple=True, always_on=True)),
    # ThingParam(BigAssFan, dict(name="bulb", simple=True, always_on=True)),
    # ThingParam(BigAssFan, dict(name="heater", simple=True, always_on=True)),
    ThingParam(SimpleLight, dict(name="plug", simple=True, always_on=True)),
    ThingParam(AdorneLight, dict(name="light", simple=True)),
    # ThingParam(SimpleLight, dict(name="bulb", simple=True)),
    # ThingParam(SimpleLight, dict(name="television", simple=True)),
    # ThingParam(BigAssFan, dict(name="bulb", simple=True, always_on=True)),
]
params = generate_proc_gen_eval_params(simulation_name=simulation_name, device=device, things_list=thing,
                                       use_pretrained_language_model=use_pretrained_language_model,
                                       n_episode=n_episode, test_frequence=test_frequence,
                                       oracle_params=dict(relative_instruction=False),
                                       )

set_logger_handler(**params['logger'], log_path=params['save_directory'])
logger = get_logger(__name__)

logger.info('begin')
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    for i in range(N_SIMULATION):
        if i != 0:
            setup_new_simulation(params)

        save_config(params)
        logger.info(f'Simulation params:\n {pformat(format_config(params))}')

        introduce_new_object = False
        rolling_window, score_threshold = params['new_objects_threshold']

        env = IoTEnv4ML(**params['env_params'])
        env.update_visibility(thing='bulb', channel='brightness', visibility=False)
        env.update_visibility(thing='heater', channel='color_temperature', visibility=False)

        test_env = IoTEnv4ML(**params['env_params'])
        test_env.set_all_things_visible()

        oracle = Oracle(thing_list=env.get_things(), **params['oracle_params'])
        train_goals = oracle.train_str_instructions
        language_model = LanguageModel(**params['language_model_params'])
        agent = DQNAgent(language_model=language_model, params=params)
        metrics_records = Records(save_path=params['save_directory'], rolling_window=rolling_window)

        num_episodes = params['n_episode']
        n_iter_test = params['n_iter_test']
        test_frequence = params['test_frequence']

        if load_agent:
            load_agent_state_dict(agent=agent, path=load_agent_path, oracle=oracle,
                                  test_env=test_env, params=n_iter_test, metrics_records=metrics_records)

        for ep in range(num_episodes):
            if env.episode_reset:
                env.reset()
            else:
                raise NotImplementedError

            logger.info('%' * 5 + f' Episode {ep} ' + '%' * 5)
            if introduce_new_object:
                logger.info(f'New things/channels were introduced at episode '
                            f'{metrics_records.new_objet_introduction_episode}')
            run_episode(agent=agent, env=env, oracle=oracle, save_transitions=True, episode=ep)
            agent.update(episode=ep, max_episodes=num_episodes)

            if ep > 0 and ep % test_frequence == 0:
                test_scores = test_agent(agent=agent, test_env=test_env, oracle=oracle, n_test=n_iter_test)
                metrics_records.update_records(agent=agent, episode=ep, test_scores=test_scores,
                                               train_goals=train_goals)
                if ep % 10 * test_frequence == 0:
                    metrics_records.save()

                # TODO check from here
                if not introduce_new_object and metrics_records.train_rolling_average.get_mean() > score_threshold:
                    logger.info('Introducing new objects')
                    metrics_records.new_objet_introduction_episode = ep
                    introduce_new_object = True
                    env.set_all_things_visible()

        metrics_records.update_records(agent=agent, episode=num_episodes,
                                       test_scores=test_agent(agent=agent, test_env=test_env, oracle=oracle,
                                                              n_test=n_iter_test),
                                       train_goals=train_goals)
        metrics_records.save()
        metrics_records.save_as_object()
        logger.info('The End')
