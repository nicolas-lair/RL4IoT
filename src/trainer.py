import argparse
import logging

from pprint import pformat

from logger import get_logger, set_logger_handler
from config import generate_trainer_params, save_config, format_config, setup_new_simulation, ThingParam
from simulator.things.TV_thing import TVFullOption, TVWithMediaControl
from simulator.things.Speaker import SpeakerWithMediaControl, SimpleSpeaker
from simulator.things.lighting_things import SimpleLight, BigAssFan, AdorneLight, StructuredHueLight
from simulator.things.Blinds import SimpleBlinds
from records import Records
from simulator.Environment import IoTEnv4ML
from simulator.oracle import Oracle
from architecture.dqnagent import DQNAgent
from architecture.language_model import LanguageModel
from utils import run_episode, test_agent, load_agent_state_dict
#
# parser = argparse.ArgumentParser()
# parser.add_argument('-n', '--name', help='Simulation name', default='diverse_lights')
# parser.add_argument('-d', '--device', help='device on which to run the simulation', default='cuda:0')
# parser.add_argument('-ns', '--n_simulation', help='number of simulation to run', type=int, default=2)
# parser.add_argument('-lm', '--pretrained_language_model', help='use pretrained language model', choices=['0', '1'],
#                     default=0)
# parser.add_argument('-l', '--optim_loss', help='DQN loss', choices=['mse', 'smooth_l1'], default='mse')
# args = parser.parse_args()
#
# simulation_name = args.name
# device = args.device
# N_SIMULATION = args.n_simulation
# use_pretrained_language_model = bool(int(args.pretrained_language_model))
# optim_loss = args.optim_loss

simulation_name = 'living_room_easy'
device = 'cuda:3'
N_SIMULATION = 1
use_pretrained_language_model = False
optim_loss = 'mse'

n_episode = 20000
test_frequence = 100

load_agent = False
load_agent_path = '../results/debug/0/agent'

oracle_params = dict(relative_instruction=True)

thing = [
    ThingParam(TVWithMediaControl, dict(name="television", simple=True, always_on=True)),
    ThingParam(SimpleSpeaker, dict(name="speaker", simple=True, always_on=True)),
    ThingParam(SimpleLight, dict(name="light", simple=True)),
    ThingParam(SimpleBlinds, dict(name="blinds", simple=True)),
    # ThingParam(BigAssFan, dict(name="heater", simple=True, always_on=True)),
    # ThingParam(SimpleLight, dict(name="plug", simple=True)),
    # ThingParam(AdorneLight, dict(name="light", simple=True, always_on=True)),
    # ThingParam(SimpleLight, dict(name="bulb", simple=True)),
    # ThingParam(SimpleLight, dict(name="television", simple=True)),
    # ThingParam(BigAssFan, dict(name="bulb", simple=True, always_on=True)),
]

params = generate_trainer_params(things_list=thing, simulation_name=simulation_name,
                                 use_pretrained_language_model=use_pretrained_language_model,
                                 device=device, n_episode=n_episode, test_frequence=test_frequence,
                                 oracle_params=oracle_params)

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

        env = IoTEnv4ML(**params['env_params'])
        test_env = IoTEnv4ML(**params['env_params'])

        oracle = Oracle(thing_list=env.get_things(), **params['oracle_params'])
        language_model = LanguageModel(**params['language_model_params'])
        agent = DQNAgent(language_model=language_model, params=params)
        metrics_records = Records(save_path=params['save_directory'])

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
            run_episode(agent=agent, env=env, oracle=oracle, save_transitions=True, episode=ep)
            agent.update(episode=ep, max_episodes=num_episodes)

            if ep > 0 and ep % test_frequence == 0:
                test_scores = test_agent(agent=agent, test_env=test_env, oracle=oracle, n_test=n_iter_test)
                metrics_records.update_records(agent=agent, episode=ep, test_scores=test_scores)
                if ep % 10 * test_frequence == 0:
                    metrics_records.save()

        metrics_records.update_records(agent, num_episodes, test_agent(agent=agent, test_env=test_env, oracle=oracle,
                                                                       n_test=n_iter_test))
        metrics_records.save()
        metrics_records.save_as_object()
        logger.info('The End')
