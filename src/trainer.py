import argparse

from pprint import pformat
import torch
import logging

from logger import get_logger, format_oracle_state_log, set_logger_handler
from config import generate_trainer_params, save_config, format_config, setup_new_simulation
from records import Records
from simulator.Environment import IoTEnv4ML
from simulator.oracle import Oracle
from simulator.Action import RootAction, DoNothing
from architecture.dqnagent import DQNAgent
from architecture.language_model import LanguageModel
from architecture.goal_sampler import Goal

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', help='Simulation name', default='new_test')
parser.add_argument('-d', '--device', help='device on which to run the simulation', default='cuda:1')
parser.add_argument('-ns', '--n_simulation', help='number of simulation to run', type=int, default=10)
parser.add_argument('-lm', '--pretrained_language_model', help='use pretrained language model', choices=['0', '1'],
                    default=0)
parser.add_argument('-l', '--optim_loss', help='DQN loss', choices=['mse', 'smooth_l1'], default='mse')
args = parser.parse_args()

simulation_name = args.name
device = args.device
N_SIMULATION = args.n_simulation
use_pretrained_language_model = bool(int(args.pretrained_language_model))
optim_loss = args.optim_loss

params = generate_trainer_params(simulation_name=simulation_name,
                                 use_pretrained_language_model=use_pretrained_language_model,
                                 device=device)

set_logger_handler(**params['logger'], log_path=params['save_directory'])
logger = get_logger(__name__)

logger.info('begin')


# logger.setLevel(logging.DEBUG)


def run_episode(agent, env, target_goal, oracle, save_transitions=True):
    action_record = []
    hidden_state = torch.zeros(1, deep_action_space_embedding_size).to(agent.device)
    state, available_actions = env.get_state_and_action()

    action = RootAction(children=available_actions, embedding=hidden_state.clone())
    done = False
    logger.debug(f'State: \n {format_oracle_state_log(env.oracle_state)}')

    while not done:
        previous_action = action
        state, available_actions = env.get_state_and_action()

        action, hidden_state = agent.select_action(state=state, instruction=target_goal.goal_embedding,
                                                   actions=available_actions, hidden_state=hidden_state,
                                                   exploration=save_transitions)
        # logger.debug(f'available_actions{[a.name for a in available_actions]}')

        logger.debug(action.name)
        action_record.append(action.name)

        (next_state, next_available_actions), reward, done, info = env.step(action=action)
        logger.debug(f'Agent state \n {format_oracle_state_log(next_state)}')

        if info == 'exec_action':
            logger.debug(f'State: \n {format_oracle_state_log(env.oracle_state)}')

        if save_transitions:
            if info in ['exec_action']:
                logger.debug(f'Running action: {"/".join(action_record)}')
                achieved_goals_str = oracle.get_state_change(env.previous_oracle_state, env.oracle_state)
                logger.debug(f'Achieved goals: {achieved_goals_str}')
                agent.goal_sampler.update(reached_goals_str=achieved_goals_str, iter=j)
                agent.store_transitions_with_oracle_feedback(achieved_goals_str=achieved_goals_str,
                                                             done=done,
                                                             state=state,
                                                             action=action,
                                                             next_state=(next_state, next_available_actions),
                                                             previous_action=previous_action)

                agent.store_transitions(goal=target_goal, state=next_state, action=DoNothing(),
                                        next_state=([], []),
                                        done=True,
                                        reward=int(target_goal.goal_string in achieved_goals_str),
                                        previous_action=action)
                action_record.append('/')
            elif info == 'do_nothing':
                pass  # TODO use internal reward function

            elif len(target_goal.goal_string) > 0:
                assert info != 'do_nothing'
                agent.store_transitions(goal=target_goal, state=state, action=action,
                                        next_state=(next_state, next_available_actions), done=done, reward=reward,
                                        previous_action=previous_action)

    logger.info(f'Action summary: {"/".join(action_record)}')


def test_agent(agent, test_env, oracle):
    logger.info('Testing agent')
    reward_table = dict()
    for thing, test_instruction in oracle.str_instructions.items():
        reward_table[thing] = dict()
        for instruction in test_instruction:
            logger.info(f'Test : {instruction}')
            current_rewards = 0
            for _ in range(params['n_iter_test']):
                logger.debug('New test episode')
                test_env.reset()
                # while oracle.is_achieved(state=test_env.oracle_state, instruction=instruction):
                #     test_env.reset()

                test_goal = Goal(goal_string=instruction, language_model=agent.language_model)
                run_episode(agent=agent, env=test_env, target_goal=test_goal, oracle=None, save_transitions=False)
                current_rewards += int(
                    oracle.was_achieved(test_env.previous_oracle_state, test_env.oracle_state, instruction))
            reward_table[thing][instruction] = round(current_rewards / params['n_iter_test'], 2)
    return reward_table


if __name__ == "__main__":
    for i in range(N_SIMULATION):
        if i != 0:
            setup_new_simulation(params)

        save_config(params)
        logger.info(f'Simulation params:\n {pformat(format_config(params))}')

        # test_record = {}
        env = IoTEnv4ML(**params['env_params'])
        test_env = IoTEnv4ML(**params['env_params'])

        oracle = Oracle(thing_list=env.get_thing_list())
        language_model = LanguageModel(**params['language_model_params'])
        agent = DQNAgent(language_model=language_model, params=params)
        metrics_records = Records(save_path=params['save_directory'])

        num_episodes = params['n_episode']
        deep_action_space_embedding_size = params['model_params']['action_embedding_size']

        # (state, available_actions) = env.reset()
        for j in range(num_episodes):
            logger.info('%' * 5 + f' Episode {j} ' + '%' * 5)
            if params['episode_reset']:
                # Initialize the environment and state + flatten
                env.reset()
            else:
                raise NotImplementedError
            target_goal = agent.sample_goal()
            logger.info(f'Targeted goal: {target_goal.goal_string}')
            # while oracle.is_achieved(state=env.oracle_state, instruction=target_goal.goal_string):
            #     env.reset()

            run_episode(agent=agent, env=env, target_goal=target_goal, oracle=oracle, save_transitions=True)

            agent.update(episode=j, max_episodes=num_episodes)

            if j > 0 and j % params['test_frequence'] == 0:
                test_scores = test_agent(agent=agent, test_env=test_env, oracle=oracle)
                metrics_records.update_records(agent=agent, episode=j, test_scores=test_scores)
                if j % 10 * params['test_frequence'] == 0:
                    metrics_records.save()

        metrics_records.update_records(agent, num_episodes, test_agent(agent=agent, test_env=test_env, oracle=oracle))
        metrics_records.save()
        metrics_records.save_as_object()
        logger.info('The End')
