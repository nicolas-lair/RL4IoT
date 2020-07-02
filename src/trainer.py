import os
from datetime import datetime

from pprint import pformat
import yaml
import torch
import joblib

from src.logger import rootLogger, format_user_state_log, set_logger_handler
from src.config import generate_params, save_config, format_config, setup_new_simulation
from simulator.Environment import IoTEnv4ML
from simulator.oracle import Oracle
from simulator.Action import RootAction
from architecture.dqnagent import DQNAgent
from architecture.language_model import LanguageModel
from architecture.goal_sampler import Goal

N_SIMULATION = 3
params = generate_params()
set_logger_handler(rootLogger, **params['logger'], log_path=params['save_directory'])
logger = rootLogger.getChild(__name__)


def run_episode(agent, env, target_goal, train=True):
    hidden_state = torch.zeros(1, deep_action_space_embedding_size).to(agent.device)
    state, available_actions = env.get_state_and_action()

    action = RootAction(children=available_actions, embedding=hidden_state.clone())
    done = False
    while not done:
        previous_action = action
        previous_hidden_state = hidden_state
        state, available_actions = env.get_state_and_action()

        action, hidden_state = agent.select_action(state=state, instruction=target_goal.goal_embedding,
                                                   actions=available_actions, hidden_state=hidden_state)
        logger.debug(f'State: \n {format_user_state_log(env.user_state)}')
        logger.debug(available_actions)
        logger.debug(action)

        (next_state, next_available_actions), reward, done, info = env.step(action=action)
        # TODO change to use Modular archi
        # next_state = torch.stack(list(flatten(next_state).values())).mean(0, keepdim=True).float()
        if not done:
            if train:
                # Do not store transition with random goals
                if len(target_goal.goal_string) > 0:
                    agent.store_transitions(goal=target_goal, state=state, action=action,
                                            next_state=(next_state, next_available_actions), done=done, reward=reward,
                                            hidden_state=previous_hidden_state, previous_action=previous_action)
    return state, action, next_state, previous_hidden_state, previous_action


def test_agent(agent, test_env, oracle):
    logger.info('Testing agent')
    reward_table = dict()
    for thing, test_instruction in oracle.instructions.items():
        reward_table[thing] = dict()
        for instruction in test_instruction:
            current_rewards = 0
            for _ in range(params['n_iter_test']):
                test_env.reset()
                test_goal = Goal(goal_string=instruction, language_model=agent.language_model)
                run_episode(agent=agent, env=test_env, target_goal=test_goal, train=False)
                current_rewards += int(
                    oracle.was_achieved(test_env.previous_user_state, test_env.user_state, instruction))
            reward_table[thing][instruction] = current_rewards / params['n_iter_test']
        reward_table[thing]['overall'] = sum(reward_table[thing].values()) / len(reward_table[thing])
    logger.info("%" * 5 + f"Test after {i} episodes" + "%" * 5 + "\n" + yaml.dump(reward_table))
    return reward_table


def save_records():
    # Test record
    joblib.dump(test_record, os.path.join(params['save_directory'], 'test_record.jbl'))

    # Goal sampler
    agent.goal_sampler.save_records(os.path.join(params['save_directory'], 'goal_sampler_record.jbl'))


if __name__ == "__main__":
    for i in range(N_SIMULATION):
        if i != 0:
            setup_new_simulation(rootLogger, params)
            
        save_config(params)
        logger.info(f'Simulation params:\n {pformat(format_config(params))}')

        test_record = {}
        env = IoTEnv4ML(params=params['env_params'])
        test_env = IoTEnv4ML(params=params['env_params'])

        oracle = Oracle(env=env)
        language_model = LanguageModel(**params['language_model_params'])
        agent = DQNAgent(params['dqn_architecture'], language_model=language_model, params=params)

        num_episodes = params['n_episode']
        deep_action_space_embedding_size = params['model_params']['action_embedding_size']

        # (state, available_actions) = env.reset()
        for i in range(num_episodes):
            logger.info('%' * 5 + f' Episode {i} ' + '%' * 5)
            if params['episode_reset']:
                # Initialize the environment and state + flatten
                env.reset()
            else:
                raise NotImplementedError
            # TODO go to root action but no global reset
            target_goal = agent.sample_goal()
            logger.info(f'Targeted goal: {target_goal.goal_string}')

            final_state, final_action, final_next_state, ante_final_hidden_state, ante_final_action = run_episode(
                agent=agent, env=env,
                target_goal=target_goal,
                train=True)

            # TODO refactoring to let the goal sampler ask to the oracle
            achieved_goals_str = oracle.get_achieved_instruction(env.previous_user_state, env.user_state)

            agent.goal_sampler.update([target_goal], achieved_goals_str, iter=i)

            failed_goals = agent.goal_sampler.get_failed_goals(achieved_goals_str=achieved_goals_str)
            achieved_goals = [agent.goal_sampler.discovered_goals[g] for g in achieved_goals_str]

            agent.store_final_transitions(achieved_goals=achieved_goals,
                                          failed_goals=failed_goals,
                                          state=final_state,
                                          action=final_action,
                                          next_state=(final_next_state, []),
                                          hidden_state=ante_final_hidden_state,
                                          previous_action=ante_final_action
                                          )

            logger.debug('Update of policy net')
            agent.udpate_policy_net()
            logger.debug('done')

            agent.update_exploration_function(iter=i + 1)

            if i % params['target_update_frequence'] == 0:
                logger.debug('Update of target net')
                agent.update_target_net()
                logger.debug('done')

            if i > 0 and i % params['test_frequence'] == 0:
                test_record[i] = test_agent(agent=agent, test_env=test_env, oracle=oracle)

            if i > 0 and i % 10 * params['test_frequence'] == 0:
                save_records()

        test_record[num_episodes] = test_agent(agent=agent, test_env=test_env, oracle=oracle)
        save_records()

        end_time = str(datetime.now(tz=None)).split('.')[0]
        print(f'Completed at {end_time}')
