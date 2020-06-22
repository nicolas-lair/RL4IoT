from datetime import datetime

import yaml
import torch
import tqdm
import joblib

from config import params
from simulator.Environment import IoTEnv4ML
from simulator.oracle import Oracle
from simulator.Action import RootAction
from architecture.dqnagent import DQNAgent
from architecture.language_model import LanguageModel
from architecture.goal_sampler import Goal


def run_episode(agent, env, target_goal, train=True):
    hidden_state = torch.zeros(1, deep_action_space_embedding_size).to(agent.device)
    state, available_actions = env.get_state_and_action()

    action = RootAction(children=available_actions, embedding=hidden_state.clone())
    done = False
    while not done:
        previous_action = action
        previous_hidden_state = hidden_state
        state, available_actions = env.get_state_and_action()
        # TODO change to use Modular archi

        action, hidden_state = agent.select_action(state=state, instruction=target_goal.goal_embedding,
                                                   actions=available_actions, hidden_state=hidden_state)
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
        if train:
            agent.udpate_policy_net()
    return state, action, next_state, previous_hidden_state, previous_action


def test_agent(agent, test_env, oracle, verbose=True):
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
    if verbose:
        print("\n" + "%" * 5 + f"Test after {i} episodes" + "%" * 5)
        print(yaml.dump(reward_table))
    return reward_table


if __name__ == "__main__":

    test_record = {}
    env = IoTEnv4ML(params=params['env_params'])
    test_env = IoTEnv4ML(params=params['env_params'])

    oracle = Oracle(env=env)
    language_model = LanguageModel(**params['language_model_params'])
    agent = DQNAgent(params['dqn_architecture'], language_model=language_model, params=params)

    num_episodes = params['n_episode']
    deep_action_space_embedding_size = params['model_params']['action_embedding_size']

    (state, available_actions) = env.reset()
    for i in tqdm.trange(num_episodes, disable=(not params['verbose'])):
        if params['episode_reset']:
            # Initialize the environment and state + flatten
            env.reset()
        else:
            raise NotImplementedError
        # TODO go to root action but no global reset
        target_goal = agent.sample_goal()
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

        if i % params['target_update_frequence'] == 0:
            agent.update_target_net()

        if i > 0 and i % params['test_frequence'] == 0:
            test_record[i] = test_agent(agent=agent, test_env=test_env, oracle=oracle, verbose=params['verbose'])

    test_record[num_episodes] = test_agent(agent=agent, test_env=test_env, oracle=oracle, verbose=params['verbose'])
    joblib.dump(test_record, f'../results/test_record_{str(datetime.now(tz=None))}')
    # joblib.dump(test_record, f'test_record_{str(datetime.now(tz=None))}')
    joblib.dump(params, f'../results/simulation_param_{str(datetime.now(tz=None))}')
    print('Complete')
