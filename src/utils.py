import torch

from Action import RootAction, DoNothing
from goal_sampler import Goal
from logger import format_oracle_state_log, get_logger

logger = get_logger(__name__)
logger.setLevel(20)


def run_episode(agent, env, oracle, episode, save_transitions=True, target_goal=None):
    if env.episode_reset:
        env.reset()
    else:
        raise NotImplementedError

    if target_goal is None:
        target_goal = agent.sample_goal()
        logger.info(f'Targeted goal: {target_goal.goal_string}')

    action_record = []
    hidden_state = torch.zeros(1, agent.action_model.action_embedding_size).to(agent.device)
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
        logger.debug(f'available_actions{[a.name for a in available_actions]}')

        logger.debug(action.name)
        action_record.append(action.name)

        (next_state, next_available_actions), reward, done, info = env.step(action=action)
        logger.debug(f'Agent state \n {format_oracle_state_log(next_state)}')

        if info == 'exec_action':
            logger.debug(f'State: \n {format_oracle_state_log(env.oracle_state)}')

        if save_transitions:
            # if info in ['exec_action'] or done:
            if info in ['exec_action']:
                logger.debug(f'Running action: {"/".join(action_record)}')
                achieved_goals_str = oracle.get_state_change(env.previous_oracle_state, env.oracle_state)
                logger.debug(f'Achieved goals: {achieved_goals_str}')
                agent.goal_sampler.update(reached_goals_str=achieved_goals_str, iter=episode)
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


def test_agent(agent, test_env, oracle, n_test):
    logger.info('Testing agent')
    reward_table = dict()
    for thing, test_instruction in oracle.str_instructions.items():
        reward_table[thing] = dict()
        for instruction in test_instruction:
            logger.info(f'Test : {instruction}')
            current_rewards = 0
            for _ in range(n_test):
                logger.debug('New test episode')
                test_env.reset()
                initial_state = test_env.oracle_state
                # while oracle.is_achieved(state=test_env.oracle_state, instruction=instruction):
                #     test_env.reset()

                test_goal = Goal(goal_string=instruction, language_model=agent.language_model)
                run_episode(agent=agent, env=test_env, target_goal=test_goal, oracle=None, save_transitions=False,
                            episode=None)
                current_rewards += int(
                    oracle.was_achieved(initial_state, test_env.oracle_state, instruction))
            reward_table[thing][instruction] = round(current_rewards / n_test, 2)
    return reward_table
