import random

import torch

from simulator.things.Thing import Thing
from Action import RootAction, DoNothing
from goal_sampler import Goal
from logger import format_oracle_state_log, get_logger

logger = get_logger(__name__)
logger.setLevel(20)


def load_agent_state_dict(agent, path, oracle, test_env, params, metrics_records):
    logger.info('Loading agent')
    agent.load(folder=path)
    test_scores = test_agent(agent=agent, test_env=test_env, oracle=oracle, n_test=params['n_iter_test'])
    metrics_records.update_records(agent=agent, episode=0, test_scores=test_scores)


def automatic_action_selection(agent, env, target_goal, exploration=False):
    state, available_actions = env.get_state_and_action()
    sample = random.random() if exploration else 1
    if len(target_goal.goal_string) != 0 and sample >= agent.exploration_threshold:
        thing_action = [a for a in available_actions if isinstance(a, Thing)]
        action = None
        for t in thing_action:
            if t.name in target_goal.goal_string:
                action = t
                break
        assert action is not None
    else:
        action = random.choice(available_actions)

    action_embedding, a_type = agent._embed_one_action(action)
    action_embedding = agent.action_model(action_embedding, [[a_type]], instruction=[target_goal.goal_embedding])
    action_embedding = action_embedding.squeeze(0)
    return action, action_embedding


def run_episode(agent, env, oracle=None, episode=None, save_transitions=True, target_goal=None, reset=True,
                automatic_thing_selection=False, test_mode=False):
    if test_mode:
        save_transitions = False
        exploration = False
    else:
        exploration = True
        assert episode is not None
        assert oracle is not None

    if reset:
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
        if automatic_thing_selection and (env.action_step == 'thing'):
            action, hidden_state = automatic_action_selection(agent, env, target_goal, exploration=exploration)
        else:
            action, hidden_state = agent.select_action(state=state, instruction=target_goal.goal_embedding,
                                                       actions=available_actions, hidden_state=hidden_state,
                                                       exploration=exploration)
        logger.debug(f'available_actions{[a.name for a in available_actions]}')

        logger.debug(action.name)
        action_record.append(action.name)

        (next_state, next_available_actions), reward, done, info = env.step(action=action)
        logger.debug(f'Agent state \n {format_oracle_state_log(env.state)}')

        if info == 'exec_action':
            logger.debug(f'State: \n {format_oracle_state_log(env.oracle_state)}')

        if save_transitions:
            if automatic_thing_selection and isinstance(action, Thing):
                pass
            else:
                save_transitions_replay_buffer(agent=agent, env=env, oracle=oracle, episode=episode,
                                               target_goal=target_goal, state=state, action=action,
                                               next_state=next_state, next_actions=next_available_actions,
                                               previous_action=previous_action, reward=reward, done=done, info=info)

    logger.info(f'Action summary: {"/".join(action_record)}')


def save_transitions_replay_buffer(agent, env, oracle, episode, target_goal, previous_action, state, action, next_state,
                                   next_actions, reward, done, info):
    # if info in ['exec_action'] or done:
    if info in ['exec_action']:
        # logger.debug(f'Running action: {"/".join(action_record)}')
        achieved_goals_str = oracle.get_state_change(env.previous_oracle_state, env.oracle_state)
        logger.debug(f'Achieved goals: {achieved_goals_str}')
        agent.goal_sampler.update(reached_goals_str=achieved_goals_str, iter=episode)
        agent.store_transitions_with_oracle_feedback(achieved_goals_str=achieved_goals_str,
                                                     done=done,
                                                     state=state,
                                                     action=action,
                                                     next_state=(next_state, next_actions),
                                                     previous_action=previous_action)

        if env.allow_do_nothing and len(target_goal.goal_string) > 0:
            agent.store_transitions(goal=target_goal, state=next_state, action=DoNothing(),
                                    next_state=([], []),
                                    done=True,
                                    reward=int(target_goal.goal_string in achieved_goals_str),
                                    previous_action=action)
        # action_record.append('/')
    elif info == 'do_nothing':
        pass  # TODO use internal reward function

    elif len(target_goal.goal_string) > 0:
        assert info != 'do_nothing'
        agent.store_transitions(goal=target_goal, state=state, action=action,
                                next_state=(next_state, next_actions), done=done, reward=reward,
                                previous_action=previous_action)


def test_agent(agent, test_env, oracle, n_test, automatic_thing_selection=False):
    logger.info('Testing agent')
    agent.eval()
    reward_table = dict()
    for thing, thing_goals in oracle.goals_description_set.items():
        table = dict()
        for goals in thing_goals:
            power_before_episode = goals.is_relative
            for instruction in goals.get_sentences_iterator():
                logger.info(f'Test : {instruction}')
                test_goal = Goal(goal_string=instruction, language_model=agent.language_model)
                current_rewards = 0
                for _ in range(n_test):
                    logger.debug('New test episode')
                    test_env.reset()
                    if power_before_episode:
                        test_env.get_things(goals.object_name).power_on()

                    initial_state = test_env.oracle_state

                    run_episode(agent=agent, env=test_env, target_goal=test_goal, test_mode=True,
                                reset=False, automatic_thing_selection=automatic_thing_selection)
                    current_rewards += int(
                        oracle.was_achieved(initial_state, test_env.oracle_state, instruction))
                table[instruction] = round(current_rewards / n_test, 2)
        reward_table[thing] = dict(sorted(table.items()))
    agent.train()
    return reward_table


def extend_dict(d, k, v):
    if isinstance(v, dict):
        for k_, v_ in v.items():
            extend_dict(d[k], k_, v_)
    else:
        assert k in d.keys()
        d.update({k: v})
