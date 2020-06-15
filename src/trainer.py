import torch
import tqdm

from config import params
from simulator.Environment import IoTEnv4ML
from simulator.oracle import Oracle
from architecture.dqnagent import DQNAgent
from architecture.dqn import FlatQnet
from architecture.language_model import LanguageModel
from architecture.utils import flatten

if __name__ == "__main__":
    env = IoTEnv4ML(params=params['env_params'])
    oracle = Oracle(env=env)
    language_model = LanguageModel(**params['language_model_params'])
    agent = DQNAgent(FlatQnet, language_model=language_model, params=params)

    num_episodes = 50
    deep_action_space_embedding_size = params['model_params']['action_embedding_size']
    for i in tqdm.trange(num_episodes):
        # Initialize the environment and state + flatten
        (state, available_actions) = env.reset()
        # TODO change to use Modualr archi
        flatten_state = flatten(state)
        state = torch.stack(list(flatten_state.values())).mean(0, keepdim=True).float()

        hidden_state = torch.zeros(1, deep_action_space_embedding_size)
        target_goal = agent.sample_goal()
        running_episode = True

        done = False
        while not done:
            action, hidden_state = agent.select_action(state=state, instruction=target_goal.goal_embedding,
                                                       actions=available_actions, hidden_state=hidden_state)
            (next_state, next_available_actions), reward, done, info = env.step(action=action)
            # TODO change to use Modualr archi
            next_state = torch.stack(list(flatten(next_state).values())).mean(0, keepdim=True).float()
            if not done:
                assert isinstance(next_available_actions, list)

                # Do not store transition with random goals
                if len(target_goal.goal_string) > 0:
                    agent.store_transitions(goal=target_goal, state=state, action=action,
                                            next_state=(next_state, next_available_actions), done=done, reward=reward,
                                            hidden_state=hidden_state)
                state = next_state
                available_actions = next_available_actions

            agent.udpate_policy_net()

        achieved_goals_str = oracle.get_achieved_instruction(env.previous_user_state, env.user_state)

        agent.goal_sampler.update([target_goal], achieved_goals_str, iter=i)

        failed_goals = agent.goal_sampler.get_failed_goals(achieved_goals_str=achieved_goals_str)
        achieved_goals = [agent.goal_sampler.discovered_goals[g] for g in achieved_goals_str]

        agent.store_final_transitions(achieved_goals=achieved_goals,
                                      failed_goals=failed_goals,
                                      state=state,
                                      action=action,
                                      next_state=(next_state, []),
                                      hidden_state=hidden_state
                                      )

        if i % params['target_update_frequence'] == 0:
            agent.update_target_net()

    print('Complete')
