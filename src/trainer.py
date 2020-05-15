import torch

from simulator.Environment import IoTEnv4ML
from architecture.dqnagent import DQNAgent
from architecture.dqn import FlatCritic
from architecture.goal_sampler import GoalSampler
from architecture.replay_buffer import ReplayBuffer


def dqn_update():

model_params = {'instruction_embedding': None, 'state_embedding': None, 'action_embedding': None, 'n_step': None,
                'net_params': None}
exploration_params = {}
env = IoTEnv4ML()
agent = DQNAgent(FlatCritic(**model_params), exploration_params=exploration_params,
                 goal_sampler=GoalSampler(language_model=None), language_model=None)
replay_buffer = ReplayBuffer(max_size=10000)

num_episodes = 50
for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()
    (overall_state, available_actions) = state
    hidden_state = torch.zeros(1, model_params['action_embedding'])
    g = agent.sample_goal()
    running_episode = True

    done = False
    while not done:
        action, hidden_state = agent.select_action(state=overall_state, instruction=g.goal_embedding,
                                                   actions=available_actions, hidden_state=hidden_state)
        next_state, reward, done, info = env.step(action=action)
        replay_buffer.store(g, state, action, next_state, done, reward, hidden_state)

        state = next_state


    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
