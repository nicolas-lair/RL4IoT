from simulator.Environment import IoTEnv4ML
from architecture.agent import Agent
from architecture.dqn import FlatCritic
from architecture.goal_sampler import GoalSampler

model_params = {'instruction_embedding': None, 'state_embedding': None, 'action_embedding': None, 'n_step': None,
                'net_params': None}
exploration_params = {}
env = IoTEnv4ML()
agent = Agent(FlatCritic(**model_params),
              exploration_params=exploration_params)
goal_sampler = GoalSampler(language_model=None)


num_episodes = 50
for i_episode in range(num_episodes):
    # Initialize the environment and state
    overall_state, available_actions = env.reset()
    g = agent.sample_goal()
    running_episode = True

    while running_episode:

        action = agent.select_action(state=overall_state, instruction=g.goal_embedding, actions=available_actions)
        env.step(action=action)

    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
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
