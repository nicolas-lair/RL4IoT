from Environment import IoTEnv
from oracle import Oracle
from goal_sampler import Goal, GoalSampler

if __name__ == "__main__":
    env = IoTEnv()
    oracle = Oracle(env=env)
    goal_sampler = GoalSampler(language_model=lambda x: x)

    #####################TURN ON LIGHTBULB ###############
    current_state = env.reset()

    print(current_state)
    action = {
        "thing": "lightbulb",
        "channel": "color",
        "action": "turnOn",
        "params": None
    }

    new_state, _, _, _ = env.step(action)
    print(new_state)

    achieved_instruction = oracle.get_achieved_instruction(previous_state=current_state, next_state=new_state)
    print(achieved_instruction)

    goal_sampler.update(target_goals=[], reached_goals_str=achieved_instruction, iter=1)

    ##################TURN OFF LIGHTBULB########################
    print("*" * 20)
    current_state = new_state
    action = {
        "thing": "lightbulb",
        "channel": "color",
        "action": "turnOff",
        "params": None
    }

    target = goal_sampler.sample_goal()
    new_state, _, _, _ = env.step(action)
    print(new_state)

    achieved_instruction = oracle.get_achieved_instruction(previous_state=current_state, next_state=new_state)
    print(achieved_instruction)

    goal_sampler._update_discovered_goals(achieved_instruction, iter=2)
    goal_sampler.update(target_goals=[target], reached_goals_str=achieved_instruction, iter=2)

    print('end')

    # ################ACTIVATE PLUG ##########################
    # print("*" * 20)
    # current_state = new_state
    # action = {
    #     "thing": "plugswitch",
    #     "channel": "switch_binary",
    #     "action": "turnOn",
    #     "params": None
    # }
    #
    # new_state, _, _, _ = env.step(action)
    # print(new_state)
    #
    # achieved_instruction = oracle.get_achieved_instruction(previous_state=current_state, next_state=new_state)
    # print(achieved_instruction)
    #
    # ################DEACTIVATE PLUG ##########################
    # print("*" * 20)
    # current_state = new_state
    # action = {
    #     "thing": "plugswitch",
    #     "channel": "switch_binary",
    #     "action": "turnOff",
    #     "params": None
    # }
    #
    # new_state, _, _, _ = env.step(action)
    # print(new_state)
    #
    # achieved_instruction = oracle.get_achieved_instruction(previous_state=current_state, next_state=new_state)
    # print(achieved_instruction)
    #
    # ##################INCREASE LUMINOSITY########################
    # print("*" * 20)
    # current_state = new_state
    # action = {
    #     "thing": "lightbulb",
    #     "channel": "color",
    #     "action": "increase",
    #     "params": None
    # }
    #
    # new_state, _, _, _ = env.step(action)
    # print(new_state)
    #
    # achieved_instruction = oracle.get_achieved_instruction(previous_state=current_state, next_state=new_state)
    # print(achieved_instruction)
    #
    # ##################INCREASE LUMINOSITY AGAIN########################
    # print("*" * 20)
    # current_state = new_state
    # action = {
    #     "thing": "lightbulb",
    #     "channel": "color",
    #     "action": "increase",
    #     "params": None
    # }
    #
    # new_state, _, _, _ = env.step(action)
    # print(new_state)
    #
    # achieved_instruction = oracle.get_achieved_instruction(previous_state=current_state, next_state=new_state)
    # print(achieved_instruction)
    #
    # ################ DECREASE LUMINOSITY ##########################
    # print("*" * 20)
    # current_state = new_state
    # action = {
    #     "thing": "lightbulb",
    #     "channel": "color",
    #     "action": "decrease",
    #     "params": None
    # }
    #
    # new_state, _, _, _ = env.step(action)
    # print(new_state)
    #
    # achieved_instruction = oracle.get_achieved_instruction(previous_state=current_state, next_state=new_state)
    # print(achieved_instruction)
    # ################ INCREASE COLOR TEMEPRATURE ##########################
    # print("*" * 20)
    # current_state = new_state
    # action = {
    #     "thing": "lightbulb",
    #     "channel": "color_temperature",
    #     "action": "increase",
    #     "params": None
    # }
    #
    # new_state, _, _, _ = env.step(action)
    # print(new_state)
    #
    # achieved_instruction = oracle.get_achieved_instruction(previous_state=current_state, next_state=new_state)
    # print(achieved_instruction)
    #
    # ################ DECREASE LUMINOSITY ##########################
    # print("*" * 20)
    # current_state = new_state
    # action = {
    #     "thing": "lightbulb",
    #     "channel": "color_temperature",
    #     "action": "decrease",
    #     "params": None
    # }
    #
    # new_state, _, _, _ = env.step(action)
    # print(new_state)
    #
    # achieved_instruction = oracle.get_achieved_instruction(previous_state=current_state, next_state=new_state)
    # print(achieved_instruction)