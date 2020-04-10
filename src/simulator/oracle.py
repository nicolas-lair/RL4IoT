class Oracle:
    def __init__(self, env):
        self.env = env

    def get_achieved_instruction(self, previous_state, next_state):
        achieved_instruction = []
        for thing in self.env.get_thing_list():
            achieved_instruction.extend(thing.get_state_change(previous_state, next_state))
        return achieved_instruction
