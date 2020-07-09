import itertools
from simulator.utils import color_list, percent_level


class Oracle:
    def __init__(self, env):
        self.env = env
        self.state_description_set, self.str_instructions = self.build_instruction_set()

    def get_state_change(self, previous_state, next_state):
        """

        :param previous_state:
        :param next_state:
        :return: list of string one per StateDescription change in the environment
        """
        achieved_instruction = []
        for thing in self.env.get_thing_list():
            achieved_instruction.extend(thing.get_state_change(previous_state[thing.name], next_state[thing.name]))
        return achieved_instruction

    def get_state_descriptions(self, state):
        """
        :param state:
        :return: list of StateDescription object
        """
        current_descriptions = []
        for thing in self.env.get_thing_list():
            current_descriptions.extend(thing.get_state_description(state[thing.name]))
        return current_descriptions

    def was_achieved(self, previous_state, next_state, instruction):
        return instruction in self.get_state_change(previous_state, next_state)

    def build_instruction_set(self):
        str_instructions = dict()
        state_description_set = dict()
        for thing in self.env.get_thing_list():
            state_description_set[thing.name] = list(thing.instruction.values())
            str_instructions[thing.name] = sum([state_des.sentences for state_des in state_description_set[thing.name]],
                                               [])
            # thing_instruction = itertools.chain.from_iterable(thing.instruction.values())
            # instructions[thing.name] = {i.format(color=c, level=l, name=thing.name) for i, c, l in
            #                             itertools.product(thing_instruction, color_list, percent_level)}
            # instructions[thing.name] = sorted(instructions[thing.name])

        return state_description_set, str_instructions
