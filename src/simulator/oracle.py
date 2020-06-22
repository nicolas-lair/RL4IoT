import itertools
from simulator.utils import color_list, percent_level


class Oracle:
    def __init__(self, env):
        self.env = env
        self.instructions = self.build_instruction_set()

    def get_achieved_instruction(self, previous_state, next_state):
        achieved_instruction = []
        for thing in self.env.get_thing_list():
            achieved_instruction.extend(thing.get_state_change(previous_state[thing.name], next_state[thing.name]))
        return achieved_instruction

    def was_achieved(self, previous_state, next_state, instruction):
        return instruction in self.get_achieved_instruction(previous_state, next_state)

    def build_instruction_set(self):
        instructions = dict()
        for thing in self.env.get_thing_list():
            thing_instruction = itertools.chain.from_iterable(thing.instruction.values())
            instructions[thing.name] = {i.format(color=c, level=l, name=thing.name) for i, c, l in
                                        itertools.product(thing_instruction, color_list, percent_level)}
            instructions[thing.name] = sorted(instructions[thing.name])

            # instructions = list(itertools.chain.from_iterable(instructions))
            # instructions = [
            #     {i.format(color=color, level=level) for color, level in itertools.product(color_list, percent_level)}
            #     for i
            #     in instructions]
            # instructions = set().union(*instructions)
        return instructions
