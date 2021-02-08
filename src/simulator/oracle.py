from itertools import product


class Oracle:
    def __init__(self, thing_list):
        self.thing_list = thing_list
        self.goals_description_set, self.str_instructions = self.build_instruction_set()

    def get_state_change(self, previous_state, next_state, ignore_power=False, as_string=True):
        """

        :param previous_state:
        :param next_state:
        :return: list of string one per StateDescription change in the environment
        """
        achieved_instruction = []
        for thing in self.thing_list:
            achieved_instruction.extend(thing.get_state_change(previous_state[thing.name], next_state[thing.name],
                                                               ignore_power,
                                                               as_string))
        return achieved_instruction

    def get_state_description(self, state, ignore_power=False, as_string=False):
        """
        :param state:
        :return: list of StateDescription object
        """
        current_descriptions = []
        for thing in self.thing_list:
            current_descriptions.extend(thing.get_state_description(state[thing.name], ignore_power))

        if as_string:
            current_descriptions = sum([state_des.sentences for state_des in current_descriptions], [])
        return current_descriptions

    def is_achieved(self, state, instruction):
        return instruction in sum([state_des.sentences for state_des in self.get_state_description(state)], [])

    def was_achieved(self, previous_state, next_state, instruction):
        is_achieved = self.is_achieved(next_state, instruction)
        return is_achieved or (instruction in self.get_state_change(previous_state, next_state))

    def get_thing_goals(self, thing):
        # TODO change to make robust when a description involves two object
        initialization_states = [thing.init(is_visible=True, init_type='random') for _ in range(50)]
        states_pair = product(initialization_states, repeat=2)
        state_change = [thing.get_state_change(i, j, ignore_power=True, as_string=False) for i, j in states_pair]
        state_description = [thing.get_state_description(i) for i in initialization_states]
        goals_set = set(sum(state_description, []) + sum(state_change, []))
        return goals_set

    def build_instruction_set(self, mode='first'):
        str_instructions = dict()
        state_description_set = dict()
        for thing in self.thing_list:
            state_description_set[thing.name] = self.get_thing_goals(thing)
            if mode == 'all':
                str_instructions[thing.name] = sum(
                    [state_des.sentences for state_des in state_description_set[thing.name]],
                    [])
            elif mode == 'first':
                str_instructions[thing.name] = [state_des.sentences[0] for state_des in
                                                state_description_set[thing.name]]
            else:
                raise NotImplementedError(f'mode should be one of all or first, was {mode}')
            # thing_instruction = itertools.chain.from_iterable(thing.instruction.values())
            # instructions[thing.name] = {i.format(color=c, level=l, name=thing.name) for i, c, l in
            #                             itertools.product(thing_instruction, color_list, percent_level)}
            # instructions[thing.name] = sorted(instructions[thing.name])

        return state_description_set, str_instructions
