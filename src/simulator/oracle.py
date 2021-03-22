from itertools import product


class Oracle:
    def __init__(self, thing_list, absolute_instruction, relative_instruction):
        self.thing_dict = {t.name: t for t in thing_list}
        self.absolute_instruction = absolute_instruction
        self.relative_instruction = relative_instruction
        self.goals_description_set, self.str_instructions = self.build_instruction_set(ignore_visibility=True)
        self.train_goals_description_set, self.train_str_instructions = self.build_instruction_set(
            ignore_visibility=False)

    def get_state_change(self, previous_state, next_state, ignore_power=False, as_string=True):
        """

        :param previous_state:
        :param next_state:
        :return: list of string one per StateDescription change in the environment
        """
        achieved_instruction = []
        for thing_name in previous_state:
            thing = self.thing_dict[thing_name]
            achieved_instruction.extend(thing.get_thing_change(previous_state[thing.name], next_state[thing.name],
                                                               ignore_power=ignore_power,
                                                               as_string=as_string,
                                                               absolute=self.absolute_instruction,
                                                               relative=self.relative_instruction
                                                               )
                                        )
        return achieved_instruction

    def get_state_description(self, state, ignore_power=False, as_string=False):
        """
        :param state:
        :return: list of StateDescription object
        """
        current_descriptions = []
        for thing_name in state:
            thing = self.thing_dict[thing_name]
            current_descriptions.extend(thing.get_thing_description(state[thing.name], ignore_power))

        if as_string:
            current_descriptions = sum([state_des.sentences for state_des in current_descriptions], [])
        return current_descriptions

    def is_achieved(self, state, instruction):
        return instruction in sum([state_des.sentences for state_des in self.get_state_description(state)], [])

    def was_achieved(self, previous_state, next_state, instruction):
        is_achieved = self.is_achieved(next_state, instruction)
        return is_achieved or (instruction in self.get_state_change(previous_state, next_state))

    def get_thing_goals(self, thing, ignore_visibility=True):
        # TODO change to make robust when a description involves two object
        initialization_states = [thing.init_node(
            is_visible=True, init_type='random', oracle=ignore_visibility) for _ in range(50)]
        states_pair = product(initialization_states, repeat=2)
        state_change = [
            thing.get_thing_change(i, j, ignore_power=True, as_string=False, absolute=self.absolute_instruction,
                                   relative=self.relative_instruction) for i, j in states_pair]
        state_description = [thing.get_thing_description(i, ignore_power=True) for i in initialization_states]
        goals_set = set(sum(state_description, []) + sum(state_change, []))
        return goals_set

    def build_instruction_set(self, mode='first', ignore_visibility=False):
        str_instructions = dict()
        state_description_set = dict()
        for thing in self.thing_dict.values():
            if ignore_visibility or thing.is_visible:
                state_description_set[thing.name] = self.get_thing_goals(thing, ignore_visibility=ignore_visibility)
                str_instructions[thing.name] = sum(
                    [state_des.get_sentences_iterator(mode=mode) for state_des in state_description_set[thing.name]],
                    [])
                str_instructions[thing.name] = sorted(str_instructions[thing.name])
            else:
                state_description_set[thing.name] = []
                str_instructions[thing.name] = []
            thing.reset()
        return state_description_set, str_instructions


if __name__ == '__main__':
    from lighting_things import BigAssFan

    thing = BigAssFan(name='light', simple=True)
    oracle = Oracle([thing])
    instr1 = oracle.str_instructions

    channels = thing.get_channels()
    channels[0].update_visibility(False)
    oracle = Oracle([thing])
    instr2 = oracle.str_instructions

    assert instr1 == instr2
