from random import choice
import copy


def initialize_instruction(goal_dict, name, location):
    goal_dict = copy.deepcopy(goal_dict)
    for g in goal_dict.values():
        for v in g.values():
            v.set_name_and_location(name=name, location=location)
    return goal_dict


class GoalDescription:
    def __init__(self, sentences, need_power=True, is_relative=False, mode='first'):
        if isinstance(sentences, str):
            sentences = [sentences]

        self.mode = mode
        self.sentences = sentences
        self.need_power = need_power
        self.is_relative = is_relative  # relative goals are only meaningful if thing is powered on
        self.object_name = None
        self.object_location = None

    def get_instruction(self, mode=None):
        mode = mode if mode is not None else self.mode
        if mode == 'random':
            return choice(self.sentences)
        elif mode == 'first':
            return self.sentences[0]
        else:
            raise NotImplementedError(f'mode should be one of all or first, was {mode}')

    def __eq__(self, other):
        equal = False
        for i in other.sentences:
            if i in self.sentences:
                equal = True
                break
        return equal

    # Needed to be used in set
    def __hash__(self):
        return hash(''.join(self.sentences))

    def set_name_and_location(self, name, location):
        self.object_name = name
        self.object_location = location
        location = 'in ' + location if location else ''
        self.sentences = [' '.join(s.format(name=name, location=location).split()) for s in self.sentences]

    def get_sentences_iterator(self, mode=None):
        mode = mode if mode is not None else self.mode
        if mode == 'first':
            return [self.get_instruction(mode='first')]
        elif mode == 'all':
            return self.sentences
        else:
            raise NotImplementedError(f'mode should be one of all or first, was {mode}')


if __name__ == '__main__':
    i1 = GoalDescription(['Hello', 'Hi'])
    i2 = GoalDescription(['Hello', 'Hi'])
    i3 = GoalDescription(['Hello'])
    i4 = GoalDescription(['Coucou'])

    assert i1 == i2 == i3
    assert i4 != i1
    assert i4 != i2
    assert i4 != i3

    i1 = GoalDescription(['{name} is {location}', '{name} is {location}'])
    i1.set_name_and_location(name='Peter', location=None)
    print(i1.sentences)

    i2 = GoalDescription(['{name} is {location}', '{name} is {location}'])
    i2.set_name_and_location(name='Peter', location='kitchen')
    print(i2.sentences)
