from random import choice
import copy

def initialize_instruction(instruction_dict, name, location):
    ins_dict = copy.deepcopy(instruction_dict)
    for v in ins_dict.values():
        v.set_name_and_location(name=name, location=location)
    return ins_dict


class StateDescription:
    def __init__(self, sentences, need_power=True):
        if isinstance(sentences, str):
            sentences = [sentences]

        self.sentences = sentences
        self.need_power = need_power

    def get_instruction(self, mode='random'):
        if mode == 'random':
            return choice(self.sentences)
        elif mode == 'first':
            return self.sentences[0]

    def __eq__(self, other):
        equal = False
        for i in other.sentences:
            if i in self.sentences:
                equal = True
                break
        return equal

    # Needed to be used in set
    def __hash__(self):
        return hash(*self.sentences)

    def set_name_and_location(self, name, location):
        location = 'in ' + location if location else ''
        self.sentences = [' '.join(s.format(name=name, location=location).split()) for s in self.sentences]


if __name__ == '__main__':
    i1 = StateDescription(['Hello', 'Hi'])
    i2 = StateDescription(['Hello', 'Hi'])
    i3 = StateDescription(['Hello'])
    i4 = StateDescription(['Coucou'])

    assert i1 == i2 == i3
    assert i4 != i1
    assert i4 != i2
    assert i4 != i3

    i1 = StateDescription(['{name} is {location}', '{name} is {location}'])
    i1.set_name_and_location(name='Peter', location=None)
    print(i1.sentences)

    i2 = StateDescription(['{name} is {location}', '{name} is {location}'])
    i2.set_name_and_location(name='Peter', location='kitchen')
    print(i2.sentences)
