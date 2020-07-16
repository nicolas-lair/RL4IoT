from random import choice


class StateDescription:
    def __init__(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]

        self.sentences = sentences

    def get_random_instruction(self):
        return choice(self.sentences)

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


if __name__ == '__main__':
    i1 = StateDescription(['Hello', 'Hi'])
    i2 = StateDescription(['Hello', 'Hi'])
    i3 = StateDescription(['Hello'])
    i4 = StateDescription(['Coucou'])

    assert i1 == i2 == i3
    assert i4 != i1
    assert i4 != i2
    assert i4 != i3

