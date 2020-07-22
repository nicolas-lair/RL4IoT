import random
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('goal', 'state', 'action', 'next_state', 'done', 'reward', 'hidden_state', 'previous_action'))


class ReplayBuffer:
    def __init__(self, max_size):
        self.memory = deque(maxlen=max_size)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, item):
        transition = self.memory[item]
        return transition

    def store(self, **kwargs):
        self.memory.append(Transition(**kwargs))

    def sample(self, n_sample):
        return random.sample(self.memory, n_sample)
