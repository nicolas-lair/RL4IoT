import random
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('goal', 'state',  'action', 'next_state', 'done', 'reward', 'hidden_state'))

class ReplayBuffer:
    def __init__(self, max_size):
        self.memory = deque(maxlen=max_size)

    def __len__(self):
        return len(self.memory)

    def store(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, n_sample):
        return random.sample(self.memory, n_sample)
