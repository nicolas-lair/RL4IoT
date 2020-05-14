import random
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('goal', 'state', 'action', 'next_state', 'reward', 'previous_action'))

class ReplayBuffer:
    def __init__(self, max_size):
        self.memory = deque(maxlen=max_size)

    def __len__(self):
        return len(self.memory)

    def store(self, transitions):
        if isinstance(transitions, list):
            [self.store(t) for t in transitions]
        elif isinstance(transitions, Transition):
            self.memory.append(transitions)

    def sample(self, n_sample):
        return random.sample(self.memory, n_sample)
