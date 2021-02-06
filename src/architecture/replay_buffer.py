import random
from collections import namedtuple
import numpy as np
from typing import List, Tuple
import torch

from sumtree import SumSegmentTree, MinSegmentTree

Transition = namedtuple('Transition',
                        ('goal', 'state', 'action', 'next_state', 'done', 'reward', 'previous_action'))


def get_replay_buffer(per, **kwargs):
    if per:
        return PrioritizedReplayBuffer(**kwargs)
    else:
        return ReplayBuffer(**kwargs)


class ReplayBuffer:
    def __init__(self, max_size, **kwargs):
        self.memory = []
        self.ptr = 0
        self.max_size = max_size

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, item):
        transition = self.memory[item]
        return transition

    def store(self, **kwargs):
        # Build the list of size self.max_size and then replace the transitions by new ones
        if len(self) < self.max_size:
            self.memory.append(Transition(**kwargs))
        else:
            self.memory[self.ptr] = Transition(**kwargs)
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, n_sample):
        indices = random.sample(range(len(self)), n_sample)
        samples = [self.memory[i] for i in indices]
        return indices, samples, torch.ones(n_sample)
        # return random.sample(self.memory, n_sample), torch.ones(n_sample)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, max_size, alpha, beta, prior_eps=1e-6):
        super().__init__(max_size=max_size)
        self.alpha = alpha
        self.beta = beta
        self.prior_eps = prior_eps
        self.max_priority, self.tree_ptr = 1.0, 0

        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(self, **kwargs):
        """Store experience and priority."""
        super().store(**kwargs)

        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample(self, n_sample):
        indices = self._sample_proportional(n_sample)
        samples = [self.memory[i] for i in indices]
        weights = np.array([self._calculate_weight(i) for i in indices])
        return indices, samples, weights

    def _sample_proportional(self, n_sample) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / n_sample

        for i in range(n_sample):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-self.beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-self.beta)
        weight = weight / max_weight

        return weight

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)
        priorities = priorities + self.prior_eps

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def update_beta(self, episode, max_episodes):
        fraction = min(episode / max_episodes, 1.0)
        self.beta = self.beta + fraction * (1.0 - self.beta)
