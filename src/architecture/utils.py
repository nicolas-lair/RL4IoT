import random

import torch


def differentiable_or(proba_tensor):
    """
    Given a list of events probabilities, compute the logical OR between these events and outputs 1
    if at least one of the proba > 0.5, 0 otherwise
    :param proba_tensor: torch tensor float
    :return: boolean
    """
    temp_neg = (proba_tensor - 0.5).relu()
    temp_neg /= proba_tensor - 0.5
    u = 1.1 * temp_neg.sum()
    bool_value = u.clamp(-1, 1)
    return bool_value


if __name__ == "__main__":
    for i in range(10000):
        t = torch.rand(random.randint(1, 10))
        b = differentiable_or(t)
        assert b == (t.max() > 0.5)
