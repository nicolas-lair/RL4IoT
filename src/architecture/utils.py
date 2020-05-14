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
    u = 1.1 * temp_neg.sum(dim=1)
    bool_value = u.clamp(-1, 1)
    return bool_value




def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []

if __name__ == "__main__":
    for i in range(10000):
        t = torch.rand(random.randint(1, 10))
        b = differentiable_or(t)
        assert b == (t.max() > 0.5)
