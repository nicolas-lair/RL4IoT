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


def flatten(d):
    out = {}
    for key, val in d.items():
        if isinstance(val, dict):
            val = [val]
        if isinstance(val, list):
            for subdict in val:
                deeper = flatten(subdict).items()
                out.update({key + '_' + key2: val2 for key2, val2 in deeper})
        else:
            out[key] = val
    return out


def dict_to_device(d, device):
    out = {}
    for key, val in d.items():
        if isinstance(val, dict):
            out[key] = dict_to_device(val, device)
        elif isinstance(val, torch.Tensor):
            out[key] = val.to(device)
        else:
            print(type(val))
            raise NotImplementedError
    return out

if __name__ == "__main__":
    for i in range(10000):
        t = torch.rand(random.randint(1, 10))
        b = differentiable_or(t)
        assert b == (t.max() > 0.5)
