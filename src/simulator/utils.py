import random

from itertools import cycle
import torch

color_list = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink']
color_h_inf = [0, 18, 50, 64, 167, 252, 300, 335]
color_h_sup = [17, 49, 63, 166, 251, 299, 334, 360]


def get_color_name_from_hsb(h, s, b):
    assert isinstance(h, int) and (0 <= h <= 360), 'hue should be a int between 0 and 360'
    assert isinstance(s, int) and (0 <= s <= 100), 'saturation should be a int between 0 and 100'
    assert isinstance(b, int) and (0 <= b <= 100), 'brightness should be a int between 0 and 100'

    colors = cycle(color_list)

    for h_inf, h_sup in zip(color_h_inf, color_h_sup):
        color = next(colors)
        if h_inf <= h <= h_sup:
            return color
    raise EOFError


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
    test_dict = {0: "red",
                 5: "red",
                 17: "red",
                 18: "orange",
                 35: "orange",
                 50: "yellow",
                 60: "yellow",
                 64: "green",
                 80: "green",
                 166: "green",
                 167: "blue",
                 200: "blue",
                 252: "purple",
                 280: "purple",
                 300: "pink",
                 320: "pink",
                 335: "red",
                 360: "red"}
    for h, true_color in test_dict.items():
        c = get_color_name_from_hsb(h, 50, 50)
        print(h, c, c == true_color)

    for i in range(10000):
        t = torch.rand(random.randint(1, 10))
        b = differentiable_or(t)
        assert b == (t.max() > 0.5)