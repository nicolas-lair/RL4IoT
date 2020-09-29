from itertools import cycle

color_list = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink']
color_h_inf = [0, 18, 50, 64, 167, 252, 300, 335]
color_h_sup = [17, 49, 63, 166, 251, 299, 334, 360]

percent_level = ['very low', 'low', 'average', 'high', 'very high']
percent_bound_level = [0, 20, 40, 60, 80, 100]


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


def color_to_hsb(string_color):
    idx = color_list.index(string_color)
    h = (color_h_inf[idx] + color_h_sup[idx]) // 2
    return h, 75, 70


def level_to_percent(lvl):
    idx = percent_level.index(lvl)
    p = (percent_bound_level[idx] + percent_bound_level[idx + 1]) // 2
    return p


def percent_to_level(p):
    i = 0
    while i < len(percent_bound_level):
        if p <= percent_bound_level[i + 1]:
            break
        else:
            i += 1
    return percent_level[i]


if __name__ == "__main__":
    # test_dict = {0: "red",
    #              5: "red",
    #              17: "red",
    #              18: "orange",
    #              35: "orange",
    #              50: "yellow",
    #              60: "yellow",
    #              64: "green",
    #              80: "green",
    #              166: "green",
    #              167: "blue",
    #              200: "blue",
    #              252: "purple",
    #              280: "purple",
    #              300: "pink",
    #              320: "pink",
    #              335: "red",
    #              360: "red"}
    # for h, true_color in test_dict.items():
    #     c = get_color_name_from_hsb(h, 50, 50)
    #     print(h, c, c == true_color)

    test_dict = {
        0: "very low",
        10: "very low",
        20: "very low",
        30: "low",
        40: "low",
        50: "average",
        60: "average",
        70: "high",
        80: "high",
        90: "very high",
        100: "very high",
    }

    for p, true_lvl in test_dict.items():
        lvl = percent_to_level(p)
        print(p, lvl, lvl == true_lvl)
