from itertools import cycle

color_list = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink']
color_h_inf = [0, 18, 50, 64, 167, 252, 300, 335]
color_h_sup = [17, 49, 63, 166, 251, 299, 334, 360]

N_LEVELS = 5
percent_bound_level = [0, 20, 40, 60, 80, 100]
brightness_level = ['very dark', 'dark', 'average', 'bright', 'very bright']
volume_level = ['very quiet', 'quiet', 'average', 'loud', 'very loud']
temperature_level = ['very cold', 'cold', 'average', 'warm', 'very warm']
levels_dict = {
    'brightness': brightness_level, 'volume': volume_level, 'temperature': temperature_level
}


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


def find_level_list(lvl):
    right_list = None
    for list_ in levels_dict.values():
        right_list = list_ if lvl in list_ else None
        if right_list:
            break
    if right_list is None:
        raise NotImplementedError
    return right_list


def level_to_percent(lvl):
    lvl_list = find_level_list(lvl)
    idx = lvl_list.index(lvl)
    p = (percent_bound_level[idx] + percent_bound_level[idx + 1]) // 2
    return p


def percent_to_level(p, lvl_type):
    i = 0
    while i < len(percent_bound_level):
        if p <= percent_bound_level[i + 1]:
            break
        else:
            i += 1
    return levels_dict[lvl_type][i]


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
        0: "very dark",
        10: "very dark",
        20: "very dark",
        30: "dark",
        40: "dark",
        50: "average",
        60: "average",
        70: "bright",
        80: "bright",
        90: "very bright",
        100: "very bright",
    }

    for p, true_lvl in test_dict.items():
        lvl = percent_to_level(p, lvl_type='brightness')
        print(p, lvl, lvl == true_lvl)
