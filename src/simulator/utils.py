from itertools import cycle


def get_percent_bound_level(lvl_list, min_lvl=0, max_lvl=100):
    n_lvl = len(lvl_list)
    step = max_lvl / n_lvl
    bound_level = [min_lvl + int(step * i) for i in range(n_lvl + 1)]
    return bound_level


def _get_color_name_from_hsb(h, s, b, color_list, color_h_inf, color_h_sup):
    assert isinstance(h, int) and (0 <= h <= 360), 'hue should be a int between 0 and 360'
    assert isinstance(s, int) and (0 <= s <= 100), 'saturation should be a int between 0 and 100'
    assert isinstance(b, int) and (0 <= b <= 100), 'brightness should be a int between 0 and 100'

    colors = cycle(color_list)

    for h_inf, h_sup in zip(color_h_inf, color_h_sup):
        color = next(colors)
        if h_inf <= h <= h_sup:
            return color
    raise EOFError


def color_to_hue(string_color, color_list, color_h_inf, color_h_sup):
    idx = color_list.index(string_color)
    h = (color_h_inf[idx] + color_h_sup[idx]) // 2
    return h


def color_to_hsb(string_color, color_list, color_h_inf, color_h_sup):
    h = color_to_hue(string_color, color_list, color_h_inf, color_h_sup)
    return h, 75, 70


def find_level_list(lvl, lvl_dict):
    right_list = None
    for list_ in lvl_dict.values():
        right_list = list_ if lvl in list_ else None
        if right_list:
            break
    if right_list is None:
        raise NotImplementedError
    return right_list


def level_to_percent(lvl, lvl_dict):
    lvl_list = find_level_list(lvl, lvl_dict)
    percent_bound_level = get_percent_bound_level(lvl_list)
    idx = lvl_list.index(lvl)
    p = (percent_bound_level[idx] + percent_bound_level[idx + 1]) // 2
    return p


def _percent_to_level(p, lvl_type, lvl_dict):
    i = 0
    lvl_list = lvl_dict[lvl_type]
    percent_bound_level = get_percent_bound_level(lvl_list)

    while i < len(percent_bound_level):
        if p <= percent_bound_level[i + 1]:
            break
        else:
            i += 1
    return lvl_list[i]


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

    print(get_percent_bound_level([1, 2, 3]))

    # test_dict = {
    #     0: "very dark",
    #     10: "very dark",
    #     20: "very dark",
    #     30: "dark",
    #     40: "dark",
    #     50: "average",
    #     60: "average",
    #     70: "bright",
    #     80: "bright",
    #     90: "very bright",
    #     100: "very bright",
    # }
    #
    # for p, true_lvl in test_dict.items():
    #     lvl = percent_to_level(p, lvl_type='brightness')
    #     print(p, lvl, lvl == true_lvl)
