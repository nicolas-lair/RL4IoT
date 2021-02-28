from functools import partial
from simulator.utils import level_to_percent, _get_color_name_from_hsb, _percent_to_level, color_to_hue

N_COLORS = 3
N_LEVELS = 3
DEFAULT_LEVEL = True

if N_COLORS == 3:
    color_list = ['red', 'yellow', 'blue']
    color_h_inf = [0, 121, 241]
    color_h_sup = [120, 240, 360]
else:
    color_list = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink']
    color_h_inf = [0, 18, 50, 64, 167, 252, 300, 335]
    color_h_sup = [17, 49, 63, 166, 251, 299, 334, 360]

if DEFAULT_LEVEL:
    if N_LEVELS == 5:
        brightness_level = volume_level = temperature_level = ['very low', 'low', 'average', 'high', 'very high']
    elif N_LEVELS == 3:
        brightness_level = volume_level = temperature_level = ['low', 'average', 'high']
    else:
        raise NotImplementedError
else:
    if N_LEVELS == 5:
        brightness_level = ['very dark', 'dark', 'average', 'bright', 'very bright']
        volume_level = ['very quiet', 'quiet', 'average', 'loud', 'very loud']
        temperature_level = ['very cold', 'cold', 'average', 'warm', 'very warm']
    elif N_LEVELS == 3:
        brightness_level = ['dark', 'average', 'bright']
        volume_level = ['quiet', 'average', 'loud']
        temperature_level = ['cold', 'average', 'warm']
    else:
        raise NotImplementedError

dimmers_levels_dict = {
    'brightness': brightness_level, 'volume': volume_level, 'temperature': temperature_level
}

TVchannels_list = ['1', '2', '3']

setPercent_params = dimmers_levels_dict

setString_params = dict(
    TVchannels=TVchannels_list
)

setHSB_params = dict(
    colors=color_list
)

discrete_parameters = {
    'setPercent': setPercent_params,
    'setHSB': setHSB_params,
    'setString': setString_params,
}

params_interpreters = {
    'setPercent': partial(level_to_percent, lvl_dict=dimmers_levels_dict),
    'setHSB': partial(color_to_hue, color_list=color_list, color_h_inf=color_h_inf, color_h_sup=color_h_sup),
    'setString': lambda x: x,
}

get_color_name_from_hsb = partial(_get_color_name_from_hsb, color_list=color_list, color_h_inf=color_h_inf,
                                  color_h_sup=color_h_sup)
percent_to_level = partial(_percent_to_level, lvl_dict=dimmers_levels_dict)

if __name__ == '__main__':
    for c in color_list:
        print(c, params_interpreters['setHSB'](c))
