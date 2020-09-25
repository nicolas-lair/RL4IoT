from simulator.utils import color_list, levels_dict, color_to_hsb, level_to_percent

TVchannels_list = [str(i) for i in range(6)]

setPercent_params = levels_dict

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
    'setPercent': level_to_percent,
    'setHSB': color_to_hsb,
    'setString': lambda x: x,
}

