import copy
from functools import partial

from simulator.Channel import Channel
from simulator.Items import SwitchItem, DimmerItem, ColorItem, StringItem, PlayerItem, RollerShutterItem
from simulator.discrete_parameters import percent_to_level, get_color_name_from_hsb, color_list, dimmers_levels_dict, \
    TVchannels_list
from simulator.instructions import GoalDescription

power_description = {
    'turn_on': GoalDescription(sentences=["Turn on {name} {location}",
                                          "You turned on the {name} {location}"], need_power=False),
    'turn_off': GoalDescription(sentences=["Turn off {name} {location}",
                                           "You turned off the {name} {location}"], need_power=False)
}

roller_description = {
    'open': GoalDescription(sentences=["Open {name} {location}",
                                       "You opened the {name} {location}"], need_power=False),
    'close': GoalDescription(sentences=["Close {name} {location}",
                                        "You closed the {name} {location}"], need_power=False)
}

media_description = {
    'play': GoalDescription(sentences=["Play {name} {location}"]),
    'pause': GoalDescription(sentences=["Pause {name} {location}"])
}

dimmer_like_descriptions = dict()
for k, v in dimmers_levels_dict.items():
    temp_dict = dict()
    for lvl in v:
        s_list = [f"Change {{name}} {k} {{location}} to {lvl}",
                  f"You changed the {k} of {{name}} {{location}} to {lvl}"]
        temp_dict[f'{k}_level_{lvl}'] = GoalDescription(sentences=s_list)
    dimmer_like_descriptions[k] = temp_dict

color_description = dict()
for color in color_list:
    s_list = [f"Change {{name}} color {{location}} to {color}",
              f"You changed the color of {{name}} {{location}} to {color}"]
    color_description[f'{color}_color'] = GoalDescription(sentences=s_list)

source_description = dict()
for source in TVchannels_list:
    s_list = [f"Set {{name}} source {{location}} to {source}",
              f"You changed the source of {{name}} {{location}} to {source}"]
    source_description[f'{source}_TVsource'] = GoalDescription(sentences=s_list)

descriptions = {
    'power': power_description,
    'roller': roller_description,
    'color': color_description,
    **dimmer_like_descriptions,
    'TVsource': source_description,
    'media': media_description
}

change = dict()
for k in dimmers_levels_dict:
    change[k] = {
        f'increase_{k}': GoalDescription(sentences=[f"Increase {k} of {{name}} {{location}}",
                                                    f"You increased the {k} of {{name}} {{location}}"]),
        f'decrease_{k}': GoalDescription(sentences=[f"Decrease {k} of {{name}} {{location}}",
                                                    f"You decreased the {k} of {{name}} {{location}}"])
    }


def build_description_and_change_dicts(description_keys):
    d = copy.deepcopy(descriptions)
    c = copy.deepcopy(change)
    selected_description, selected_change = dict(), dict()
    for k in description_keys:
        selected_description.update(d.get(k, dict()))
        selected_change.update(c.get(k, dict()))

    goals = {
        'description': selected_description,
        'change': selected_change
    }
    return goals


def label_increase_change(increase, type):
    assert isinstance(increase, bool)
    if type == 'brightness':
        return 'increase_brightness' if increase else 'decrease_brightness'
    elif type == 'temperature':
        return 'increase_temperature' if increase else 'decrease_temperature'
    elif type == 'volume':
        return 'increase_volume' if increase else 'decrease_volume'
    else:
        raise NotImplementedError


def get_increase_change(previous_brightness, next_brightness, type):
    if next_brightness > previous_brightness:
        return label_increase_change(increase=True, type=type)
    elif previous_brightness > next_brightness:
        return label_increase_change(increase=False, type=type)
    else:
        return None


class PowerChannel(Channel):
    def __init__(self, name='power', description='switch on and off',
                 methods=dict(turnOn=True, turnOff=True), **kwargs):
        super().__init__(name=name,
                         description=description,
                         item=SwitchItem(**methods),
                         read=True,
                         write=True,
                         associated_state_description=lambda x: 'turn_on' if x else 'turn_off',
                         associated_state_change=None,
                         **kwargs
                         )


class WrapperDimmerChannel(Channel):
    def __init__(self, name, description, type,
                 methods=dict(setPercent=True, increase=True, decrease=True), **kwargs):
        super().__init__(name=name,
                         description=description,
                         item=DimmerItem(**methods, discretization={'setPercent': type}),
                         read=True,
                         write=True,
                         associated_state_description=lambda
                             p: f'{type}_level_{percent_to_level(p, lvl_type=type)}',
                         associated_state_change=partial(get_increase_change, type=type),
                         **kwargs
                         )


class BrightnessChannel(WrapperDimmerChannel):
    def __init__(self, name='brightness', description='brightness',
                 methods=dict(setPercent=True, increase=True, decrease=True), **kwargs):
        super().__init__(name=name,
                         description=description,
                         type='brightness',
                         methods=methods,
                         **kwargs
                         )


class ColorTemperatureChannel(WrapperDimmerChannel):
    def __init__(self, name='color_temperature', description='temperature',
                 methods=dict(setPercent=True, increase=True, decrease=True), **kwargs):
        super().__init__(name=name,
                         description=description,
                         type='temperature',
                         methods=methods,
                         **kwargs
                         )


class VolumeChannel(WrapperDimmerChannel):
    def __init__(self, name='volume', description='volume',
                 methods=dict(setPercent=True, increase=True, decrease=True), **kwargs):
        super().__init__(name=name,
                         description=description,
                         type='volume',
                         methods=methods,
                         **kwargs
                         )


class ColorChannel(Channel):
    def __init__(self, name='color', description='color',
                 methods=None, associated_state_description=None, associated_state_change=None, **kwargs):

        if methods is None:
            methods = dict(turnOn=True, turnOff=True, increase=True, decrease=True, setPercent=True, setHSB=True)

        if associated_state_description is None:
            associated_state_description = [
                lambda h, s, b: 'turn_on' if b > 0 else 'turn_off',
                lambda h, s, b: f'{get_color_name_from_hsb(h, s, b)}_color',
                lambda h, s, b: f'brightness_level_{percent_to_level(b, "brightness")}',
            ]
        if associated_state_change is None:
            def associated_state_change(h1, s1, b1, h2, s2, b2): return get_increase_change(b1, b2, type='brightness')

        super().__init__(name=name,
                         description=description,
                         item=ColorItem(**methods, discretization={'setHSB': 'colors', 'setPercent': 'brightness'}),
                         associated_state_description=associated_state_description,
                         associated_state_change=associated_state_change,
                         **kwargs
                         )


class TVSourceChannel(Channel):
    def __init__(self, name='source', description='channel source', **kwargs):
        super().__init__(
            name=name,
            description=description,
            item=StringItem(setString=True, discretization={'setString': 'TVchannels'}),
            read=True,  # TODO Fix Hack
            write=True,
            associated_state_description=lambda s: f'{s}_TVsource',
            associated_state_change=None,
            **kwargs
        )


class MediaPlayerChannel(Channel):
    def __init__(self, name='media', description='media control', methods=None, **kwargs):
        if methods is None:
            methods = dict(play=True, pause=True,
                           PlayPause=False, next=False, previous=False, rewind=False, fasforward=False)
        super().__init__(
            name=name,
            description=description,
            item=PlayerItem(**methods),  # TODO how to visualize state ?
            read=True,
            write=True,
            associated_state_description=lambda x: 'play' if x else 'pause',
            associated_state_change=None,
            **kwargs
        )


class RollerShutterChannel(Channel):
    def __init__(self, name='roller', description='roller', **kwargs):
        super().__init__(
            name=name,
            description=description,
            item=RollerShutterItem(up=True, down=True, setPercent=False),
            read=True,
            write=True,
            associated_state_description=lambda x: 'open' if (x < 50) else 'close',
            associated_state_change=None,
            **kwargs
        )
