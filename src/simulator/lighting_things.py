from functools import partial

from simulator.Channel import Channel
from simulator.Items import ColorItem, DimmerItem, SwitchItem, INCREASE_DECREASE_STEP
from simulator.Thing import Thing
from simulator.instructions import StateDescription
from simulator.utils import color_list, levels_dict, get_color_name_from_hsb, percent_to_level

Light_instruction = {
    'turn_on': StateDescription(sentences=["You turned on the {name} {location}"], need_power=False),
    'turn_off': StateDescription(sentences=["You turned off the {name} {location}"], need_power=False),
    'increase_lum': StateDescription(sentences=["You increased the luminosity of {name} {location}"]),
    'decrease_lum': StateDescription(sentences=["You decreased the luminosity of {name} {location}"]),
    'warmer_color': StateDescription(sentences=["You made the light of {name} warmer {location}"]),
    'colder_color': StateDescription(sentences=["You made the light of {name} colder {location}"]),
}

for color in color_list:
    Light_instruction[f'{color}_color'] = StateDescription(
        sentences=["You set the color of {{name}} {{location}} to {color}".format(color=color)])

for level in levels_dict['brightness']:
    # if level == 'average':
    #     s_list = ["{{location}} the luminosity of {{name}} is now average"]
    # else:
    #     s_list = ["{{name}} is now {level}"]
    s_list = ["{{location}} the luminosity of {{name}} is now {level}"]
    Light_instruction[f'lum_level_{level}'] = StateDescription(sentences=[s.format(level=level) for s in s_list])

for level in levels_dict['temperature']:
    Light_instruction[f'temp_level_{level}'] = StateDescription(
        sentences=["{{location}} the light temperature of {{name}} is now {level}".format(level=level)])


def label_increase_change(increase, type):
    assert isinstance(increase, bool)
    if type == 'brightness':
        return 'increase_lum' if increase else 'decrease_lum'
    elif type == 'temperature':
        return 'warmer_color' if increase else 'colder_color'


def get_increase_change(previous_brightness, next_brightness, type):
    if next_brightness - previous_brightness > INCREASE_DECREASE_STEP:
        return label_increase_change(increase=True, type=type)
    elif previous_brightness - next_brightness > INCREASE_DECREASE_STEP:
        return label_increase_change(increase=False, type=type)
    else:
        return None


class PowerChannel(Channel):
    def __init__(self, name, description):
        super().__init__(name=name,
                         description=description,
                         item=SwitchItem(turnOn=True, turnOff=True),
                         read=True,
                         write=True,
                         associated_state_description=lambda x: 'turn_on' if x else 'turn_off',
                         associated_state_change=None)


class BrightnessChannel(Channel):
    def __init__(self, name, description, methods=dict(setPercent=True)):
        super().__init__(name=name,
                         description=description,
                         item=DimmerItem(**methods, discretization={'setPercent': 'brightness'}),
                         read=True,
                         write=True,
                         associated_state_description=lambda p: f'lum_level_{percent_to_level(p, "brightness")}',
                         associated_state_change=partial(get_increase_change, type='brightness')
                         )


class ColorTemperatureChannel(Channel):
    def __init__(self, name, description, methods=dict(setPercent=True)):
        super().__init__(name=name,
                         description=description,
                         item=DimmerItem(**methods, discretization={'setPercent': 'temperature'}),
                         read=True,
                         write=True,
                         associated_state_description=lambda p: f'temp_level_{percent_to_level(p, "temperature")}',
                         associated_state_change=partial(get_increase_change, type="temperature")
                         )


class ColorChannel(Channel):
    def __init__(self, name, description):
        super().__init__(name=name,
                         description=description,
                         item=ColorItem(turnOn=True, turnOff=True, increase=True, decrease=True, setPercent=True,
                                        setHSB=True, discretization={'setHSB': 'colors', 'setPercent': 'brightness'}),
                         associated_state_description=[
                             lambda h, s, b: 'turn_on' if b > 0 else 'turn_off',
                             lambda h, s, b: f'{get_color_name_from_hsb(h, s, b)}_color',
                             lambda h, s, b: f'lum_level_{percent_to_level(b, "brightness")}',
                         ],
                         associated_state_change=lambda h1, s1, b1, h2, s2, b2: get_increase_change(b1, b2,
                                                                                                    type='brightness'),
                         )


class LightBulb(Thing):
    pass


class AdorneLightBulb(LightBulb):
    """
    Simple light bulb supporting just brightness adjustment

    https://www.openhab.org/addons/bindings/adorne/
    """

    def __init__(self, name="level one light bulb", description='This is a simple light bulb', init_type='random',
                 init_params=None, is_visible=True, location=None):
        self.power = PowerChannel(name='power', description='Turn device on and off')
        self.brightness = BrightnessChannel(name='brightness', description="Set device's brightness", )
        super().__init__(name=name, description=description, init_type=init_type, init_params=init_params,
                         is_visible=is_visible, location=location, instruction_dict=Light_instruction)

    def is_powered(self, state=None):
        state = self.get_state(oracle=True) if state is None else state
        return state['power']['state'][0] == 1


class BigAssFanLightBulb(LightBulb):
    """
    Light Object from BigAssFan https://www.openhab.org/addons/bindings/bigassfan/
    """

    def __init__(self, name="intermediate light bulb", description='This is a light bulb with color temperature setting',
                 init_type='random', init_params=None, is_visible=True, location=None):
        self.light_power = PowerChannel(name='light_power', description='Power on / off the light')
        self.light_level = BrightnessChannel(name='light_level', description="Adjust the brightness of the light", )
        self.light_hue = ColorTemperatureChannel(name='light_hue',
                                                 description="Adjust the color temperature of the light",
                                                 methods=dict(setPercent=True, increase=True, decrease=True))
        super().__init__(name=name, description=description, init_type=init_type, init_params=init_params,
                         is_visible=is_visible, location=location, instruction_dict=Light_instruction)

    def is_powered(self, state=None):
        state = self.get_state(oracle=True) if state is None else state
        return state['light_power']['state'][0] == 1


class HueLightBulb(LightBulb):
    """
    Thing type 0210 (https://www.openhab.org/addons/bindings/hue/)
    """

    def __init__(self, name="colored light bulb", description='This is a colored light bulb', init_type='random',
                 init_params=None, is_visible=True, location=None):
        self.color = ColorChannel(
            name='color',
            description="This channel supports full color control with hue, saturation and brightness values"
        )

        self.color_temperature = ColorTemperatureChannel(
            name='color_temperature',
            description='This channel supports adjusting the color temperature from cold (0%) to warm (100%)',
            methods=dict(increase=True, decrease=True, setPercent=True)
        )

        super().__init__(name=name, description=description, init_type=init_type, init_params=init_params,
                         is_visible=is_visible, location=location, instruction_dict=Light_instruction)

    def is_powered(self, state=None):
        state = self.get_state(oracle=True) if state is None else state
        return state['color']['state'][2] > 0


if __name__ == "__main__":
    a = AdorneLightBulb(location='the kitchen', init_type='random')

    a.reset()
    state1 = a.get_state(oracle=True)
    descriptionstate1 = a.get_state_description()
    print(state1)
    print([d.get_instruction() for d in descriptionstate1])

    a.reset()
    state2 = a.get_state(oracle=True)
    descriptionstate2 = a.get_state_description(state2)
    print(state2)
    print([d.get_instruction() for d in descriptionstate2])

    a.reset()
    state3 = a.get_state(oracle=True)
    descriptionstate3 = a.get_state_description(state3)
    print(state3)
    print([d.get_instruction() for d in descriptionstate3])

    a.reset()
    state4 = a.get_state(oracle=True)
    descriptionstate4 = a.get_state_description(state4)
    print(state4)
    print([d.get_instruction() for d in descriptionstate4])

    a.reset()
    state5 = a.get_state(oracle=True)
    descriptionstate5 = a.get_state_description(state5)
    print(state5)
    print([d.get_instruction() for d in descriptionstate5])

    print('*' * 5 + 'STATE CHANGE' + '*' * 5)
    print(a.get_state_change(state1, state1))
    print(a.get_state_change(state1, state2))
    print(a.get_state_change(state1, state3))
    print(a.get_state_change(state1, state4))
    print(a.get_state_change(state1, state5))