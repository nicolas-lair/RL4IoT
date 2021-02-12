from abc import ABC, abstractmethod
from functools import partial

from simulator.Channel import Channel
from simulator.Items import ColorItem, DimmerItem, SwitchItem, INCREASE_DECREASE_STEP
from simulator.Thing import Thing
from simulator.instructions import GoalDescription
from simulator.discrete_parameters import color_list, levels_dict, get_color_name_from_hsb, percent_to_level

Light_description = {
    'turn_on': GoalDescription(sentences=["Turn on {name} {location}",
                                          "You turned on the {name} {location}"], need_power=False),
    'turn_off': GoalDescription(sentences=["Turn off {name} {location}",
                                           "You turned off the {name} {location}"], need_power=False)
}

for color in color_list:
    s_list = ["Set {{name}} {{location}} to {color}",
              "You set the color of {{name}} {{location}} to {color}"]
    Light_description[f'{color}_color'] = GoalDescription(sentences=[s.format(color=color) for s in s_list])

for level in levels_dict['brightness']:
    s_list = ["Set {{name}} brightness {{location}} to {level}",
              "{{location}} the brightness of {{name}} is now {level}"]
    Light_description[f'lum_level_{level}'] = GoalDescription(sentences=[s.format(level=level) for s in s_list])

for level in levels_dict['temperature']:
    s_list = ["Set {{name}} temperature {{location}} to {level}",
              "{{location}} the light temperature of {{name}} is now {level}"]
    Light_description[f'temp_level_{level}'] = GoalDescription(sentences=[s.format(level=level) for s in s_list])

Light_change = {
    'increase_lum': GoalDescription(sentences=["Increase brightness of {name} {location}",
                                               "You increased the brightness of {name} {location}"]),
    'decrease_lum': GoalDescription(sentences=["Decrease brightness of {name} {location}",
                                               "You decreased the brightness of {name} {location}"]),
    'warmer_color': GoalDescription(sentences=["Increase temperature of {name} {location}",
                                               "You made the light of {name} warmer {location}"]),
    'colder_color': GoalDescription(sentences=["Decrease temperature of {name} {location}",
                                               "You made the light of {name} colder {location}"]),
}

Light_goals = {
    'description': Light_description,
    'change': Light_change
}


def label_increase_change(increase, type):
    assert isinstance(increase, bool)
    if type == 'brightness':
        return 'increase_lum' if increase else 'decrease_lum'
    elif type == 'temperature':
        return 'warmer_color' if increase else 'colder_color'


def get_increase_change(previous_brightness, next_brightness, type):
    if (next_brightness - previous_brightness > INCREASE_DECREASE_STEP) \
            or (previous_brightness + INCREASE_DECREASE_STEP > 100):
        return label_increase_change(increase=True, type=type)
    elif (previous_brightness - next_brightness > INCREASE_DECREASE_STEP) \
            or (previous_brightness - INCREASE_DECREASE_STEP < 0):
        return label_increase_change(increase=False, type=type)
    else:
        return None


class PowerChannel(Channel):
    def __init__(self, name='power', description='switch on and off'):
        super().__init__(name=name,
                         description=description,
                         item=SwitchItem(turnOn=True, turnOff=True),
                         read=True,
                         write=True,
                         associated_state_description=lambda x: 'turn_on' if x else 'turn_off',
                         associated_state_change=None)


class BrightnessChannel(Channel):
    def __init__(self, name='brightness', description='brightness',
                 methods=dict(setPercent=True, increase=True, decrease=True)):
        super().__init__(name=name,
                         description=description,
                         item=DimmerItem(**methods, discretization={'setPercent': 'brightness'}),
                         read=True,
                         write=True,
                         associated_state_description=lambda
                             p: f'lum_level_{percent_to_level(p, lvl_type="brightness")}',
                         associated_state_change=partial(get_increase_change, type='brightness')
                         )


class ColorTemperatureChannel(Channel):
    def __init__(self, name='color_temperature', description='temperature',
                 methods=dict(setPercent=True, increase=True, decrease=True)):
        super().__init__(name=name,
                         description=description,
                         item=DimmerItem(**methods, discretization={'setPercent': 'temperature'}),
                         read=True,
                         write=True,
                         associated_state_description=lambda
                             p: f'temp_level_{percent_to_level(p, lvl_type="temperature")}',
                         associated_state_change=partial(get_increase_change, type="temperature")
                         )


class ColorChannel(Channel):
    def __init__(self, name='color', description='color'):
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
    def is_powered(self, state=None):
        state = self.power.get_state() if state is None else state['power']['state']
        # state2 = self.get_state(oracle=True) if state is None else state
        # assert state == state2
        return bool(state)


class SimpleLight(LightBulb):
    """
    Very simple light with only on and off
    """

    def __init__(self, name="simple light", description='This is a very simple light bulb', init_type='random',
                 init_params=None, is_visible=True, location=None, simple=False):
        if simple:
            description = name
            self.power = PowerChannel()
        else:
            self.power = PowerChannel(name='power', description='Turn device on and off')

        super().__init__(name=name, description=description, init_type=init_type, init_params=init_params,
                         is_visible=is_visible, location=location, goals_dict=Light_goals)

    def power_on(self):
        self.power.do_action('turnOn')


class AlwaysOnLight(LightBulb):
    """
    Very simple light with only on and off
    900 episodes to master
    """

    def __init__(self, name="bright light", description='This is a simple light bulb', init_type='random',
                 init_params=None, is_visible=True, location=None, simple=False):
        if simple:
            description = name
            self.brightness = BrightnessChannel()
        else:
            self.brightness = BrightnessChannel(name='brightness', description="Set device's brightness", )

        super().__init__(name=name, description=description, init_type=init_type, init_params=init_params,
                         is_visible=is_visible, location=location, goals_dict=Light_goals)

    def is_powered(self, state=None):
        return True

    def power_on(self):
        pass


class AdorneLightBulb(LightBulb):
    """
    Simple light bulb supporting just brightness adjustment
    10000 episodes to master
    https://www.openhab.org/addons/bindings/adorne/spo
    """

    def __init__(self, name="medium light", description='This is a simple light bulb', init_type='random',
                 init_params=None, is_visible=True, location=None, simple=False):
        if simple:
            description = name
            self.power = PowerChannel()
            self.brightness = BrightnessChannel()
        else:
            self.power = PowerChannel(name='power', description='Turn device on and off')
            self.brightness = BrightnessChannel(name='brightness', description="Set device's brightness", )

        super().__init__(name=name, description=description, init_type=init_type, init_params=init_params,
                         is_visible=is_visible, location=location, goals_dict=Light_goals)

    def power_on(self):
        self.power.do_action('turnOn')


class BigAssFanLightBulb(LightBulb):
    """
    Light Object from BigAssFan https://www.openhab.org/addons/bindings/bigassfan/
    """

    def __init__(self, name="intermediate light",
                 description='This is a light bulb with color temperature setting',
                 init_type='random', init_params=None, is_visible=True, location=None, simple=False):
        if simple:
            description = name
            self.power = PowerChannel()
            self.brightness = BrightnessChannel()
            self.color_temperature = ColorTemperatureChannel()
        else:
            self.power = PowerChannel(name='light_power', description='Power on / off the light')
            self.brightness = BrightnessChannel(name='light_level', description="Adjust the brightness of the light", )
            self.color_temperature = ColorTemperatureChannel(name='light_hue',
                                                             description="Adjust the color temperature of the light",
                                                             methods=dict(setPercent=True, increase=True,
                                                                          decrease=True))
        super().__init__(name=name, description=description, init_type=init_type, init_params=init_params,
                         is_visible=is_visible, location=location, goals_dict=Light_goals)

    # def is_powered(self, state=None):
    #     state = self.get_state(oracle=True) if state is None else state
    #     return state['light_power']['state'][0] == 1

    def power_on(self):
        self.power.do_action('turnOn')


class HueLightBulb(LightBulb):
    """
    Thing type 0210 (https://www.openhab.org/addons/bindings/hue/)
    """

    def __init__(self, name="colored light bulb", description='This is a colored light bulb', init_type='random',
                 init_params=None, is_visible=True, location=None, simple=False):
        if simple:
            description = name
            self.color = ColorChannel()
            self.color_temperature = ColorTemperatureChannel()

        else:
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
                         is_visible=is_visible, location=location, goals_dict=Light_goals)

    def is_powered(self, state=None):
        state = self.get_state(oracle=True) if state is None else state
        return state['color']['state'][2] > 0

    def power_on(self):
        pass


if __name__ == "__main__":
    from oracle import Oracle

    adorne = AdorneLightBulb()
    bigassfan = BigAssFanLightBulb()
    hue = HueLightBulb()

    ########### ADORNE ###############
    thing = adorne
    oracle = Oracle([adorne])
    print('*' * 10 + thing.name + '*' * 10)
    print('     ' + '*' * 3 + 'Turn on' + '*' * 3)
    s1 = thing.init(is_visible=True, init_type='custom', init_params=dict(power=0, brightness=0))
    thing.do_action('power', 'turnOn')
    s2 = thing.get_state(oracle=True)
    print(thing.get_state_change(s1, s2))

    print('     ' + '*' * 3 + 'Turn off' + '*' * 3)
    s1 = thing.init(is_visible=True, init_type='custom', init_params=dict(power=1, brightness=0))
    thing.do_action('power', 'turnOff')
    s2 = thing.get_state(oracle=True)
    print(thing.get_state_change(s1, s2))

    print('     ' + '*' * 3 + 'Change brightness while off' + '*' * 3)
    s1 = thing.init(is_visible=True, init_type='custom', init_params=dict(power=0, brightness=0))
    thing.do_action('brightness', 'setPercent', 50)
    s2 = thing.get_state(oracle=True)
    print(thing.get_state_change(s1, s2))

    print('     ' + '*' * 3 + 'Change brightness while on' + '*' * 3)
    s1 = thing.init(is_visible=True, init_type='custom', init_params=dict(power=1, brightness=0))
    thing.do_action('brightness', 'setPercent', 50)
    s2 = thing.get_state(oracle=True)
    print(thing.get_state_change(s1, s2))

    print('     ' + '*' * 3 + 'Unchange brightness while on' + '*' * 3)
    s1 = thing.init(is_visible=True, init_type='custom', init_params=dict(power=1, brightness=50))
    thing.do_action('brightness', 'setPercent', 50)
    s2 = thing.get_state(oracle=True)
    print(thing.get_state_change(s1, s2))
    assert oracle.was_achieved({thing.name: s1}, {thing.name: s2},
                               'the luminosity of level one light bulb is now average')

    ########### BIGASSFAN LIGHTBULB ###############
    thing = bigassfan
    print('*' * 10 + thing.name + '*' * 10)
    print('     ' + '*' * 3 + 'Turn on' + '*' * 3)
    s1 = thing.init(is_visible=True, init_type='custom', init_params=dict(light_power=0, light_level=0, light_hue=0))
    thing.do_action('light_power', 'turnOn')
    s2 = thing.get_state(oracle=True)
    print(thing.get_state_change(s1, s2))

    print('     ' + '*' * 3 + 'Turn off' + '*' * 3)
    s1 = thing.init(is_visible=True, init_type='custom', init_params=dict(light_power=1, light_level=0, light_hue=0))
    thing.do_action('light_power', 'turnOff')
    s2 = thing.get_state(oracle=True)
    print(thing.get_state_change(s1, s2))

    print('     ' + '*' * 3 + 'Change brightness while off' + '*' * 3)
    s1 = thing.init(is_visible=True, init_type='custom', init_params=dict(light_power=0, light_level=0, light_hue=0))
    thing.do_action('light_level', 'setPercent', 50)
    s2 = thing.get_state(oracle=True)
    print(thing.get_state_change(s1, s2))

    print('     ' + '*' * 3 + 'Change brightness while on' + '*' * 3)
    s1 = thing.init(is_visible=True, init_type='custom', init_params=dict(light_power=1, light_level=0, light_hue=0))
    thing.do_action('light_level', 'setPercent', 50)
    s2 = thing.get_state(oracle=True)
    print(thing.get_state_change(s1, s2))

    print('     ' + '*' * 3 + 'Turn off while off while off' + '*' * 3)
    s1 = thing.init(is_visible=True, init_type='custom', init_params=dict(light_power=0, light_level=0, light_hue=0))
    thing.do_action('light_power', 'turnOff')
    s2 = thing.get_state(oracle=True)
    print(thing.get_state_change(s1, s2))

    print('     ' + '*' * 3 + 'Turn on while off while on' + '*' * 3)
    s1 = thing.init(is_visible=True, init_type='custom', init_params=dict(light_power=1, light_level=0, light_hue=0))
    thing.do_action('light_power', 'turnOn')
    s2 = thing.get_state(oracle=True)
    print(thing.get_state_change(s1, s2))

    # s1 = hue.init(is_visible=True, init_type='custom', init_params=dict(color=(0, 0, 0), color_temperature=0))
    # hue.do_action('color', 'turnOn')
    # s2 = hue.get_state(oracle=True)
    # print(hue.get_state_change(s1, s2))
    #
    # a = AdorneLightBulb(location='the kitchen', init_type='random')
    #
    # a.reset()
    # state1 = a.get_state(oracle=True)
    # descriptionstate1 = a.get_state_description()
    # print(state1)
    # print([d.get_instruction() for d in descriptionstate1])
    #
    # a.reset()
    # state2 = a.get_state(oracle=True)
    # descriptionstate2 = a.get_state_description(state2)
    # print(state2)
    # print([d.get_instruction() for d in descriptionstate2])
    #
    # a.reset()
    # state3 = a.get_state(oracle=True)
    # descriptionstate3 = a.get_state_description(state3)
    # print(state3)
    # print([d.get_instruction() for d in descriptionstate3])
    #
    # a.reset()
    # state4 = a.get_state(oracle=True)
    # descriptionstate4 = a.get_state_description(state4)
    # print(state4)
    # print([d.get_instruction() for d in descriptionstate4])
    #
    # a.reset()
    # state5 = a.get_state(oracle=True)
    # descriptionstate5 = a.get_state_description(state5)
    # print(state5)
    # print([d.get_instruction() for d in descriptionstate5])
    #
    # print('*' * 5 + 'STATE CHANGE' + '*' * 5)
    # print(a.get_state_change(state1, state1))
    # print(a.get_state_change(state1, state2))
    # print(a.get_state_change(state1, state3))
    # print(a.get_state_change(state1, state4))
    # print(a.get_state_change(state1, state5))
