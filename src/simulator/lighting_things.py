from abc import ABC, abstractmethod
from functools import partial

from simulator.Channel import Channel
from simulator.Items import ColorItem, DimmerItem, SwitchItem, MethodUnavailableError
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
    s_list = ["Change {{name}} color {{location}} to {color}",
              "You changed the color of {{name}} {{location}} to {color}"]
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
    if next_brightness > previous_brightness:
        return label_increase_change(increase=True, type=type)
    elif previous_brightness > next_brightness:
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
    def __init__(self, name='color', description='color',
                 methods=None, associated_state_description=None, associated_state_change=None):

        if methods is None:
            methods = dict(turnOn=True, turnOff=True, increase=True, decrease=True, setPercent=True, setHSB=True)

        if associated_state_description is None:
            associated_state_description = [
                lambda h, s, b: 'turn_on' if b > 0 else 'turn_off',
                lambda h, s, b: f'{get_color_name_from_hsb(h, s, b)}_color',
                lambda h, s, b: f'lum_level_{percent_to_level(b, "brightness")}',
            ]
        if associated_state_change is None:
            def associated_state_change(h1, s1, b1, h2, s2, b2): return get_increase_change(b1, b2, type='brightness')

        super().__init__(name=name,
                         description=description,
                         item=ColorItem(**methods, discretization={'setHSB': 'colors', 'setPercent': 'brightness'}),
                         associated_state_description=associated_state_description,
                         associated_state_change=associated_state_change,
                         )


class LightBulb(Thing):
    def __init__(self, always_on=False, **kwargs):
        self.always_on = always_on
        if not self.always_on:
            assert self.power is not None
        else:
            self.power = None
        super().__init__(**kwargs)

    def is_powered(self, state=None):
        if self.always_on:
            return True
        else:
            state = self.power.get_state() if state is None else state['power']['state']
            return bool(state[0])

    def power_on(self):
        if not self.is_powered():
            self.power.do_action('turnOn')


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

        super().__init__(always_on=False, name=name, description=description, init_type=init_type,
                         init_params=init_params, is_visible=is_visible, location=location, goals_dict=Light_goals)


class AdorneLightBulb(LightBulb):
    """
    Simple light bulb supporting just brightness adjustment
    10000 episodes to master
    https://www.openhab.org/addons/bindings/adorne/spo
    """

    def __init__(self, name="medium light", description='This is a simple light bulb', init_type='random',
                 init_params=None, is_visible=True, location=None, simple=False, always_on=False):
        if simple:
            description = name
            self.power = PowerChannel()
            self.brightness = BrightnessChannel()
        else:
            self.power = PowerChannel(name='power', description='Turn device on and off')
            self.brightness = BrightnessChannel(name='brightness', description="Set device's brightness", )

        super().__init__(always_on=always_on, name=name, description=description, init_type=init_type,
                         init_params=init_params, is_visible=is_visible, location=location, goals_dict=Light_goals)


class BigAssFanLightBulb(LightBulb):
    """
    Light Object from BigAssFan https://www.openhab.org/addons/bindings/bigassfan/
    """

    def __init__(self, name="intermediate light",
                 description='This is a light bulb with color temperature setting',
                 init_type='random', init_params=None, is_visible=True, location=None, simple=False, always_on=False):
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
        super().__init__(always_on=always_on, name=name, description=description, init_type=init_type,
                         init_params=init_params, is_visible=is_visible, location=location, goals_dict=Light_goals)


class HueLightBulb(Thing):
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

        super().__init__(name=name, description=description, init_type=init_type,
                         init_params=init_params, is_visible=is_visible, location=location, goals_dict=Light_goals)

    def is_powered(self, state=None):
        state = self.color.get_state() if state is None else state
        return state['color']['state'][2] > 0

    def power_on(self):
        raise NotImplementedError


class StructuredHueLight(LightBulb):
    def __init__(self, name="colored light bulb", description='This is a colored light bulb', init_type='random',
                 init_params=None, is_visible=True, location=None, simple=False, always_on=False):
        if simple:
            description = name
        self.power = PowerChannel()
        self.color = ColorChannel(
            methods=dict(turnOn=False, turnOff=False, increase=False, decrease=False, setPercent=False,
                         setHSB=True),
            associated_state_description=[lambda h, s, b: f'{get_color_name_from_hsb(h, s, b)}_color'],
            associated_state_change=lambda *args: None
        )
        self.brightness = BrightnessChannel()
        self.color_temperature = ColorTemperatureChannel()

        super().__init__(always_on=always_on, name=name, description=description, init_type=init_type,
                         init_params=init_params, is_visible=is_visible, location=location, goals_dict=Light_goals)


if __name__ == "__main__":
    import yaml
    from simulator.discrete_parameters import params_interpreters
    from simulator.oracle import Oracle


    def test_action_effect(thing, test_name, init_params, action, debug=False):
        print('     ' + '*' * 3 + f'{test_name}' + '*' * 3)
        s1 = thing.init(is_visible=True, init_type='custom', init_params=init_params)
        action = action if isinstance(action, list) else [action]
        if debug: print(s1)
        for a in action:
            try:
                thing.do_action(*a)
                s2 = thing.get_state(oracle=True)
                print(f"{sorted(thing.get_state_change(s1, s2))} after action {a}")
                if debug: print(s2)
                s1 = s2
            except MethodUnavailableError:
                s2 = thing.get_state(oracle=True)
                print(f'Action {a} is not available for {thing.name}', sorted(thing.get_state_change(s1, s2)))
                if debug: print(s2)
                s1 = s2


    def test_goals():
        adorne = AdorneLightBulb(name='adorne', simple=True)
        adorne_on = AdorneLightBulb(name='adorne_on', always_on=True, simple=True)

        bigass = BigAssFanLightBulb(name='bigass', simple=True)
        bigass_on = BigAssFanLightBulb(name='bigass_on', always_on=True, simple=True)

        hue = HueLightBulb(name='hue')

        structuredHue = StructuredHueLight(name='structuredHue', simple=True)
        structuredHue_on = StructuredHueLight(name='structuredHue_on', always_on=True, simple=True)

        oracle = Oracle([adorne, adorne_on,
                         bigass, bigass_on,
                         hue,
                         structuredHue, structuredHue_on])
        print(yaml.dump(oracle.str_instructions))


    ########### ADORNE ###############
    def test_adorne():
        adorne = AdorneLightBulb(name="adorne", simple=True)
        print('*' * 10 + adorne.name + '*' * 10)

        test_action_effect(adorne, test_name="Turn on",
                           init_params=dict(power=0, brightness=0),
                           action=('power', 'turnOn')
                           )

        test_action_effect(adorne, test_name="Turn off",
                           init_params=dict(power=1, brightness=50),
                           action=('power', 'turnOff')
                           )

        test_action_effect(adorne, test_name="Change brightness while off",
                           init_params=dict(power=0, brightness=50),
                           action=('brightness', 'setPercent', 50)
                           )

        test_action_effect(adorne, test_name="Change brightness while on",
                           init_params=dict(power=1, brightness=80),
                           action=[('brightness', 'setPercent', params_interpreters['setPercent'](b)) for b in
                                   levels_dict['brightness']]
                           )

        test_action_effect(adorne, test_name="Unchange brightness while on",
                           init_params=dict(power=1, brightness=80),
                           action=('brightness', 'setPercent', 80)
                           )

        test_action_effect(adorne, test_name="increase function",
                           init_params=dict(power=1, brightness=50),
                           action=('brightness', 'increase')
                           )

        test_action_effect(adorne, test_name="decrease function",
                           init_params=dict(power=1, brightness=50),
                           action=('brightness', 'decrease')
                           )


    ########### BIGASSFAN LIGHTBULB ###############
    def test_bigass():
        bigassfan = BigAssFanLightBulb(name='bigass', simple=True)
        print('*' * 10 + bigassfan.name + '*' * 10)

        test_action_effect(bigassfan, test_name='Turn on',
                           init_params=dict(power=0, brightness=0, color_temperature=0),
                           action=('power', 'turnOn')
                           )
        test_action_effect(bigassfan, test_name='Turn on while already on',
                           init_params=dict(power=1, brightness=50, color_temperature=50),
                           action=('power', 'turnOn')
                           )

        test_action_effect(bigassfan, test_name='Turn off',
                           init_params=dict(power=1, brightness=50, color_temperature=50),
                           action=('power', 'turnOff')
                           )

        test_action_effect(bigassfan, test_name='Turn off while already off',
                           init_params=dict(power=0, brightness=50, color_temperature=50),
                           action=('power', 'turnOff')
                           )

        test_action_effect(bigassfan, test_name="Change brightness while off",
                           init_params=dict(power=0, brightness=50, color_temperature=0),
                           action=('brightness', 'setPercent', 50)
                           )

        test_action_effect(bigassfan, test_name="Change brightness while on",
                           init_params=dict(power=1, brightness=80, color_temperature=50),
                           action=[('brightness', 'setPercent', params_interpreters['setPercent'](b)) for b in
                                   levels_dict['brightness']]
                           )

        test_action_effect(bigassfan, test_name="Change temperature while on",
                           init_params=dict(power=1, brightness=80, color_temperature=50),
                           action=[('color_temperature', 'setPercent', params_interpreters['setPercent'](b)) for b in
                                   levels_dict['temperature']]
                           )

        test_action_effect(bigassfan, test_name="Unchange brightness while on",
                           init_params=dict(power=1, brightness=80, color_temperature=50),
                           action=('brightness', 'setPercent', 80)
                           )

        test_action_effect(bigassfan, test_name="increase function",
                           init_params=dict(power=1, brightness=50, color_temperature=50),
                           action=[('brightness', 'increase'), ('color_temperature', 'increase')]
                           )

        test_action_effect(bigassfan, test_name="decrease function",
                           init_params=dict(power=1, brightness=50, color_temperature=50),
                           action=[('brightness', 'decrease'), ('color_temperature', 'decrease')]
                           )


    ########### Hue LIGHTBULB ###############
    def test_hue():
        hue = HueLightBulb(name="hue", simple=True)
        print('*' * 10 + hue.name + '*' * 10)

        test_action_effect(hue, test_name='Turn on',
                           init_params=dict(color=(0, 0, 0), color_temperature=0),
                           action=('color', 'turnOn')
                           )
        test_action_effect(hue, test_name='Turn on while already on',
                           init_params=dict(color=(0, 0, 100), color_temperature=0),
                           action=('color', 'turnOn')
                           )

        test_action_effect(hue, test_name='Turn off',
                           init_params=dict(color=(0, 0, 100), color_temperature=0),
                           action=('color', 'turnOff')
                           )

        test_action_effect(hue, test_name='Turn off while already off',
                           init_params=dict(color=(0, 0, 0), color_temperature=0),
                           action=('color', 'turnOff')
                           )

        test_action_effect(hue, test_name="Change brightness while off",
                           init_params=dict(color=(0, 0, 0), color_temperature=0),
                           action=('color', 'setPercent', 50),
                           )

        test_action_effect(hue, test_name="Change brightness while on",
                           init_params=dict(color=(0, 0, 100), color_temperature=0),
                           action=('color', 'setPercent', 50), debug=True
                           )

        test_action_effect(hue, test_name="Unchange brightness while on",
                           init_params=dict(color=(0, 0, 100), color_temperature=0),
                           action=('color', 'setPercent', 100), debug=True
                           )

        test_action_effect(hue, test_name="increase function",
                           init_params=dict(color=(0, 0, 50), color_temperature=50),
                           action=[('color', 'increase'), ('color_temperature', 'increase')]
                           )

        test_action_effect(hue, test_name="decrease function",
                           init_params=dict(color=(0, 0, 50), color_temperature=50),
                           action=[('color', 'decrease'), ('color_temperature', 'decrease')]
                           )

        test_action_effect(hue, test_name="rotate color",
                           init_params=dict(color=(180, 0, 50), color_temperature=50),
                           action=[('color', 'setHSB', params_interpreters['setHSB'](c)) for c in color_list]
                           )


    ########### Structuredhue LIGHTBULB ###############
    def test_structuredhue():
        structuredhue = StructuredHueLight(name="structuredhue", simple=True)

        print('*' * 10 + structuredhue.name + '*' * 10)

        test_action_effect(structuredhue, test_name='Turn on',
                           init_params=dict(power=0, color=(0, 0, 0), brightness=0, color_temperature=0),
                           action=[('color', 'turnOn'), ('power', 'turnOn')]
                           )

        test_action_effect(structuredhue, test_name='Turn on while already on',
                           init_params=dict(power=1, color=(0, 0, 0), brightness=0, color_temperature=0),
                           action=[('color', 'turnOn'), ('power', 'turnOn')]
                           )

        test_action_effect(structuredhue, test_name='Turn off',
                           init_params=dict(power=1, color=(0, 0, 0), brightness=0, color_temperature=0),
                           action=[('color', 'turnOff'), ('power', 'turnOff')]
                           )

        test_action_effect(structuredhue, test_name='Turn off while already off',
                           init_params=dict(power=0, color=(0, 0, 0), brightness=0, color_temperature=0),
                           action=[('color', 'turnOff'), ('power', 'turnOff')]
                           )

        test_action_effect(structuredhue, test_name="Change brightness while off",
                           init_params=dict(power=0, color=(0, 0, 0), brightness=0, color_temperature=0),
                           action=[('color', 'setPercent', 50), ('brightness', 'setPercent', 50)]
                           )

        test_action_effect(structuredhue, test_name="Change brightness while on",
                           init_params=dict(power=1, color=(0, 0, 0), brightness=0, color_temperature=0),
                           action=[('color', 'setPercent', 50), ('brightness', 'setPercent', 50)]
                           )

        test_action_effect(structuredhue, test_name="Unchange brightness while on",
                           init_params=dict(power=1, color=(0, 0, 50), brightness=100, color_temperature=0),
                           action=[('color', 'setPercent', 50), ('brightness', 'setPercent', 100)]
                           )

        test_action_effect(structuredhue, test_name="increase function",
                           init_params=dict(power=1, color=(0, 0, 50), brightness=50, color_temperature=50),
                           action=[('color', 'increase'), ('color_temperature', 'increase'), ('brightness', 'increase')]
                           )

        test_action_effect(structuredhue, test_name="increase function while off",
                           init_params=dict(power=0, color=(0, 0, 50), brightness=50, color_temperature=50),
                           action=[('color', 'increase'), ('color_temperature', 'increase'), ('brightness', 'increase')]
                           )

        test_action_effect(structuredhue, test_name="decrease function",
                           init_params=dict(power=1, color=(0, 0, 50), brightness=50, color_temperature=50),
                           action=[('color', 'decrease'), ('color_temperature', 'decrease'), ('brightness', 'decrease')]
                           )

        test_action_effect(structuredhue, test_name="decrease function while off",
                           init_params=dict(power=0, color=(0, 0, 50), brightness=50, color_temperature=50),
                           action=[('color', 'decrease'), ('color_temperature', 'decrease'), ('brightness', 'decrease')]
                           )

        test_action_effect(structuredhue, test_name="rotate color",
                           init_params=dict(power=1, color=(180, 0, 0), brightness=0, color_temperature=0),
                           action=[('color', 'setHSB', params_interpreters['setHSB'](c)) for c in color_list]
                           )
        test_action_effect(structuredhue, test_name="rotate color while off",
                           init_params=dict(power=0, color=(180, 0, 0), brightness=0, color_temperature=0),
                           action=[('color', 'setHSB', params_interpreters['setHSB'](c)) for c in color_list]
                           )


    test_adorne()
    test_bigass()
    test_structuredhue()
    test_goals()
