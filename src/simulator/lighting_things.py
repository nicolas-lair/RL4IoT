from simulator.Thing import Thing, PowerThing
from simulator.discrete_parameters import get_color_name_from_hsb
from simulator.standard_channels import PowerChannel, BrightnessChannel, ColorTemperatureChannel, ColorChannel, \
    build_description_and_change_dicts

description_keys = ['power', 'brightness', 'temperature', 'color']
goals = build_description_and_change_dicts(description_keys)


class SimpleLight(PowerThing):
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
                         init_params=init_params, is_visible=is_visible, location=location, goals_dict=goals)


class AdorneLight(PowerThing):
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
            self.brightness = BrightnessChannel(name='brightness', description="Set device's brightness")

        super().__init__(always_on=always_on, name=name, description=description, init_type=init_type,
                         init_params=init_params, is_visible=is_visible, location=location, goals_dict=goals)


class BigAssFan(PowerThing):
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
                         init_params=init_params, is_visible=is_visible, location=location, goals_dict=goals)


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
                         init_params=init_params, is_visible=is_visible, location=location, goals_dict=goals)

    def is_powered(self, state=None):
        state = self.color.get_state() if state is None else state
        return state['color']['state'][2] > 0

    def power_on(self):
        raise NotImplementedError


class StructuredHueLight(PowerThing):
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
                         init_params=init_params, is_visible=is_visible, location=location, goals_dict=goals)


if __name__ == "__main__":
    import yaml
    from simulator.discrete_parameters import params_interpreters, color_list, dimmers_levels_dict
    from simulator.oracle import Oracle
    from test_utils import test_action_effect


    def test_goals():
        adorne = AdorneLight(name='adorne', simple=True)
        adorne_on = AdorneLight(name='adorne_on', always_on=True, simple=True)

        bigass = BigAssFan(name='bigass', simple=True)
        bigass_on = BigAssFan(name='bigass_on', always_on=True, simple=True)

        hue = HueLightBulb(name='hue')

        structuredHue = StructuredHueLight(name='structuredHue', simple=True)
        structuredHue_on = StructuredHueLight(name='structuredHue_on', always_on=True, simple=True)

        oracle = Oracle([adorne, adorne_on,
                         bigass, bigass_on,
                         hue,
                         structuredHue, structuredHue_on],
                        absolute_instruction=True,
                        relative_instruction=True)
        print(yaml.dump(oracle.str_instructions))


    ########### ADORNE ###############
    def test_adorne():
        adorne = AdorneLight(name="adorne", simple=True)
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
                                   dimmers_levels_dict['brightness']]
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
        bigassfan = BigAssFan(name='bigass', simple=True)
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
                           action=('brightness', 'setPercent', 80)
                           )

        test_action_effect(bigassfan, test_name="Change brightness while on",
                           init_params=dict(power=1, brightness=80, color_temperature=50),
                           action=[('brightness', 'setPercent', params_interpreters['setPercent'](b)) for b in
                                   dimmers_levels_dict['brightness']]
                           )

        test_action_effect(bigassfan, test_name="Change temperature while on",
                           init_params=dict(power=1, brightness=80, color_temperature=50),
                           action=[('color_temperature', 'setPercent', params_interpreters['setPercent'](b)) for b in
                                   dimmers_levels_dict['temperature']]
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


    def test_bigass_with_hidden_channels():
        bigassfan = BigAssFan(name='bigass', simple=True)
        channels = bigassfan.get_channels()
        assert channels == bigassfan.get_children_nodes()

        obs_channels, act_channels = bigassfan.get_observation_channels(), bigassfan.get_action_channels()
        assert len(obs_channels) == len(act_channels) == len(channels)
        assert act_channels == bigassfan.get_children_nodes()

        channels[0].update_visibility(False)
        obs_channels, act_channels = bigassfan.get_observation_channels(), bigassfan.get_action_channels()
        assert len(obs_channels) + 1 == len(act_channels) + 1 == len(channels)
        assert act_channels == bigassfan.get_children_nodes()

        bigassfan.reset()
        obs_channels, act_channels = bigassfan.get_observation_channels(), bigassfan.get_action_channels()
        assert len(obs_channels) + 1 == len(act_channels) + 1 == len(channels)
        assert act_channels == bigassfan.get_children_nodes()

        channels[0].update_visibility(True)
        obs_channels, act_channels = bigassfan.get_observation_channels(), bigassfan.get_action_channels()
        assert len(obs_channels) == len(act_channels) == len(channels)
        assert act_channels == bigassfan.get_children_nodes()


    test_adorne()
    test_bigass()
    test_structuredhue()
    test_goals()
    test_bigass_with_hidden_channels()
