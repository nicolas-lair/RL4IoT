from things.Thing import PowerThing
from simulator.standard_channels import PowerChannel, VolumeChannel, MediaPlayerChannel, \
    build_description_and_change_dicts

description_keys = ['power', 'volume', 'media']
goals = build_description_and_change_dicts(description_keys)


class SimpleSpeaker(PowerThing):
    def __init__(self, name="speaker", description='simple speaker', init_type='random',
                 init_params=None, is_visible=True, location=None, simple=False, always_on=False):

        if simple:
            description = name
            self.power = PowerChannel()
            self.volume = VolumeChannel()
        else:
            self.power = PowerChannel(name='power',
                                      description="Current power setting. TV can only be powered off, not on.",
                                      methods=dict(turnOn=False, turnOff=True))
            self.volume = VolumeChannel(
                name="volume",
                description="Current volume setting. Setting and reporting absolute percent values only works when using "
                            "internal speakers. When connected to an external amp, the volume should be controlled using "
                            "increase and decrease commands."
            )

        super().__init__(always_on=always_on, name=name, description=description, init_type=init_type,
                         init_params=init_params, is_visible=is_visible, location=location, goals_dict=goals)


class SpeakerWithMediaControl(SimpleSpeaker):
    def __init__(self, name="television", description='simple television', init_type='random',
                 init_params=None, is_visible=True, location=None, simple=False, always_on=False):

        if simple:
            self.media = MediaPlayerChannel()
        else:
            self.media = MediaPlayerChannel(
                name="mediaPlayer",
                description="Media control player",
            )

        super().__init__(always_on=always_on, name=name, description=description, init_type=init_type,
                         init_params=init_params, is_visible=is_visible, location=location, simple=simple)


if __name__ == "__main__":
    import yaml
    from simulator.discrete_parameters import params_interpreters, dimmers_levels_dict
    from simulator.oracle import Oracle
    from test_utils import test_action_effect


    def test_goals():
        simple_tv = SimpleSpeaker(name='simple_tv', simple=True)
        simple_tv_on = SimpleSpeaker(name='simple_tv_on', always_on=True, simple=True)

        volume_tv = SpeakerWithMediaControl(name='volume_tv', simple=True)
        volume_tv_on = SpeakerWithMediaControl(name='volume_tv_on', always_on=True, simple=True)

        oracle = Oracle([simple_tv, simple_tv_on,
                         volume_tv, volume_tv_on],
                        relative_instruction=True,
                        absolute_instruction=True)
        print('Instruction \n', yaml.dump(oracle.str_instructions))


    ########### Simple Speaker  ###############
    def test_simple_speaker():
        thing = SimpleSpeaker(name="simple_speaker", simple=True)
        print('*' * 10 + thing.name + '*' * 10)

        test_action_effect(thing, test_name="Turn on",
                           init_params=dict(power=0, volume=50),
                           action=('power', 'turnOn')
                           )

        test_action_effect(thing, test_name="Turn off",
                           init_params=dict(power=1, volume=50),
                           action=('power', 'turnOff')
                           )

        test_action_effect(thing, test_name="Change volume while off",
                           init_params=dict(power=0, volume=50),
                           action=('volume', 'setPercent', 80)
                           )

        test_action_effect(thing, test_name="Change volume while on",
                           init_params=dict(power=1, volume=50),
                           action=[('volume', 'setPercent', params_interpreters['setPercent'](b)) for b in
                                   dimmers_levels_dict['volume']]
                           )
        test_action_effect(thing, test_name="increase function",
                           init_params=dict(power=1, volume=50),
                           action=[('volume', 'increase')]
                           )

        test_action_effect(thing, test_name="decrease function",
                           init_params=dict(power=1, volume=50),
                           action=[('volume', 'decrease')]
                           )


    ########### MEDIA TV ###############
    def test_speaker_with_control():
        thing = SpeakerWithMediaControl(name="speaker with control", simple=True)
        print('*' * 10 + thing.name + '*' * 10)

        test_action_effect(thing, test_name="Turn on",
                           init_params=dict(power=0, volume=50, media=0),
                           action=('power', 'turnOn')
                           )

        test_action_effect(thing, test_name="Turn off",
                           init_params=dict(power=1, volume=50, media=1),
                           action=('power', 'turnOff')
                           )

        test_action_effect(thing, test_name="Play -> Pause - > Play",
                           init_params=dict(power=1, volume=50, media=1),
                           action=[('media', 'pause'), ('media', 'play')]
                           )

        test_action_effect(thing, test_name="Play / Pause while off",
                           init_params=dict(power=0, volume=50, media=1),
                           action=[('media', 'pause'), ('media', 'play')]
                           )


    test_goals()
    test_simple_speaker()
    test_speaker_with_control()
