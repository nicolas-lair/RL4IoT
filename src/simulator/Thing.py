from Items import *

from gym import spaces

from simulator.Channel import Channel
from simulator.utils import get_color_name_from_hsb, percent_to_level, color_list, percent_level
from simulator.TreeView import DescriptionNode
from simulator.instructions import StateDescription


class Thing(DescriptionNode):
    def __init__(self, name, description, is_visible, init_type, init_params):
        """
        Init a generic Thing object as a Node of the environment. The super function that instantiates subclass should
        be called after the definition of the channels. The channels are indeed the children of the node.
        :param name: string
        :param description: string
        :param is_visible: boolean
        :param init_type: 'default', 'random', 'custom'
        :param init_params: if init_type is custom, dict {key: value} where key are channels name and value are channels
        init value
        """
        super().__init__(name=name, description=description, children=self.get_action_channels())
        self.observation_space, self.action_space = self.build_observation_and_action_space()
        self._channels = self.get_channels()
        self.initial_values = {'is_visible': is_visible, 'init_type': init_type, 'init_params': init_params}
        # self.description_embedding = None
        # self.item_type = None

        self.is_visible = None
        self.init(**self.initial_values)

    def update_visibility(self, visibility):
        self.is_visible = visibility

    def get_channels(self):
        """
        Initialize the list of _channels of the Thing object
        :return: list of Channel objects
        """
        try:
            return self._channels
        except AttributeError:
            self._channels = [x for x in vars(self).values() if isinstance(x, Channel)]
        return self._channels

    def get_action_channels(self):
        action_channels = [x for x in vars(self).values() if (isinstance(x, Channel) and x.write)]
        return action_channels

    def build_observation_and_action_space(self):
        channels = self.get_channels()
        channels_name = [chn.name for chn in channels]

        channels_observation_space = [chn.get_observation_space() for chn in channels]
        observation_space = spaces.Dict(dict(zip(channels_name, channels_observation_space)))

        channels_action_space = [chn.get_action_space() for chn in channels]
        action_space = spaces.Dict(dict(zip(channels_name, channels_action_space)))
        return observation_space, action_space

    # def _build_description_and_item_type(self):
    #     self.description = dict()
    #     self.item_type = dict()
    #     channels = self.get_channels()
    #     for c in channels:
    #         self.description[c.name] = c.description
    #         self.item_type[c.name] = c.item.type

    def get_observation_space(self):
        return self.observation_space

    def get_action_space(self):
        # if self.action_space is None:
        #     self.build_observation_and_action_space()
        return self.action_space

    def _get_state(self):
        """
        Get internal user_state of the object as a dict of dict of channel/item user_state WITH their description and item type
        :return: user_state : { channel1_name: {
                                            'user_state': channel1_state,
                                            'description': "...",
                                            'item_type': "..."
                                            },
                            channel2_name: {...}

        """
        state = dict()
        channels = self.get_channels()
        for c in channels:
            state[c.name] = {
                'state': c.get_state(),
                'description': c.description,
                'item_type': c.item.type
            }

            if c.node_embedding is not None:
                state[c.name].update({'embedding': c.node_embedding})
        return state

    def get_state(self):
        if self.is_visible:
            return self._get_state()
        else:
            return None, None, None

    def do_action(self, channel, action, params=None):
        channel = getattr(self, channel)
        channel.do_action(action, params)

    def init(self, is_visible, init_type, init_params=None):
        if init_params is None:
            init_params = dict()
        self.is_visible = is_visible

        if init_type in ['default', 'random']:
            init_param = lambda x: init_type
        elif init_type == 'custom':
            init_param = lambda x: init_params[x]
        else:
            raise NotImplementedError('init_type should be one of {default, random, custom}')

        for c in self._channels:
            c.init(init_param(c.name))

    def reset(self):
        self.init(**self.initial_values)

    # TODO Collect all possible actions from object
    def get_state_change(self, previous_state, next_state):
        raise NotImplementedError


class LightBulb(Thing):
    """
    Thing type 0210 (https://www.openhab.org/addons/bindings/hue/)
    """

    def __init__(self, name="lightbulb", description='This is a light bulb', init_type='default', init_params=None,
                 is_visible=True):
        if init_params is None:
            init_params = dict()
        self.name = name
        self.color = Channel(
            name='color',
            description="This channel supports full color control with hue, saturation and brightness values",
            item=ColorItem(turnOn=True, turnOff=True, increase=True, decrease=True, setPercent=True,
                           setHSB=True),
        )

        self.color_temperature = Channel(
            name='color_temperature',
            description='This channel supports adjusting the color temperature from cold (0%) to warm (100%)',
            item=DimmerItem(increase=True, decrease=True, setPercent=True),
        )

        self.instruction = {
            # 'color_change': StateDescription(sentences=["You changed the color of {name} to {color}"]),
            'turn_on': StateDescription(sentences=[f"You turned on the {self.name}"]),
            'turn_off': StateDescription(sentences=[f"You turned off the {self.name}"]),
            'increase_lum': StateDescription(sentences=[f"You increased the luminosity of {self.name}"]),
            'decrease_lum': StateDescription(sentences=[f"You decreased the luminosity of {self.name}"]),
            # 'lum_change': StateDescription(sentences=["The luminosity of {name} is now {level}"]),
            'warmer_color': StateDescription(sentences=[f"You made the light of {self.name} warmer"]),
            'colder_color': StateDescription(sentences=[f"You made the light of {self.name} colder"]),
        }

        for color in color_list:
            self.instruction[f'{color}_color'] = StateDescription(
                sentences=[f"You changed the color of {self.name} to {color}"])
        for level in percent_level:
            self.instruction[f'lum_level_{level}'] = StateDescription(
                sentences=[f"The luminosity of {self.name} is now {level}"]),

            super().__init__(name=name, description=description, init_type=init_type, init_params=init_params,
                             is_visible=is_visible)
        # # TODO decide if use or not : maybe not
        # self.alert = Channel(
        #     name='alert',
        #     description='This channel supports displaying alerts by flashing the bulb either once or multiple times. Valid values are: NONE, SELECT and LSELECT.',
        #     item=StringItem()
        # )

    def get_state_description(self, current_state):
        matching_instructions = []

        # COlOR
        h, s, b = current_state["color"]["state"]
        color = get_color_name_from_hsb(h, s, b)
        matching_instructions.append(self.instruction[f'{color}_color'])

        # ON / OFF
        if b == 0:
            matching_instructions.append(self.instruction['turn_off'])
        else:
            matching_instructions.append(self.instruction['turn_on'])

        # BRIGHTNESS LVL
        brightness_lvl = percent_to_level(b)
        matching_instructions.append(self.instruction[f'lum_level_{brightness_lvl}'])

        # # COLOR TEMPERATURE LVL
        # color_temperature = current_state["color_temperature"]["state"][0]

        return matching_instructions

    def get_state_change(self, previous_state, next_state):
        previous_matching_descriptions = self.get_state_description(previous_state)
        next_matching_descriptions = self.get_state_description(next_state)
        descriptions_change = set(next_matching_descriptions).difference(set(previous_matching_descriptions))

        achieved_instructions = [d.get_random_instruction() for d in descriptions_change]

        # Check color channel change
        h, s, b = previous_state["color"]["state"]
        new_h, new_s, new_b = next_state["color"]["state"]
        # previous_color = get_color_name_from_hsb(h, s, b)
        # next_color = get_color_name_from_hsb(new_h, new_s, new_b)
        # if previous_color != next_color:
        #     achieved_instructions.append(
        #         random.choice(self.instruction['color_change']).format(name=self.name, color=next_color))
        # if b == 0 and new_b > 0:
        #     achieved_instructions.append(random.choice(self.instruction['turn_on']))
        # if b > 0 and new_b == 0:
        #     achieved_instructions.append(random.choice(self.instruction['turn_off']))

        ### INCREASE_DECREASE_STEP is a threshold for the oracle to detect a change
        if new_b >= INCREASE_DECREASE_STEP + b:
            achieved_instructions.append(random.choice(self.instruction['increase_lum']))
        if new_b + INCREASE_DECREASE_STEP <= b:
            achieved_instructions.append(random.choice(self.instruction['decrease_lum']))

        # b_lvl = percent_to_level(b)
        # new_b_lvl = percent_to_level(new_b)
        # # Increase the brightness using set Percent
        # if b_lvl != new_b_lvl:
        #     achieved_instructions.append(
        #         random.choice(self.instruction['lum_change']).format(name=self.name, level=new_b_lvl))

        previous_color_temperature = previous_state["color_temperature"]["state"][0]
        next_color_temperature = next_state["color_temperature"]["state"][0]
        # Check color_temperature change
        ### INCREASE_DECREASE_STEP is like a threshold for the oracle to detect a change
        if next_color_temperature >= INCREASE_DECREASE_STEP + previous_color_temperature:
            achieved_instructions.append(random.choice(self.instruction['warmer_color']))
        if next_color_temperature + INCREASE_DECREASE_STEP <= previous_color_temperature:
            achieved_instructions.append(random.choice(self.instruction['colder_color']))

        return achieved_instructions


class PlugSwitch(Thing):
    """
    https://www.openhab.org/addons/bindings/zwave/thing.html?manufacturer=everspring&file=an180_0_0.html
    """

    def __init__(self, name="plug switch", description='This is a switch that controls multiple switch',
                 init_type='default', init_params=None, is_visible=True):
        self.name = name
        self.switch_binary = Channel(name="switch_binary",
                                     description="Switch the power on and off.",
                                     item=SwitchItem(turnOn=True, turnOff=True),
                                     )

        self.instruction = {
            'turn_on': [f"You turned on the {self.name}"],
            'turn_off': [f"You turned off the {self.name}"],
        }
        super().__init__(name=name, description=description, init_type=init_type, init_params=init_params,
                         is_visible=is_visible)

        # Ignore both channel
        # self.alarm = Channel(
        #     name='alarm_general',
        #     description="Indicates if an alarm is triggered.",
        #     item=SwitchItem(),
        #     write=False,
        # )
        #
        # self.alarm_power = Channel(
        #     name='alarm_power',
        #     description="Indicates if a power alarm is triggered.",
        #     item=SwitchItem(),
        #     write=False,
        # )

    def do_action(self, channel, action, params=None):
        assert channel == "switch_binary", f"Switch_binary is the only available channel, {channel} was called instead"
        super().do_action(channel, action, params)

    def get_state_change(self, previous_state, next_state):
        achieved_instructions = []
        previous_state = previous_state["switch_binary"]["state"][0]
        next_state = next_state["switch_binary"]["state"][0]
        if previous_state and not next_state:
            achieved_instructions.append(random.choice(self.instruction['turn_off']))
        if not previous_state and next_state:
            achieved_instructions.append(random.choice(self.instruction['turn_on']))
        return achieved_instructions


class LGTV(Thing):
    """
    https://www.openhab.org/addons/bindings/lgwebos/

    See also PanasonicTV and SamsungTV
    """

    def __init__(self, name, description, init_type='default', init_params=None, is_visible=True):
        self.power = Channel(
            name='power',
            description="Current power setting. TV can only be powered off, not on.",
            item=SwitchItem(turnOff=True),
        )

        self.mute = Channel(
            name="mute",
            description="Current mute setting.",
            item=SwitchItem(turnOnOff=True)
        )

        self.volume = Channel(
            name="volume",
            description="Current volume setting. Setting and reporting absolute percent values only works when using "
                        "internal speakers. When connected to an external amp, the volume should be controlled using "
                        "increase and decrease commands.",
            item=DimmerItem(increase=True, decrease=True, setPercent=True)
        )

        self.channel = Channel(
            name="channel",
            description="Current channel. Use only the channel number as command to update the channel.",
            item=StringItem(setString=True)  # TODO maybe discretize this or use Number item
        )

        self.toast = Channel(
            name='toast',
            description="Displays a short message on the TV screen. See also rules section.",
            item=StringItem(setString=True)  # TODO Copy and paste ?
        )

        self.mediaPlayer = Channel(
            name="mediaplayer",
            description="Media control player",
            item=PlayerItem(PlayPause=True, next=True, previous=True),  # TODO how to visualize state ?
            read=False
        )

        self.mediaStop = Channel(
            name="mediastop",
            description="Media control stop",
            item=SwitchItem(turnOff=True),
            read=False
        )

        self.appLauncher = Channel(
            name='applauncher',
            description="Application ID of currently running application. This also allows to start applications on the TV by sending a specific Application ID to this channel.",
            item=StringItem(setString=True)
        )

        super().__init__(name=name, description=description, init_type=init_type, init_params=init_params,
                         is_visible=is_visible)

        # Ignore this for now
        # self.rcButton = Channel(
        #     name='rcButton',
        #     description="Simulates pressing of a button on the TV's remote control. See below for a list of button names.",
        #     item=StringItem(setString=True),
        #     read=False
        # )


class Speaker(Thing):
    """
    https://www.openhab.org/addons/bindings/sonyaudio/
    HT-CT800, SRS-ZR5, HT-ST5000, HT-ZF9, HT-Z9F, HT-MT500

    maybe compare with Sonos or check STR-1080 for multiple zone compatibility
    """

    def __init__(self, name, description, init_type='default', init_params=None, is_visible=True):
        self.power = Channel(
            name="power",
            description="Main power on/off",
            item=SwitchItem(turnOnOff=True)
        )

        # TODO Discretize this
        self.input = Channel(
            name="input",
            description="Set or get the input source",
            item=StringItem(setString=True)
        )
        self.volume = Channel(
            name="volume",
            description="Set or get the master volume",
            item=DimmerItem(increase=True, decrease=True, setPercent=True)
        )
        self.mute = Channel(
            name="mute",
            description="Set or get the mute state of the master volume",
            item=SwitchItem(turnOnOff=True)
        )

        super().__init__(name=name, description=description, init_type=init_type, init_params=init_params,
                         is_visible=is_visible)

        # TODO check what is sound Field, for now disable
        # self.soundField = Channel(
        #     name="soundField",
        #     description="Sound Field",
        #     item=StringItem(setString=True)
        # )


#
# class Store(Thing):
#     def __init__(self, name="store", connected_things=None, is_visible=True):
#         super().__init__(name=name, connected_things=connected_things, is_visible=is_visible)
#

class Chromecast(Thing):
    def __init__(self, name, description, init_type='default', init_params=None, is_visible=True):
        self.control = Channel(
            name='control',
            description="Player control; currently only supports play/pause and does not correctly update, if the state changes on the device itself",
            item=PlayerItem(PlayPause=True)
        )
        self.stop = Channel(
            name='stop',
            description="Send ON to this channel: Stops the Chromecast. If this channel is ON, the Chromecast is stopped, otherwise it is in another state (see control channel)",
            item=SwitchItem(turnOn=True)
        )

        # TODO link this channel to other volume _channels
        self.volume = Channel(
            name='volume',
            description="Control the volume, this is also updated if the volume is changed by another app",
            item=DimmerItem(increase=True, decrease=True, setPercent=True)
        )

        self.mute = Channel(
            name='mute',
            description="Mute the audio",
            item=SwitchItem(turnOnOff=True)
        )

        self.playuri = Channel(
            name='playuri',
            description="Can be used to tell the Chromecast to play media from a given url",
            item=StringItem(setString=True)
        )

        self.appName = Channel(
            name='appName',
            description="Name of currently running application",
            item=StringItem(),
            write=False
        )
        self.appId = Channel(
            name='appId',
            description="ID of currently running application",
            item=StringItem(),
            write=False
        )
        self.idling = Channel(
            name='idling',
            description="Read-only indication on weather Chromecast is on idle screen",
            item=SwitchItem(),
            write=False
        )

        # TODO imlement Number ITem Time
        self.currentTime = Channel(
            name='currentTime',
            description="Current time of currently playing media",
            item=NumberItem(),
            write=False
        )

        self.duration = Channel(
            name='duration',
            description="Duration of current track (null if between tracks)",
            item=NumberItem(),
            write=False
        )

        self.metadataType = Channel(
            name='metadataType',
            description="Type of metadata, this indicates what metadata may be available. One of: GenericMediaMetadata, MovieMediaMetadata, TvShowMediaMetadata, MusicTrackMediaMetadata, PhotoMediaMetadata.",
            item=StringItem(),
            write=False
        )

        super().__init__(name=name, description=description, init_type=init_type, init_params=init_params,
                         is_visible=is_visible)

        # TODO See what to to with available metadata: Not necessary maybe?


if __name__ == "__main__":
    Env = {
        'plug': PlugSwitch(),
        'light': LightBulb(),
        'TV': LGTV(),
        'speaker': Speaker(),
        'chromecast': Chromecast(),
    }

    for k, v in Env.items():
        print(k, v.get_action_space())
