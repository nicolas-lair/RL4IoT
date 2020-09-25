from abc import ABC, abstractmethod

from simulator.Items import *
from simulator.Channel import Channel
from simulator.utils import percent_to_level, levels_dict
from simulator.TreeView import DescriptionNode
from simulator.instructions import StateDescription, initialize_instruction
from discrete_parameters import TVchannels_list


class Thing(ABC, DescriptionNode):
    def __init__(self, name, description, is_visible, init_type, init_params, location, instruction_dict):
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
        # self.name = name #TODO check if necessary
        self.instruction = initialize_instruction(instruction_dict=instruction_dict, name=name, location=location)

        # self.observation_space, self.action_space = self.build_observation_and_action_space()
        self._channels = self.get_channels()
        self.initial_values = {'is_visible': is_visible, 'init_type': init_type,
                               'init_params': init_params if init_params is not None else dict()}
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

    def get_observation_channels(self):
        action_channels = [x for x in vars(self).values() if (isinstance(x, Channel) and x.read)]
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

    def _get_state(self, oracle):
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
            if oracle or c.read:
                state[c.name] = {
                    'state': c.get_state(),
                    'description': c.description,
                    'item_type': c.item.type
                }

                if c.node_embedding is not None:
                    state[c.name].update({'embedding': c.node_embedding})
        return state

    def get_state(self, oracle):
        if self.is_visible:
            return self._get_state(oracle)
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

    def get_state_change(self, previous_state, next_state):
        previous_matching_descriptions = self.get_state_description(previous_state)
        was_powered = self.is_powered(previous_state)

        next_matching_descriptions = self.get_state_description(next_state)
        is_powered = self.is_powered(next_state)

        state_change_descriptions_keys = []
        for c in self._channels:
            state_change_descriptions_keys.extend(c.get_state_change_key(previous_state, next_state))

        state_change_descriptions = [self.instruction[key] for key in state_change_descriptions_keys]

        descriptions_change = set(next_matching_descriptions).difference(set(previous_matching_descriptions))
        descriptions_change = descriptions_change.union(state_change_descriptions)

        if not is_powered * was_powered:
            descriptions_change = [d for d in descriptions_change if not d.need_power]
        achieved_instructions = [d.get_instruction() for d in descriptions_change]
        return achieved_instructions

    def get_state_description(self, state=None):
        state = self.get_state(oracle=True) if state is None else state
        is_powered = self.is_powered(state)

        # Collect all keys
        state_description_key_list = []
        for c in self._channels:
            state_description_key_list.extend(c.get_state_description_key(state))

        # Get corresponding StateDescriptions
        matching_description = [self.instruction[key] for key in state_description_key_list]

        # Filter if object is not powered
        if not is_powered:
            matching_description = [d for d in matching_description if not d.need_power]
        return matching_description

    @abstractmethod
    def is_powered(self, state=None):
        raise NotImplementedError


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
            'turn_on': StateDescription(sentences=[f"You turned on the {self.name}"]),
            'turn_off': StateDescription(sentences=[f"You turned off the {self.name}"]),
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

    def get_state_description(self, state):
        matching_instructions = []

        state = state["switch_binary"]["state"][0]
        # ON / OFF
        if state == 1:
            matching_instructions.append(self.instruction['turn_on'])
        elif state == 0:
            matching_instructions.append(self.instruction['turn_off'])
        else:
            raise NotImplementedError
        return matching_instructions

    def get_state_change(self, previous_state, next_state):
        return super().get_state_change(previous_state, next_state)


class LGTV(Thing):
    """
    https://www.openhab.org/addons/bindings/lgwebos/

    See also PanasonicTV and SamsungTV
    """

    def __init__(self, name, description, init_type='default', init_params=None, is_visible=True):
        self.name = name

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
            item=DimmerItem(increase=True, decrease=True, setPercent=True, discretization={'setPercent': 'volume'})
        )

        self.channel = Channel(
            name="channel",
            description="Current channel. Use only the channel number as command to update the channel.",
            item=StringItem(setString=True, discretization={'setString': 'TVchannels'}),
            read=False  # TODO Fix Hack
        )

        # TODO cannot handle right now, maybe copy paste
        # self.toast = Channel(
        #     name='toast',
        #     description="Displays a short message on the TV screen. See also rules section.",
        #     item=StringItem(setString=True
        #     read=False)
        # )

        self.mediaPlayer = Channel(
            name="mediaPlayer",
            description="Media control player",
            item=PlayerItem(PlayPause=True, next=False, previous=False),  # TODO how to visualize state ?
            read=False
        )

        self.mediaStop = Channel(
            name="mediaStop",
            description="Media control stop",
            item=SwitchItem(turnOff=True),
            read=False
        )

        # TODO cannot handle this one for now
        # self.appLauncher = Channel(
        #     name='applauncher',
        #     description="Application ID of currently running application. This also allows to start applications on the TV by sending a specific Application ID to this channel.",
        #     item=StringItem(setString=True)
        # )

        self.instruction = {
            'turn_on': StateDescription(sentences=[f'You turned on the {self.name}']),
            'turn_off': StateDescription(sentences=[f'You turned off {self.name}']),
            'mute': StateDescription(sentences=[f'You muted the {self.name}']),
            'unmute': StateDescription(sentences=[f'You restored the sound on {self.name}']),
            'increased_volume': StateDescription(sentences=[f'You increased the volume of {self.name}']),
            'decreased_volume': StateDescription(sentences=[f'You decreased the volume of {self.name}']),
            'play': StateDescription(sentences=[f'You played the film on {self.name}']),
            'pause': StateDescription(sentences=[f'You paused the film on {self.name}']),
            # 'next': StateDescription(sentences=[f'You changed the {self.name} to the next channel']),
            # 'previous': StateDescription(sentences=[f'You changed the {self.name} to the previous channel']),
            'stop': StateDescription(sentences=[f'You stopped the film on {self.name}']),
        }

        for level in levels_dict['volume']:
            self.instruction[f'volume_level_{level}'] = StateDescription(
                sentences=[f"The volume of {self.name} is now {level}"])
        for channel in TVchannels_list:
            self.instruction[f'TVchannel_{channel}'] = StateDescription(
                sentences=[f"{self.name} is now on channel {channel}"])

        super().__init__(name=name, description=description, init_type=init_type, init_params=init_params,
                         is_visible=is_visible)

        # Ignore this for now
        # self.rcButton = Channel(
        #     name='rcButton',
        #     description="Simulates pressing of a button on the TV's remote control. See below for a list of button names.",
        #     item=StringItem(setString=True),
        #     read=False
        # )

    def get_state_description(self, current_state):
        matching_instructions = []

        # power
        if current_state["power"]["state"] == 0:
            matching_instructions.append(self.instruction['turn_off'])
        else:
            matching_instructions.append(self.instruction['turn_on'])

        # mute
        if current_state["mute"]["state"] == 0:
            matching_instructions.append(self.instruction['mute'])
        else:
            matching_instructions.append(self.instruction['unmute'])

        # volume level
        volume = current_state['volume']['state'][0]
        if volume != 0:
            volume_lvl = percent_to_level(volume, lvl_type='volume')
            matching_instructions.append(self.instruction[f'volume_level_{volume_lvl}'])

        # channel
        channel = current_state['channel']['state'][0]
        matching_instructions.append(self.instruction[f'TVchannel_{channel}'])

        # mediaplayer
        player_status = current_state['mediaPlayer']['state'][0]
        if player_status == 1:
            matching_instructions.append(self.instruction[f'play'])
        else:
            matching_instructions.append(self.instruction[f'pause'])

        # stop_status = current_state['mediastop']['state'][0]
        # if stop_status == 0:
        #     matching_instructions.append(self.instruction['stop'])

        return matching_instructions

    def get_state_change(self, previous_state, next_state):
        achieved_descriptions = super().get_state_change(previous_state, next_state)

        achieved_state_transition = []

        previous_volume = previous_state["volume"]["state"][0]
        next_volume = next_state["volume"]["state"][0]
        # Check volume change
        ### INCREASE_DECREASE_STEP is like a threshold for the oracle to detect a change
        if next_volume >= INCREASE_DECREASE_STEP + previous_volume:
            achieved_state_transition.append(self.instruction['increased_volume'])
        if next_volume + INCREASE_DECREASE_STEP <= previous_volume:
            achieved_state_transition.append(self.instruction['decreased_volume'])

        # Check Stop
        previous_stop = previous_state["mediaStop"]["state"][0]
        next_stop = next_state["mediaStop"]["state"][0]
        if previous_stop != next_stop:
            achieved_state_transition.append(self.instruction['stop'])

        achieved_str_instructions = achieved_descriptions + [i.get_instruction() for i in
                                                             achieved_state_transition]
        return achieved_str_instructions


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
