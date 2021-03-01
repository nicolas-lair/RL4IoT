from abc import ABC, abstractmethod

from gym import spaces

from simulator.Channel import Channel, ChannelState
from simulator.TreeView import DescriptionNode
from simulator.instructions import GoalDescription, initialize_instruction


class ThingState(ChannelState):
    def __init__(self, state, description):
        state['description'] = description
        super().__init__(state)


class Thing(ABC, DescriptionNode):
    def __init__(self, name, description, is_visible, init_type, init_params, location, goals_dict):
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
        super().__init__(name=name, description=description, children=self.get_action_channels(), node_type='thing')
        self.goals_dict = initialize_instruction(goal_dict=goals_dict, name=name, location=location)

        # self.observation_space, self.action_space = self.build_observation_and_action_space()
        self._channels = self.get_channels()
        self.initial_values = {'is_visible': is_visible, 'init_type': init_type,
                               'init_params': init_params if init_params is not None else dict()}

        self.is_visible = None
        self.init(**self.initial_values)

    def update_visibility(self, visibility):
        self.is_visible = visibility
        self.initial_values['is_visible'] = True

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
                            channel2_name: {...},
                            ...
                            description: self.description

        """
        state = dict()
        channels = self.get_channels()
        for c in channels:
            if oracle or c.read:
                state[c.name] = c.get_state()
        state = ThingState(state, description=self.description)
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
            init_func = lambda x: init_type
        elif init_type == 'custom':
            init_func = lambda x: init_params[x]
        else:
            raise NotImplementedError('init_type should be one of {default, random, custom}')

        for c in self._channels:
            c.init(init_func(c.name))

        return self.get_state(oracle=True)

    def reset(self):
        self.init(**self.initial_values)

    def get_state_change(self, previous_state, next_state, ignore_power=False, as_string=True):
        previous_matching_descriptions = self.get_state_description(previous_state)
        was_powered = self.is_powered(previous_state)

        next_matching_descriptions = self.get_state_description(next_state)
        is_powered = self.is_powered(next_state)

        state_change_descriptions_keys = []
        for c in self._channels:
            state_change_descriptions_keys.extend(c.get_state_change_key(previous_state, next_state))

        state_change_descriptions = [self.goals_dict['change'][key] for key in state_change_descriptions_keys]
        if not (ignore_power or (is_powered and was_powered)):
            state_change_descriptions = [d for d in state_change_descriptions if not d.need_power]

        descriptions_change = set(next_matching_descriptions).difference(set(previous_matching_descriptions))
        descriptions_change = descriptions_change.union(state_change_descriptions)

        if as_string:
            descriptions_change = [d.get_instruction() for d in descriptions_change]
        return list(descriptions_change)

    def get_state_description(self, state=None, ignore_power=False):
        state = self.get_state(oracle=True) if state is None else state
        is_powered = self.is_powered(state)

        # Collect all keys
        state_description_key_list = []
        for c in self._channels:
            state_description_key_list.extend(c.get_state_description_key(state))

        # Get corresponding StateDescriptions
        matching_description = [self.goals_dict['description'][key] for key in state_description_key_list]

        # Filter if object is not powered
        if not ignore_power and not is_powered:
            matching_description = [d for d in matching_description if not d.need_power]
        return matching_description

    @abstractmethod
    def is_powered(self, state=None):
        raise NotImplementedError


class PowerThing(Thing):
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
            'turn_on': GoalDescription(sentences=[f"You turned on the {self.name}"]),
            'turn_off': GoalDescription(sentences=[f"You turned off the {self.name}"]),
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
