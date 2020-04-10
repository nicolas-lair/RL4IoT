from Items import *

from gym import spaces

from utils import get_color_name_from_hsb


# TODO Update Volume, Mute, PLayer, Stop for connected things for TV, Chromecast and Speaker
class Thing:
    def __init__(self, name, connected_things, is_visible=True):
        self.initial_value = {'name': name, 'connected_things': connected_things, 'is_visible': is_visible}
        self.name = name
        self.is_visible = is_visible
        self.connected_things = set()

        self.observation_space = None
        self.action_space = None

        self.description = None
        self.item_type = None

        if connected_things is not None:
            self.connect_thing(connected_things)
        self._channels = None

    def update_visibility(self, visibility):
        self.is_visible = visibility

    def connect_thing(self, things):
        if isinstance(things, Thing):
            self.connected_things.add(things)
        elif isinstance(things, list):
            for t in things:
                self.connect_thing(t)
        else:
            raise NotImplementedError

    def get_channels(self):
        """
        Initialize the list of _channels of the Thing object
        :return: list of Channel objects
        """
        if self._channels is None:
            self._channels = [x for x in vars(self).values() if isinstance(x, Channel)]
        return self._channels

    def build_observation_and_action_space(self):
        channels = self.get_channels()
        channels_name = [chn.name for chn in channels]

        channels_observation_space = [chn.get_observation_space() for chn in channels]
        self.observation_space = spaces.Dict(dict(zip(channels_name, channels_observation_space)))

        channels_action_space = [chn.get_action_space() for chn in channels]
        self.action_space = spaces.Dict(dict(zip(channels_name, channels_action_space)))

    def _build_description_and_item_type(self):
        self.description = dict()
        self.item_type = dict()
        channels = self.get_channels()
        for c in channels:
            self.description[c.name] = c.description
            self.item_type[c.name] = c.item.type

    def get_observation_space(self):
        if self.observation_space is None:
            self.build_observation_and_action_space()
        return self.observation_space

    def get_action_space(self):
        if self.action_space is None:
            self.build_observation_and_action_space()
        return self.action_space

    def _get_state(self):
        """
        Get internal state of the object as a dict of dict of channel/item state WITH their description and item type
        :return: state : { channel1_name: {
                                            'state': channel1_state,
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
        return state

    def get_state(self):
        if self.is_visible:
            return self._get_state()
        else:
            return None, None, None

    def do_action(self, channel, action, params=None):
        channel = getattr(self, channel)
        channel.do_action(action, params)

    def reset(self):
        self.__init__(**self.initial_value)

    def get_state_change(self, previous_state, next_state):
        raise NotImplementedError


class Channel:
    def __init__(self, name, description, item, value=None, read=True, write=True):
        self.name = name
        self.description = description

        self.item = item
        self.initial_value = value
        if value is not None:
            self.item.set_state(value)

        self.read = read
        self.write = write

    def get_state(self):
        return self.item.get_state()

    def set_state(self, value):
        return self.item.set_state(value)

    def get_observation_space(self):
        return self.item.observation_space

    def get_action_space(self):
        return self.item.action_space

    def do_action(self, action, params=None):
        if params is None:
            params = []
        getattr(self.item, action)(self.item, *params)


class LightBulb(Thing):
    """
    Thing type 0210 (https://www.openhab.org/addons/bindings/hue/)
    """

    def __init__(self, name="lightbulb", connected_things=None, is_visible=True):
        super().__init__(name=name, connected_things=connected_things, is_visible=is_visible)
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
            value=50
        )

        # # TODO decide if use or not : maybe not
        # self.alert = Channel(
        #     name='alert',
        #     description='This channel supports displaying alerts by flashing the bulb either once or multiple times. Valid values are: NONE, SELECT and LSELECT.',
        #     item=StringItem()
        # )

    def get_state_change(self, previous_state, next_state):
        achieved_instructions = []

        # Check color channel change
        previous_color_state = previous_state["color"]["state"]
        next_color_state = next_state["color"]["state"]
        previous_color = get_color_name_from_hsb(*previous_color_state)
        next_color = get_color_name_from_hsb(*next_color_state)
        if previous_color != next_color:
            achieved_instructions.append(f"You changed the color to {next_color} of {self.name}")
        if previous_color_state[2] == 0 and next_color_state[2] > 0:
            achieved_instructions.append(f"You turned on the {self.name}")
        if previous_color_state[2] > 0 and next_color_state[2] == 0:
            achieved_instructions.append(f"You turned off the {self.name}")
        ### INCREASE_DECREASE_STEP is like a threshold for the oracle to detect a change
        if next_color_state[2] >= INCREASE_DECREASE_STEP + previous_color_state[2]:
            achieved_instructions.append(f"You increased the luminosity of {self.name}")
        if next_color_state[2] + INCREASE_DECREASE_STEP <= previous_color_state[2]:
            achieved_instructions.append(f"You decreased the luminosity of {self.name}")

        previous_color_temperature = previous_state["color_temperature"]["state"][0]
        next_color_temperature = next_state["color_temperature"]["state"][0]
        # Check color_temperature change
        ### INCREASE_DECREASE_STEP is like a threshold for the oracle to detect a change
        if next_color_temperature >= INCREASE_DECREASE_STEP + previous_color_temperature:
            achieved_instructions.append(f"You made the light of {self.name} warmer")
        if next_color_temperature + INCREASE_DECREASE_STEP <= previous_color_temperature:
            achieved_instructions.append(f"You made the light of {self.name} colder")

        return achieved_instructions


class PlugSwitch(Thing):
    """
    https://www.openhab.org/addons/bindings/zwave/thing.html?manufacturer=everspring&file=an180_0_0.html
    """

    def __init__(self, name="plugswitch", connected_things=None, is_visible=True):
        super().__init__(name=name, connected_things=connected_things, is_visible=is_visible)
        self.switch_binary = Channel(name="switch_binary",
                                     description="Switch the power on and off.",
                                     item=SwitchItem(turnOn=True, turnOff=True),
                                     )

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
        power_status = self.switch_binary.get_state()
        for thing in self.connected_things:
            thing.update_visibility(power_status)

    def get_state_change(self, previous_state, next_state):
        achieved_instructions = []
        previous_state = previous_state["switch_binary"]["state"][0]
        next_state = next_state["switch_binary"]["state"][0]
        if previous_state and not next_state:
            achieved_instructions.append(f"You turned off the {self.name}")
        if not previous_state and next_state:
            achieved_instructions.append(f"You turned on the {self.name}")
        return achieved_instructions

class LGTV(Thing):
    """
    https://www.openhab.org/addons/bindings/lgwebos/

    See also PanasonicTV and SamsungTV
    """

    def __init__(self, name="LGTV", power=1, mute=0, connected_things=None, is_visible=True):
        super().__init__(name=name, connected_things=connected_things, is_visible=is_visible)
        self.initial_value.update({'power': power, 'mute': mute})
        self.power = Channel(
            name='power',
            description="Current power setting. TV can only be powered off, not on.",
            item=SwitchItem(turnOff=True),
            value=power
        )

        self.mute = Channel(
            name="mute",
            description="Current mute setting.",
            item=SwitchItem(turnOnOff=True),
            value=mute
        )

        self.volume = Channel(
            name="volume",
            description="Current volume setting. Setting and reporting absolute percent values only works when using internal speakers. When connected to an external amp, the volume should be controlled using increase and decrease commands.",
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

    def __init__(self, name="speaker", connected_things=None, is_visible=True):
        super().__init__(name=name, connected_things=connected_things, is_visible=is_visible)

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

        # TODO check what is sound Field, for now disable
        # self.soundField = Channel(
        #     name="soundField",
        #     description="Sound Field",
        #     item=StringItem(setString=True)
        # )


class Store(Thing):
    def __init__(self, name="store", connected_things=None, is_visible=True):
        super().__init__(name=name, connected_things=connected_things, is_visible=is_visible)


class Chromecast(Thing):
    def __init__(self, name="Chromecast", connected_things=None, is_visible=True):
        super().__init__(name=name, connected_things=connected_things, is_visible=is_visible)

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

        # TODO See what to to with available metadata: Not necessary maybe?


if __name__ == "__main__":
    Env = {
        'plug': PlugSwitch(),
        'light': LightBulb(),
        'TV': LGTV(),
        'speaker': Speaker(),
        'chromecast': Chromecast()
    }

    for k, v in Env.items():
        print(k, v.get_action_space())
