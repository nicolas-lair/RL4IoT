from Items import *

from gym import spaces


# TODO Update Volume, Mute, PLayer, Stop for connected things for TV, Chromecast and Speaker
class Thing:
    def __init__(self, name, connected_things, is_visible=True):
        self.initial_value = {'name': name, 'connected_things': connected_things, 'is_visible': is_visible}
        self.name = name
        self.is_visible = is_visible
        self.connected_things = set()

        self.observation_space = None
        self.action_space = None

        if connected_things is not None:
            self.connect_thing(connected_things)
        self.channels = []

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
        Initialize the list of channels of the Thing object
        :return: list of Channel objects
        """
        channels = [x for x in vars(self).values() if isinstance(x, Channel)]
        self.channels = channels
        return channels

    def build_gym_space(self):
        channels = self.channels if len(self.channels) > 0 else self.get_channels()
        channels_name = [chn.name for chn in channels]

        channels_observation_space = [chn.get_observation_space() for chn in channels]
        self.observation_space = spaces.Dict(dict(zip(channels_name, channels_observation_space)))

        channels_action_space = [chn.get_action_space() for chn in channels]
        self.action_space = spaces.Dict(dict(zip(channels_name, channels_action_space)))

    def get_observation_space(self):
        if self.observation_space is None:
            self.build_gym_space()
        return self.observation_space

    def get_action_space(self):
        if self.action_space is None:
            self.build_gym_space()
        return self.action_space

    def _get_state(self):
        state = dict()
        description = dict()
        item_type = dict()
        channels = self.channels if len(self.channels) > 0 else self.get_channels()
        for c in channels:
            state[c.name] = c.get_state()
            description[c.name] = c.description
            item_type[c.name] = c.item.type
        return state, description, item_type

    def get_state(self):
        return self._get_state() if self.is_visible else None

    def do_action(self, channel, action, params=None):
        channel = getattr(self, channel)
        channel.do_action(action, params)

    def reset(self):
        self.__init__(**self.initial_value)


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
        getattr(self.item, action)(*params)


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
            item=DimmerItem(increase=True, decrease=True, setPercent=True)
        )

        # # TODO decide if use or not : maybe not
        # self.alert = Channel(
        #     name='alert',
        #     description='This channel supports displaying alerts by flashing the bulb either once or multiple times. Valid values are: NONE, SELECT and LSELECT.',
        #     item=StringItem()
        # )


class PlugSwitch(Thing):
    """
    https://www.openhab.org/addons/bindings/zwave/thing.html?manufacturer=everspring&file=an180_0_0.html
    """

    def __init__(self, name="plugswitch", connected_things=None, is_visible=True):
        super().__init__(name=name, connected_things=connected_things, is_visible=is_visible)
        self.switch_binary = Channel(name="switch_binary",
                                     description="Switch the power on and off.",
                                     item=SwitchItem(turnOnOff=True),
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

    def add_connected_things(self, thing):
        assert isinstance(thing, Thing), "thing must be an object of class Thing"
        self.connected_things.append(thing)

    def do_action(self, channel, action, params=None):
        assert channel == "switch_binary", f"Switch_binary is the only available channel, {channel} was called instead"
        super().do_action(channel, action, params)
        power_status = self.switch_binary.get_state()
        for thing in self.connected_things:
            thing.update_visibility(power_status)


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

        # TODO link this channel to other volume channels
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
