from .Items import *


class Thing:
    def __init__(self):
        pass

    # TODO
    def get_channels(self):
        pass


class Channel:
    def __init__(self, name, description, item, value=None, read=True, write=True):
        self.name = name
        self.description = description

        self.item = item
        self.item.set_state(value)

        self.read = read
        self.write = write


class LGTV(Thing):
    """
    https://www.openhab.org/addons/bindings/lgwebos/
    """

    def __init__(self, power=1, mute=0):
        super().__init__()
        self.power = Channel(
            name='power',
            description="Current power setting. TV can only be powered off, not on.",
            item=SwitchItem(Off=True),
            value=power
        )

        self.mute = Channel(
            name="mute",
            description="Current mute setting.",
            item=SwitchItem(OnOff=True),
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
            item=SwitchItem(Off=True),
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


class LightBulb(Thing):
    """
    Thing type 0210 (https://www.openhab.org/addons/bindings/hue/)
    """

    def __init__(self):
        super().__init__()
        self.color = Channel(
            name='color',
            description="This channel supports full color control with hue, saturation and brightness values",
            item=ColorItem(turnOn=True, turnOff=True, increase=True, decrease=True, setPercent=True,
                           setHSB=True),
            value=(0, 0, 0)
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

    def __init__(self):
        super(PlugSwitch, self).__init__()
        self.switch_binary = Channel(name="switch_binary",
                                     description="Switch the power on and off.",
                                     item=SwitchItem(OnOff=True))

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



class Speaker(Thing):
    def __init__(self):
        super(Speaker, self).__init__()


class Store(Thing):
    def __init__(self):
        super(Store, self).__init__()


class Chromecast(Thing):
    def __init__(self):
        super(Chromecast, self).__init__()
