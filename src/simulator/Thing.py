from .Items import *


class Thing:
    def __init__(self):
        pass


class TV(Thing):
    def __init__(self):
        super().__init__()


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
                           setHSB=True))
        self.color_temperature = Channel(
            name='color_temperature',
            description='This channel supports adjusting the color temperature from cold (0%) to warm (100%)',
            item=DimmerItem(increase=True, decrease=True, setPercent=True)
        )
        self.alert = Channel(
            name='alert',
            description='This channel supports displaying alerts by flashing the bulb either once or multiple times. Valid values are: NONE, SELECT and LSELECT.',
            item=StringItem()
        )


class Speaker(Thing):
    def __init__(self):
        super(Speaker, self).__init__()


class Store(Thing):
    def __init__(self):
        super(Store, self).__init__()


class Chromecast(Thing):
    def __init__(self):
        super(Chromecast, self).__init__()


class Channel:
    def __init__(self, name, description, item, value=None):
        self.name = name
        self.description = description
        self.item = item
        self.value = value
