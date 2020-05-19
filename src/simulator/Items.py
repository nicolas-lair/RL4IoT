from abc import ABC, abstractmethod

import numpy as np
from gym import spaces

from Action import ACTION_SPACE


class MethodUnavailableError(Exception):
    pass


INCREASE_DECREASE_STEP = 10

ITEM_TYPE = ['color', 'contact', 'dimmer', 'location', 'number', 'player', 'rollershutter', 'string', 'switch']


def check_method_availability(func):
    """
    Decorator to check if a method is available for an instance of item
    :param func:
    :return:
    """

    def wrapper_check(self, *args, **kwargs):
        if self.methods[func.__name__]:
            func(*args, **kwargs)
        else:
            raise MethodUnavailableError

    return wrapper_check


class AbstractItem(ABC):
    def __init__(self, type, methods):
        assert type in ITEM_TYPE, 'Wrong item type'
        self.type = type
        # Filter to keep only valid and active methods for the items
        self.methods = [m for m, b in methods.items() if (b and m in ACTION_SPACE.keys())]

        self.observation_space = None
        self.action_space = spaces.Dict({k: ACTION_SPACE[k] for k in self.methods})
        self.attr_error_message = None

    @abstractmethod
    def get_state(self):
        raise NotImplementedError

    @abstractmethod
    def set_state(self, value):
        raise NotImplementedError

    def set_attribute(self, attribute, value):
        if self.type == 'string':
            raise NotImplementedError
        assert self.observation_space.contains(value), self.attr_error_message
        if isinstance(attribute, list):
            for a, v in zip(attribute, value):
                setattr(self, a, v)
            return 1
        else:
            setattr(self, attribute, value)

    def get_available_actions(self):
        return self.methods


class ColorItem(AbstractItem):
    def __init__(self, turnOn=False, turnOff=False, increase=False, decrease=False, setPercent=False, setHSB=False):
        super().__init__(type="color", methods=locals())
        self.hue = 0
        self.saturation = 0
        self.brightness = 0

        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([360, 100, 100]), dtype=float)

        self.attr_error_message = "h, s, b should be positive int, h <=360, s <=100, b<=100"

    def get_state(self):
        return [self.hue, self.saturation, self.brightness]

    def set_state(self, value):
        self.set_attribute(['hue', 'saturation', 'brightness'], value)

    @check_method_availability
    def turnOn(self):
        self.brightness = self.brightness if self.brightness != 0 else 100

    @check_method_availability
    def turnOff(self):
        self.brightness = 0

    @check_method_availability
    def increase(self):
        self.brightness = min(self.brightness + INCREASE_DECREASE_STEP, 100)

    @check_method_availability
    def decrease(self):
        self.brightness = max(self.brightness - INCREASE_DECREASE_STEP, 0)

    @check_method_availability
    def setPercent(self, percent):
        self.setHSB([self.hue, percent, self.saturation])

    @check_method_availability
    def setHSB(self, h, s, b):
        self.set_state((h, s, b))


class ContactItem(AbstractItem):
    def __init__(self, OpenClose=False, Open=False, Close=False):
        super().__init__(type="contact", methods=locals())
        self.state = 0

        self.observation_space = spaces.Discrete(2)
        self.attr_error_message = "onoff should be 0 or 1 (boolean)"

    def get_state(self):
        return [self.state]

    def set_state(self, value):
        self.set_attribute('onoff', value)

    @check_method_availability
    def OpenClose(self):
        self.state = 1 - self.state

    @check_method_availability
    def Open(self):
        self.state = 0  # No contact

    @check_method_availability
    def Close(self):
        self.state = 1  # Contact


class DimmerItem(AbstractItem):
    def __init__(self, turnOn=False, turnOff=False, increase=False, decrease=False, setPercent=False):
        super().__init__(type="dimmer", methods=locals())
        self.percent = 0

        self.observation_space = spaces.Box(low=0, high=100, shape=(1,), dtype=float)
        self.attr_error_message = "Percent should be an int between 0 and 100"

    def get_state(self):
        return [self.percent]

    def set_state(self, value):
        self.set_attribute(['percent'], [value])

    @check_method_availability
    def turnOn(self):
        self.percent = self.percent if self.percent != 0 else 100

    @check_method_availability
    def turnOff(self):
        self.percent = 0

    @check_method_availability
    def increase(self):
        self.percent = min(self.percent + INCREASE_DECREASE_STEP, 100)

    @check_method_availability
    def decrease(self):
        self.percent = max(self.percent - INCREASE_DECREASE_STEP, 0)

    @check_method_availability
    def setPercent(self, percent):
        self.set_state(percent)


class LocationItem(AbstractItem):
    def __init__(self, setLocation=False):
        super().__init__(type="location", methods=locals())
        self.latitude = None
        self.longitude = None
        self.altitude = None

        self.observation_space = spaces.Box(low=np.array([-90, -180, -1000]),
                                            high=np.array([90, 180, 10000], dtype=float))
        self.attr_error_message = "longitude should be a float between -90 and 90, " \
                                  "latitude should be float between -180 and 180" \
                                  "and altitude should float between -1000 and 10000"

    def get_state(self):
        return [self.latitude, self.longitude, self.altitude]

    def set_state(self, value):
        self.set_attribute(['latitude', 'longitude', 'altitude'], value)

    @check_method_availability
    def setLocation(self, lat, lon, alt):
        self.set_state((lat, lon, alt))


class NumberItem(AbstractItem):
    def __init__(self, setValue=False, setQuantity=False, type=float):
        methods = locals().copy()
        del methods['type']
        super().__init__(type="number", methods=methods)
        self.value = 0

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=type)
        self.attr_error_message = "value should a int or float"

    def get_state(self):
        return [self.value]

    def set_state(self, value):
        self.set_attribute(['value'], [value])

    @check_method_availability
    def setValue(self, value):
        self.set_state(value)

    # TODO
    @check_method_availability
    def setQuantity(self, quantity):
        raise NotImplementedError


class PlayerItem(AbstractItem):  # TODO Check state of players
    def __init__(self, PlayPause=False, next=False, previous=False, rewind=False, fastforward=False):
        super().__init__(type="player", methods=locals())
        self.playpause = False

        self.observation_space = spaces.Discrete(2)
        self.attr_error_message = "playpause should be 0 or 1 (boolean)"

    def get_state(self):
        return [self.playpause]

    def set_state(self, value):
        self.set_attribute('playpause', value)

    @check_method_availability
    def PlayPause(self):
        self.playpause = bool(1 - self.playpause)

    @check_method_availability
    def play(self):
        self.playpause = 1

    @check_method_availability
    def pause(self):
        self.playpause = 0

    @check_method_availability
    def next(self):
        raise NotImplementedError

    @check_method_availability
    def previous(self):
        raise NotImplementedError

    @check_method_availability
    def rewind(self):
        raise NotImplementedError

    @check_method_availability
    def fastforward(self):
        raise NotImplementedError


class RollerShutterItem(AbstractItem):

    def __init__(self, up=False, down=False, stop=False, move=False, setPercent=False):
        super().__init__(type="rollershutter", methods=locals())
        self.percent = 0

        self.observation_space = spaces.Box(low=0, high=100, shape=(1,), dtype=float)
        self.attr_error_message = "Percent should be an int between 0 and 100"

    def get_state(self):
        return [self.percent]

    def set_state(self, value):
        self.set_attribute(['percent'], [value])

    @check_method_availability
    def up(self):
        raise NotImplementedError

    @check_method_availability
    def down(self):
        raise NotImplementedError

    @check_method_availability
    def stop(self):
        raise NotImplementedError

    @check_method_availability
    def move(self):
        raise NotImplementedError

    @check_method_availability
    def setPercent(self, percent):
        self.set_state(percent)


class StringItem(AbstractItem):
    def __init__(self, setString=False):
        super().__init__(type="string", methods=locals())
        self.string = ""

        self.observation_space = None
        self.action_space = None
        self.attr_error_message = "String should be of type string"

    def get_state(self):
        return [self.string]

    def set_state(self, value):
        assert isinstance(value, str), self.attr_error_message
        self.set_attribute(['string'], [value])

    @check_method_availability
    def setString(self, string):
        self.set_state(string)


class SwitchItem(AbstractItem):

    def __init__(self, turnOnOff=False, turnOn=False, turnOff=False):
        super().__init__(type="switch", methods=locals())
        self.onoff = 0

        self.observation_space = spaces.Discrete(2)
        self.attr_error_message = "onoff should be 0 or 1 (boolean)"

    def get_state(self):
        return [self.onoff]

    def set_state(self, value):
        self.set_attribute('onoff', value)

    @check_method_availability
    def turnOnOff(self):
        self.onoff = 1 - self.onoff

    @check_method_availability
    def turnOn(self):
        self.onoff = 1

    @check_method_availability
    def turnOff(self):
        self.onoff = 0
