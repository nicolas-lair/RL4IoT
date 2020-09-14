from abc import ABC, abstractmethod
from collections.abc import Iterable
import random

import numpy as np
from gym import spaces

from simulator.Action import ACTION_SPACE


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
        if func.__name__ in self.methods:
            func(*args, **kwargs)
        else:
            raise MethodUnavailableError

    return wrapper_check


class AbstractItem(ABC):
    def __init__(self, type, methods, discretization):
        assert type in ITEM_TYPE, 'Wrong item type'
        self.type = type
        # Filter to keep only valid and active methods for the items
        action_space = ACTION_SPACE.keys()
        self.methods = [meth for meth, bool_flag in methods.items() if (meth in action_space and bool_flag)]
        self.discretization = {m: None for m in self.methods}
        if discretization:
            self.discretization.update(discretization)

        self.observation_space = None
        # self.action_space = spaces.Dict({k: ACTION_SPACE[k] for k in self.methods})
        self.attr_error_message = None

    @abstractmethod
    def get_state(self):
        raise NotImplementedError

    @abstractmethod
    def set_state(self, value):
        raise NotImplementedError

    def set_attribute(self, attribute, value):
        # if self.type == 'string':
        #     raise NotImplementedError
        # assert self.observation_space.contains(np.array(value)), self.attr_error_message
        if isinstance(attribute, list):
            for a, v in zip(attribute, value):
                setattr(self, a, v)
            return 1
        else:
            setattr(self, attribute, value)

    def get_available_actions(self):
        return self.methods


class ColorItem(AbstractItem):
    def __init__(self, turnOn=False, turnOff=False, increase=False, decrease=False, setPercent=False, setHSB=False,
                 discretization=None):
        super().__init__(type="color", methods=locals(), discretization=discretization)
        self.hue = None
        self.saturation = None
        self.brightness = None

        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([360, 100, 100]), dtype=float)
        self.attr_error_message = "h, s, b should be positive int, h <=360, s <=100, b<=100"

    def get_state(self):
        return [self.hue, self.saturation, self.brightness]

    def set_state(self, value):
        self.set_attribute(['hue', 'saturation', 'brightness'], value)

    def initialize_value(self, init):
        if init == 'default':
            self.hue = 0
            self.saturation = 0
            self.brightness = 0
        elif init == 'random':
            self.hue = random.randint(0, 360)
            self.saturation = random.randint(0, 100)
            onoff = random.randint(0, 1)
            self.brightness = onoff * random.randint(1, 100)
        elif isinstance(init, Iterable):
            self.set_state(init)
        else:
            raise NotImplementedError

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
        self.set_state((self.hue, self.saturation, percent))

    @check_method_availability
    def setHSB(self, h, s, b):
        self.set_state((h, s, b))


class ContactItem(AbstractItem):
    def __init__(self, OpenClose=False, Open=False, Close=False, discretization=None):
        super().__init__(type="contact", methods=locals(), discretization=discretization)
        self.state = None
        self.observation_space = spaces.Discrete(2)
        self.attr_error_message = "onoff should be 0 or 1 (boolean)"

    def get_state(self):
        return [self.state]

    def set_state(self, value):
        self.set_attribute('state', value)

    def initialize_value(self, init):
        if init == 'default':
            self.state = 0
        elif init == 'random':
            self.state = random.randint(0, 1)
        elif init in [0, 1, True, False]:
            self.set_state(init)
        else:
            raise NotImplementedError

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
    def __init__(self, turnOn=False, turnOff=False, increase=False, decrease=False, setPercent=False,
                 discretization=None):
        super().__init__(type="dimmer", methods=locals(), discretization=discretization)
        self.percent = None
        self.observation_space = spaces.Box(low=0, high=100, shape=(1,), dtype=float)
        self.attr_error_message = "Percent should be an int between 0 and 100"

    def get_state(self):
        return [self.percent]

    def set_state(self, value):
        self.set_attribute(['percent'], [value])

    def initialize_value(self, init):
        if init == 'default':
            self.percent = 0
        elif init == 'random':
            self.percent = random.randint(0, 100)
        elif init in range(0, 101):
            self.set_state(init)
        else:
            raise NotImplementedError

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
    def __init__(self, setLocation=False, discretization=None):
        super().__init__(type="location", methods=locals(), discretization=discretization)
        self.latitude = None
        self.longitude = None
        self.altitude = None
        self.observation_space = spaces.Box(low=np.array([-90, -180, -1000]),
                                            high=np.array([90, 180, 10000], dtype=float))
        self.attr_error_message = "longitude should be a float between -90 and 90, " \
                                  "latitude should be float between -180 and 180" \
                                  "and altitude should float between -1000 and 10000"

    def initialize_value(self, init):
        if init == 'default':
            self.latitude = 0
            self.longitude = 0
            self.altitude = 0
        elif init == 'random':
            self.latitude = random.uniform(-90, 90)
            self.longitude = random.uniform(-180, 180)
            self.altitude = random.uniform(-1000, 10000)
        elif isinstance(init, Iterable):
            self.set_state(init)
        else:
            raise NotImplementedError

    def get_state(self):
        return [self.latitude, self.longitude, self.altitude]

    def set_state(self, value):
        self.set_attribute(['latitude', 'longitude', 'altitude'], value)

    @check_method_availability
    def setLocation(self, lat, lon, alt):
        self.set_state((lat, lon, alt))


class NumberItem(AbstractItem):
    def __init__(self, setValue=False, setQuantity=False, type=float, discretization=None):
        super().__init__(type="number", methods=locals(), discretization=discretization)
        self.value = None

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=type)
        self.attr_error_message = "value should a int or float"

    def initialize_value(self, init):
        if init == 'default':
            self.value = 0
        elif init == 'random':
            self.value = random.uniform(-10000000, 10000000)
        elif isinstance(init, float):
            self.set_state(init)
        else:
            raise NotImplementedError

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


class PlayerItem(AbstractItem):  # TODO Check user_state of players
    def __init__(self, PlayPause=False, next=False, previous=False, rewind=False, fastforward=False,
                 discretization=None):
        super().__init__(type="player", methods=locals(), discretization=discretization)
        self.playpause = None
        self.observation_space = spaces.Discrete(2)
        self.attr_error_message = "playpause should be 0 or 1 (boolean)"

    def initialize_value(self, init):
        if init == 'default':
            self.playpause = False
        elif init == 'random':
            self.playpause = bool(random.randint(0, 1))
        elif init in [0, 1, True, False]:
            self.set_state(init)
        else:
            raise NotImplementedError

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

    def __init__(self, up=False, down=False, stop=False, move=False, setPercent=False, discretization=None):
        super().__init__(type="rollershutter", methods=locals(), discretization=discretization)
        self.percent = None
        self.observation_space = spaces.Box(low=0, high=100, shape=(1,), dtype=float)
        self.attr_error_message = "Percent should be an int between 0 and 100"

    def initialize_value(self, init):
        if init == 'default':
            self.percent = 0
        elif init == 'random':
            self.percent = random.randint(0, 100)
        elif init in range(0, 101):
            self.set_state(init)
        else:
            raise NotImplementedError

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
    def __init__(self, setString=False, init='default', discretization=None):
        super().__init__(type="string", methods=locals(), discretization=discretization)
        self.string = None

        self.observation_space = None
        self.action_space = None
        self.attr_error_message = "String should be of type string"

    def initialize_value(self, init):
        if init == 'default':
            self.string = ""
        elif init == 'random':
            self.string = random.choice([str(i) for i in range(5)]) #TODO Fix Hack
        elif isinstance(init, str):
            self.set_state(init)
        else:
            raise NotImplementedError

    def get_state(self):
        return [self.string]

    def set_state(self, value):
        assert isinstance(value, str), self.attr_error_message
        self.set_attribute(['string'], [value])

    @check_method_availability
    def setString(self, string):
        self.set_state(string)


class SwitchItem(AbstractItem):

    def __init__(self, turnOnOff=False, turnOn=False, turnOff=False, init='default', discretization=None):
        super().__init__(type="switch", methods=locals(), discretization=discretization)
        self.onoff = None
        self.observation_space = spaces.Discrete(2)
        self.attr_error_message = "onoff should be 0 or 1 (boolean)"

    def initialize_value(self, init):
        if init == 'default':
            self.onoff = 0
        elif init == 'random':
            self.onoff = random.randint(0, 1)
        elif init in [0, 1, True, False]:
            self.set_state(init)
        else:
            raise NotImplementedError

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
