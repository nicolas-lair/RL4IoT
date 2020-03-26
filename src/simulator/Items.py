class MethodUnavailableError(Exception):
    pass


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


class AbstractItem:
    def __init__(self, type, methods):
        self.type = type
        self.methods = methods
        del self.methods['self']

    def get_state(self):
        pass

    def setAttribute(self, attribute, value, valuetest, error_message, success_message=""):
        try:
            assert valuetest
            for a, v in zip(attribute, value):
                setattr(self, a, v)
            return 1, success_message
        except AssertionError:
            return 0, error_message
        except Exception as e:
            return 0, e


class ColorItem(AbstractItem):
    def __init__(self, turnOn=False, turnOff=False, increase=False, decrease=False, setPercent=False, setHSB=False):
        super().__init__(type="color", methods=locals())
        self.hue = 0
        self.saturation = 0
        self.brightness = 0

    def get_state(self):
        return [self.hue, self.saturation, self.brightness]

    @check_method_availability
    def turnOn(self):
        self.brightness = self.brightness if self.brightness != 0 else 100

    @check_method_availability
    def turnOff(self):
        self.brightness = 0

    @check_method_availability
    def increase(self):
        self.brightness = min(self.brightness + 10, 100)

    @check_method_availability
    def decrease(self):
        self.brightness = max(self.brightness - 10, 0)

    @check_method_availability
    def setPercent(self, percent):
        assertion_test = isinstance(percent, int) and (0 <= percent <= 100)
        error_message = "Percent should be an int between 0 and100"
        self.setAttribute(['percent'], [percent], assertion_test, error_message)

    @check_method_availability
    def setHSB(self, h, s, b):
        assertion_test = isinstance(h, int) and isinstance(s, int) and isinstance(b, int) and (0 <= h <= 360) and (
                0 <= s <= 100) and (0 <= b <= 100)
        error_message = "h, s, b should be positive int, h <=360, s <=100, b<=100"
        self.setAttribute(['hue', 'saturation', 'brightness'], [h, s, b], assertion_test, error_message)


class ContactItem(AbstractItem):
    def __init__(self, OpenClose=False):
        super().__init__(type="contact", methods=locals())
        self.state = 0

    def get_state(self):
        return [self.state]

    @check_method_availability
    def OpenClose(self):
        self.state = 1 - self.state


class DimmerItem(AbstractItem):
    def __init__(self, turnOn=False, turnOff=False, increase=False, decrease=False, setPercent=False):
        super().__init__(type="dimmer", methods=locals())
        self.percent = 0

    def get_state(self):
        return [self.percent]

    @check_method_availability
    def turnOn(self):
        self.percent = self.percent if self.percent != 0 else 100

    @check_method_availability
    def turnOff(self):
        self.percent = 0

    @check_method_availability
    def increase(self):
        self.percent = min(self.percent + 10, 100)

    @check_method_availability
    def decrease(self):
        self.percent = max(self.percent - 10, 0)

    @check_method_availability
    def setPercent(self, percent):
        assertion_test = isinstance(percent, int) and (0 <= percent <= 100)
        error_message = "Percent should be an int between 0 and 100"
        self.setAttribute(['percent'], [percent], assertion_test, error_message)


class LocationItem(AbstractItem):
    def __init__(self, setLocation=False):
        super().__init__(type="location", methods=locals())
        self.latitude = None
        self.longitude = None
        self.altitude = None

    def get_state(self):
        return [self.latitude, self.longitude, self.altitude]

    @check_method_availability
    def setLocation(self, lat, lon, alt):
        assertion_test = isinstance(lat, float) and isinstance(lon, float) and isinstance(alt, float) and (
                -90 <= lat <= 90) and (-180 <= lon <= 180) and (-500 <= alt <= 4000)
        error_message = "h, s, b should be positive int, h <=360, s <=100, b<=100"
        self.setAttribute(['latitude', 'longitude', 'altitude'], [alt, lon, alt], assertion_test, error_message)


class NumberItem(AbstractItem):
    def __init__(self, setValue=False, setQuantity=False):
        super().__init__(type="numnber", methods=locals())
        self.value = None

    def get_state(self):
        return [self.value]

    @check_method_availability
    def setValue(self, value):
        assertion_test = isinstance(value, int)
        error_message = "Percent should be an int"
        self.setAttribute(['value'], [value], assertion_test, error_message)

    # TODO
    @check_method_availability
    def setQuantity(self, quantity):
        pass


class PlayerItem(AbstractItem):
    def __init__(self, PlayPause=False, next=False, previous=False, rewind=False, fastforward=False):
        super().__init__(type="number", methods=locals())
        self.playpause = False

    def get_state(self):
        return [self.playpause]

    @check_method_availability
    def PlayPause(self):
        self.playpause = bool(1 - self.playpause)

    @check_method_availability
    def next(self):
        pass

    @check_method_availability
    def previous(self):
        pass

    @check_method_availability
    def rewind(self):
        pass

    @check_method_availability
    def fastforward(self):
        pass


class RollerShutterItem(AbstractItem):

    def __init__(self, up=False, down=False, stop=False, move=False, setPercent=False):
        super().__init__(type="rollershutter", methods=locals())
        self.percent = 0

    def get_state(self):
        return [self.percent]

    @check_method_availability
    def up(self):
        pass

    @check_method_availability
    def down(self):
        pass

    @check_method_availability
    def stop(self):
        pass

    @check_method_availability
    def move(self):
        pass

    @check_method_availability
    def setPercent(self, percent):
        assertion_test = isinstance(percent, int) and (0 <= percent <= 100)
        error_message = "Percent should be an int between 0 and 100"
        self.setAttribute(['percent'], [percent], assertion_test, error_message)


class StringItem(AbstractItem):
    def __init__(self, setString=False):
        super().__init__(type="string", methods=locals())
        self.string = ""

    def get_state(self):
        return [self.string]

    @check_method_availability
    def setString(self, string):
        assertion_test = isinstance(string, str)
        error_message = "String should be of type string"
        self.setAttribute(['string'], [string], assertion_test, error_message)


class SwitchItem(AbstractItem):

    def __init__(self, OnOff=False):
        super().__init__(type="switch", methods=locals())
        self.onoff = 0

    def get_state(self):
        return [self.onoff]

    @check_method_availability
    def OnOff(self):
        self.onoff = 1 - self.onoff
