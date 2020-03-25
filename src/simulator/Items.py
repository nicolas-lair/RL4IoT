class AbstractItem:
    def __init__(self, type):
        self.type = type
        self.method = None

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
    def __init__(self):
        super(ColorItem, self).__init__("color")
        self.hue = 0
        self.saturation = 0
        self.brightness = 0
        self.state = None
        self.method = None

    def get_state(self):
        return [self.hue, self.saturation, self.brightness]

    def turnOn(self):
        self.brightness = self.brightness if self.brightness != 0 else 100

    def turnOff(self):
        self.brightness = 0

    def increase(self):
        self.brightness = min(self.brightness + 10, 100)

    def decrease(self):
        self.brightness = max(self.brightness - 10, 0)

    def setPercent(self, percent):
        assertion_test = isinstance(percent, int) and (0 <= percent <= 100)
        error_message = "Percent should be an int between 0 and100"
        self.setAttribute(['percent'], [percent], assertion_test, error_message)

    def setHSB(self, h, s, b):
        assertion_test = isinstance(h, int) and isinstance(s, int) and isinstance(b, int) and (0 <= h <= 360) and (
                0 <= s <= 100) and (0 <= b <= 100)
        error_message = "h, s, b should be positive int, h <=360, s <=100, b<=100"
        self.setAttribute(['hue', 'saturation', 'brightness'], [h, s, b], assertion_test, error_message)


class ContactItem(AbstractItem):
    def __init__(self):
        super().__init__("contact")
        self.openclosed = False

    def get_state(self):
        return [self.openclosed]

    def OpenClose(self):
        self.openclosed = 1 - self.openclosed


class DimmerItem(AbstractItem):
    def __init__(self):
        super().__init__('dimmer')
        self.percent = 0

    def get_state(self):
        return [self.percent]

    def turnOn(self):
        self.percent = self.percent if self.percent != 0 else 100

    def turnOff(self):
        self.percent = 0

    def increase(self):
        self.percent = min(self.percent + 10, 100)

    def decrease(self):
        self.percent = max(self.percent - 10, 0)

    def setPercent(self, percent):
        assertion_test = isinstance(percent, int) and (0 <= percent <= 100)
        error_message = "Percent should be an int between 0 and 100"
        self.setAttribute(['percent'], [percent], assertion_test, error_message)


class LocationItem(AbstractItem):
    def __init__(self):
        super().__init__('location')
        self.latitude = None
        self.longitude = None
        self.altitude = None

    def get_state(self):
        return [self.latitude, self.longitude, self.altitude]

    # TODO
    def setLocation(self, lat, lon, alt):
        assertion_test = isinstance(lat, float) and isinstance(lon, float) and isinstance(alt, float) and (
                -90 <= lat <= 90) and (-180 <= lon <= 180) and (-500 <= alt <= 4000)
        error_message = "h, s, b should be positive int, h <=360, s <=100, b<=100"
        self.setAttribute(['latitude', 'longitude', 'altitude'], [alt, lon, alt], assertion_test, error_message)


class NumberItem(AbstractItem):

    def __init__(self):
        super().__init__('number')
        self.value = None

    def get_state(self):
        return [self.value]

    def setValue(self, value):
        assertion_test = isinstance(value, int)
        error_message = "Percent should be an int"
        self.setAttribute(['value'], [value], assertion_test, error_message)

    # TODO
    def setQuantity(self, quantity):
        pass


class PlayerItem(AbstractItem):

    def __init__(self):
        super().__init__('player')
        self.playpause = False

    def get_state(self):
        return [self.playpause]

    def PlayPause(self):
        self.playpause = bool(1 - self.playpause)

    def next(self):
        pass

    def previous(self):
        pass

    def rewind(self):
        pass

    def fastforward(self):
        pass


class RollerShutterItem(AbstractItem):

    def __init__(self):
        super().__init__('rollershutter')
        self.percent = 0

    def get_state(self):
        return [self.percent]

    def up(self):
        pass

    def down(self):
        pass

    def stop(self):
        pass

    def move(self):
        pass

    def setPercent(self, percent):
        assertion_test = isinstance(percent, int) and (0 <= percent <= 100)
        error_message = "Percent should be an int between 0 and 100"
        self.setAttribute(['percent'], [percent], assertion_test, error_message)


class StringItem(AbstractItem):

    def __init__(self):
        super().__init__('string')
        self.string = ""

    def get_state(self):
        return [self.string]

    def setString(self, string):
        assertion_test = isinstance(string, str)
        error_message = "String should be of type string"
        self.setAttribute(['string'], [string], assertion_test, error_message)


class SwitchItem(AbstractItem):

    def __init__(self):
        super().__init__('switch')
        self.onoff = False

    def get_state(self):
        return [self.onoff]

    def OnOff(self):
        self.onoff = 1 - self.onoff
