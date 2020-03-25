class Thing:
    def __init__(self):
        pass


class AbstractChannel:
    def __init__(self, name, description, type, value=None, read=False, write=False):
        self.name = name
        self.description= description
        self.type = type
        self.value=value
        self.read = read
        self.write=write
