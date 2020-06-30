from collections.abc import Iterable

from simulator.Action import OpenHABAction
from simulator.TreeView import DescriptionNode


class Channel(DescriptionNode):
    def __init__(self, name, description, item, read=True, write=True):

        self.item = item
        super().__init__(name=name, description=description, children=[OpenHABAction(m) for m in self.item.methods])

        # self.initial_value = value
        # if value is not None:
        #     self.item.set_state(value)

        self.read = read
        self.write = write

    def get_state(self):
        return self.item.get_state()

    # def set_state(self, value):
    #     return self.item.set_state(value)

    def get_observation_space(self):
        return self.item.observation_space

    def get_action_space(self):
        return self.item.action_space

    def do_action(self, action, params=None):
        if params is None:
            params = []
        elif isinstance(params, str) or not isinstance(params, Iterable):
            params = [params]
        getattr(self.item, action)(self.item, *params)

    def get_available_actions(self):
        return self.item.methods

    def get_state_change(self):
        pass

    def init(self, init_param):
        self.item.initialize_value(init_param)
