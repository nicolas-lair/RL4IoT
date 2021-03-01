from collections import OrderedDict
from collections.abc import Iterable

import json

from simulator.Action import OpenHABAction
from simulator.TreeView import DescriptionNode


class ChannelState(OrderedDict):
    def __init__(self, state):
        super().__init__(sorted(state.items()))

    def __hash__(self):
        return hash(json.dumps(self, sort_keys=True))


class Channel(DescriptionNode):
    def __init__(self, name, description, item, read=True, write=True, associated_state_description=None,
                 associated_state_change=None):

        self.item = item
        super().__init__(name=name, description=description,
                         children=[OpenHABAction(m, discretization=self.item.discretization[m]) for m in
                                   self.item.methods],
                         node_type='channel')

        # self.initial_value = value
        # if value is not None:
        #     self.item.set_state(value)

        self.read = read
        self.write = write

        if associated_state_description is None:
            self.associated_state_description = []
        elif isinstance(associated_state_description, list):
            self.associated_state_description = associated_state_description
        else:
            self.associated_state_description = [associated_state_description]

        if associated_state_change is None:
            self.associated_state_change = []
        elif isinstance(associated_state_change, list):
            self.associated_state_change = associated_state_change
        else:
            self.associated_state_change = [associated_state_change]

    def get_state(self):
        item_state = self.item.get_state()
        state = {'state': item_state,
                 'description': self.description,
                 'item_type': self.item.type
                 }
        if self.node_embedding is not None: state.update({'embedding': self.node_embedding})
        return ChannelState(state)

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

    def init(self, init_param):
        self.item.initialize_value(init_param)

    def get_state_description_key(self, state):
        keys = [f(*state[self.name]['state']) for f in self.associated_state_description]
        return list(filter(None, keys))

    def get_state_change_key(self, previous_state, next_state):
        keys = [f(*previous_state[self.name]['state'], *next_state[self.name]['state']) for f in
                self.associated_state_change]
        return list(filter(None, keys))
