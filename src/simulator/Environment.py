import numpy as np
import gym

from gym import spaces

from Thing import Thing, PlugSwitch, Speaker, LightBulb, LGTV, Chromecast


class IoTEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Define object in the environment
        self.plug = PlugSwitch()
        self.lightbulb = LightBulb()

        # Connect objects and set
        # self.plug.connect_thing(self.lightbulb)

        # Compute observation_space
        self.observation_space = spaces.Dict({
            "plug": self.plug.get_observation_space(),
            "lightbulb": self.lightbulb.get_observation_space(),
        })

        # Compute action_space
        self.action_space = spaces.Dict({
            "plug": self.plug.get_action_space(),
            "lightbulb": self.lightbulb.get_action_space()
        })

        self._things_lookup_table = self._build_things_lookup_table()
        self.previous_state = None
        self.state = self.build_state()

    def get_thing_list(self):
        return self._things_lookup_table.values()

    def get_thing(self, name):
        return self._things_lookup_table[name]

    def _build_things_lookup_table(self):
        return {x.name: x for x in vars(self).values() if isinstance(x, Thing)}

    def step(self, action):
        """
        action: {
            "thing": X,
            "channel": Y,
            "action": Z,
            "params: P
        }
        :type action: dict
        :param action:
        :return:
        """
        thing = self.get_thing(action["thing"])
        thing.do_action(action["channel"], action["action"], action["params"])

        self.previous_state = self.state
        self.state = self.build_state()

        reward = None
        done = None
        info = None
        return self.state, reward, done, info

    def build_state(self, thing=None):
        """

        :param thing: None means build state for all Things in the environment
        :return:
        """
        if thing is None:
            thing_list = self.get_thing_list()
        elif isinstance(thing, Thing):
            thing_list = [thing]
        elif isinstance(thing, list):
            thing_list = thing
        else:
            raise TypeError('thing should be None, a thing object or a list of Thing object')

        state = dict()
        for thing in thing_list:
            state[thing.name] = thing.get_state()
        return state

    def reset(self):
        for thing in self.get_thing_list():
            thing.reset()
        state = self.build_state()
        return state

    def render(self, mode='human'):
        raise NotImplementedError


if __name__ == "__main__":
    env = IoTEnv()
    initial_state = env.reset()
