import numpy as np
import gym

from gym import spaces

from .Thing import PlugSwitch, Speaker, LightBulb, LGTV, Chromecast


class Environment(gym.Env):
    def __init__(self):
        super(Environment, self).__init__()
        self.plug = PlugSwitch()
        self.lightbulb = LightBulb()

        self.observation_space = spaces.Dict({
            "plug": self.plug.get_observation_space(),
            "lightbulb": self.lightbulb.get_observation_space(),
        })

        self.action_space = spaces.Dict({
            "plug": self.plug.get_action_space(),
            "lightbulb": self.lightbulb.get_action_space()
        })

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass
