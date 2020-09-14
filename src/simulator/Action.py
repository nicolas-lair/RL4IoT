import numpy as np
import gym
from gym import spaces
from sklearn.preprocessing import OneHotEncoder

from simulator.utils import color_list, level_to_percent, color_to_hsb, levels_dict, N_LEVELS, find_level_list
from simulator.TreeView import NoDescriptionNode

gym.logger.set_level(40)
ACTION_SPACE = {
    'turnOn': spaces.Discrete(2),
    'turnOff': spaces.Discrete(2),
    'increase': spaces.Discrete(2),
    'decrease': spaces.Discrete(2),
    'setPercent': spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
    'setHSB': spaces.Box(low=np.array([0, 0, 0]), high=np.array([360, 100, 100]), dtype=np.float32),
    'OpenClose': spaces.Discrete(2),
    'Open': spaces.Discrete(2),
    'Close': spaces.Discrete(2),
    'setLocation': spaces.Box(low=np.array([-90, -180, -1000]), high=np.array([90, 180, 10000], dtype=np.float32)),
    'setValue': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
    'setQuantity': None,
    'PlayPause': spaces.Discrete(2),
    'play': spaces.Discrete(2),
    'pause': spaces.Discrete(2),
    'next': spaces.Discrete(2),
    'previous': spaces.Discrete(2),
    'rewind': spaces.Discrete(2),
    'fastforward': spaces.Discrete(2),
    'up': spaces.Discrete(2),
    'down': spaces.Discrete(2),
    'move': spaces.Discrete(2),
    'stop': spaces.Discrete(2),
    'turnOnOff': spaces.Discrete(2),
    'setString': [],
    'do_nothing': []
}

TVchannels_list = [str(i) for i in range(6)]

baseaction_type_embedder = OneHotEncoder(sparse=False)
baseaction_type_embedder.fit(np.array(list(ACTION_SPACE.keys())).reshape(-1, 1))

color_embedder = OneHotEncoder(sparse=False)
color_embedder.fit(np.array(color_list).reshape(-1, 1))

level_one_hot_encoder = OneHotEncoder(sparse=False)
level_one_hot_encoder.fit(np.array(range(N_LEVELS)).reshape(-1, 1))

tv_embedder = OneHotEncoder(sparse=False)
tv_embedder.fit(np.array(TVchannels_list).reshape(-1, 1))


class RootAction(NoDescriptionNode):
    def __init__(self, children, embedding):
        super().__init__(name='root_action', node_type='root', children=children, node_embedding=embedding)


class DoNothing(NoDescriptionNode):
    def __init__(self):
        name = 'do_nothing'
        embedding = baseaction_type_embedder.transform(np.array([name]).reshape(-1, 1)).flatten()
        super().__init__(name=name, node_type='openHAB_action', node_embedding=embedding, children=[])


class OpenHABAction(NoDescriptionNode):
    def __init__(self, name, discretization=None):
        embedding = baseaction_type_embedder.transform(np.array([name]).reshape(-1, 1)).flatten()
        super().__init__(name, children=[ExecAction()], node_embedding=embedding, node_type='openHAB_action')
        if name == 'setPercent':
            try:
                lvl_list = levels_dict[discretization]
            except KeyError:
                raise NotImplementedError
            self.children = [Params(p) for p in lvl_list]

        elif name == 'setHSB':
            if discretization == 'colors':
                self.children = [Params(c) for c in color_list]
            else:
                raise NotImplementedError

        elif name == 'setString':
            if discretization == 'TVchannels':
                self.children = [Params(c) for c in TVchannels_list]
            else:
                raise NotImplementedError
        elif name == 'setValue':
            self.children = None
            raise NotImplementedError


# class BackAction(Node):
#     def __init__(self, name='back_action'):
#         embedding = np.ones(len(ACTION_SPACE))
#         super().__init__(name, description=None, father, children = node_embedding=embedding)

class ExecAction(NoDescriptionNode):
    def __init__(self, name='exec_action'):
        embedding = np.zeros(len(ACTION_SPACE))
        super().__init__(name, children=[], node_embedding=embedding, node_type='openHAB_action')


class Params(NoDescriptionNode):
    def __init__(self, name):
        levels = sum(levels_dict.values(), [])
        n = name
        if name in levels:
            embedder = level_one_hot_encoder
            n = find_level_list(name).index(name)
        elif name in color_list:
            embedder = color_embedder
        elif name in TVchannels_list:
            embedder = tv_embedder
        else:
            raise NotImplementedError
        embedding = embedder.transform(np.array([n]).reshape(-1, 1)).flatten()
        node_type = (name in levels) * 'level' + (name in color_list) * 'color' + (
                    name in TVchannels_list) * 'TVchannels' + '_params'

        super().__init__(name=name, children=[ExecAction()], node_embedding=embedding, node_type=node_type)

        if self.name in levels:
            self.interpreter = level_to_percent
        elif self.name in color_list:
            self.interpreter = color_to_hsb
        elif self.name in TVchannels_list:
            self.interpreter = lambda x: x
        else:
            raise NotImplementedError

    def interpret_params(self):
        return self.interpreter(self.name)
