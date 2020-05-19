import numpy as np
from gym import spaces
from sklearn.preprocessing import OneHotEncoder

from utils import color_list
from TreeView import Node

ACTION_SPACE = {
    'turnOn': spaces.Discrete(2),
    'turnOff': spaces.Discrete(2),
    'increase': spaces.Discrete(2),
    'decrease': spaces.Discrete(2),
    'setPercent': spaces.Box(low=0, high=100, shape=(1,), dtype=float),
    'setHSB': spaces.Box(low=np.array([0, 0, 0]), high=np.array([360, 100, 100]), dtype=float),
    'OpenClose': spaces.Discrete(2),
    'Open': spaces.Discrete(2),
    'Close': spaces.Discrete(2),
    'setLocation': spaces.Box(low=np.array([-90, -180, -1000]), high=np.array([90, 180, 10000], dtype=float)),
    'setValue': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=type),
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
}

baseaction_type_embedder = OneHotEncoder(sparse=False)
baseaction_type_embedder.fit(np.array(list(ACTION_SPACE.keys())).reshape(-1, 1))

color_embedder = OneHotEncoder(sparse=False)
color_embedder.fit(np.array(color_list).reshape(-1, 1))

percent_level = ['very low', 'low', 'average', 'high', 'very high']
percent_embedder = OneHotEncoder(sparse=False)
percent_embedder.fit(np.array(percent_level).reshape(-1, 1))


class OpenHABAction(Node):
    def __init__(self, name):
        embedding = baseaction_type_embedder.transform(np.array([name]).reshape(-1, 1)).flatten()
        super().__init__(name, description=None, children=[ExecAction()], node_embedding=embedding)
        if name == 'setPercent':
            self.children = [Params(p) for p in percent_level]
        elif name == 'setHSB':
            self.children = [Params(c) for c in color_list]
        elif name == 'setValue':
            self.children = None
            raise NotImplementedError


# class BackAction(Node):
#     def __init__(self, name='back_action'):
#         embedding = np.ones(len(ACTION_SPACE))
#         super().__init__(name, description=None, father, children = node_embedding=embedding)

class ExecAction(Node):
    def __init__(self, name='exec_action'):
        embedding = np.zeros(len(ACTION_SPACE))
        super().__init__(name, description=None, node_embedding=embedding)


class Params(Node):
    def __init__(self, name):
        if name in percent_level:
            embedder = percent_embedder
        elif name in color_list:
            embedder = color_embedder
        else:
            raise NotImplementedError
        embedding = embedder.transform(np.array([name]).reshape(-1, 1)).flatten()
        super().__init__(name=name, description=None, children=[ExecAction()], node_embedding=embedding)
