import numpy as np
import gym
from gym import spaces
from sklearn.preprocessing import OneHotEncoder

from simulator.discrete_parameters import discrete_parameters, params_interpreters
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
    'do_nothing': [],
}

baseaction_type_embedder = OneHotEncoder(sparse=False)
baseaction_type_embedder.fit(np.array(list(ACTION_SPACE.keys())).reshape(-1, 1))


def create_embedders(discrete_params):
    def aux(d):
        if isinstance(d, dict):
            return {k: aux(v) for k, v in d.items()}
        elif isinstance(d, list):
            emb = OneHotEncoder(sparse=False)
            emb.fit(np.array(d).reshape(-1, 1))
            return emb
        else:
            raise InterruptedError

    return aux(discrete_params)


params_embedders = create_embedders(discrete_parameters)


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
        if name in discrete_parameters:
            self.children = [
                Params(name=p,
                       type=name,
                       embedder=params_embedders[name][discretization])
                for p in discrete_parameters[name][discretization]]


class ExecAction(NoDescriptionNode):
    def __init__(self, name='exec_action'):
        embedding = np.zeros(len(ACTION_SPACE))
        super().__init__(name, children=[], node_embedding=embedding, node_type='openHAB_action')


class Params(NoDescriptionNode):
    def __init__(self, name, type, embedder):
        self.value = params_interpreters[type](name)
        embedding = embedder.transform(np.array([name]).reshape(-1, 1)).flatten()
        super().__init__(name=name, children=[ExecAction()], node_embedding=embedding, node_type=type + '_params')

    def interpret_params(self):
        return self.value
