from itertools import chain

import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch

import gym

from Items import ITEM_TYPE
from Thing import Thing, PlugSwitch, LightBulb, LGTV, Speaker, Chromecast
from Channel import Channel
from description_embedder import Description_embedder
from Action import ExecAction, OpenHABAction, Params
from TreeView import Node

obj_dict = {
    'plug': PlugSwitch,
    'lightbulb': LightBulb,
    'tv': LGTV,
    'speaker': Speaker,
    'chromecast': Chromecast
}


class IoTEnv(gym.Env):
    def __init__(self, obj_params):
        super().__init__()
        # Define object in the environment

        self._things_lookup_table = {}
        for key, (obj_type, obj_params) in obj_params.items():
            self._things_lookup_table[key] = obj_dict[obj_type](name=key, **obj_params)

        # self.plug = PlugSwitch(obj_params['plug1'])
        # self.lightbulb = LightBulb(obj_params['lightbulb1'])

        # Connect objects and set
        # self.plug.connect_thing(self.lightbulb)

        # Compute observation_space

        self.observation_space = gym.spaces.Dict(
            {k: v.get_observation_space() for k, v in self._things_lookup_table.items()}
        )

        self.action_space = gym.spaces.Dict(
            {k: v.get_action_space() for k, v in self._things_lookup_table.items()}
        )

        # self.observation_space = gym.spaces.Dict({
        #     "plug": self.plug.get_observation_space(),
        #     "lightbulb": self.lightbulb.get_observation_space(),
        # })
        #
        # # Compute action_space
        # self.action_space = gym.spaces.Dict({
        #     "plug": self.plug.get_action_space(),
        #     "lightbulb": self.lightbulb.get_action_space()
        # })

        # self._things_lookup_table = self._build_things_lookup_table()
        self.previous_user_state = None
        self.user_state = self.build_state()

    def get_thing_list(self):
        return list(self._things_lookup_table.values())

    def get_thing(self, name):
        return self._things_lookup_table[name]

    # def _build_things_lookup_table(self):
    #     return {x.name: x for x in vars(self).values() if isinstance(x, Thing)}

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

        self.previous_user_state = self.user_state
        self.user_state = self.build_state()

        reward = None
        done = None
        info = None
        return self.user_state, reward, done, info

    def build_state(self, thing=None):
        """

        :param thing: None means build user_state for all Things in the environment
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
        thing_list = self.get_thing_list()
        for thing in thing_list:
            thing.reset()
        self.user_state = self.build_state()
        self.previous_user_state = self.user_state
        return self.user_state

    def render(self, mode='human'):
        raise NotImplementedError


class IoTEnv4ML(gym.Wrapper):
    def __init__(self, params, env_class=IoTEnv):
        super().__init__(env=env_class(params['thing_params']))
        self.description_embedder = Description_embedder(**params['description_embedder_params'])
        self.item_type_embedder = OneHotEncoder(sparse=False)
        self.item_type_embedder.fit(np.array(ITEM_TYPE).reshape(-1, 1))
        self.running_action = {"thing": None, 'channel': None, 'action': None, 'params': None}

        self.state_embedding_size = params['state_encoding_size']
        self.channel_embedding_size = params['description_embedder_params']['dimension'] + len(ITEM_TYPE) + params[
            'state_encoding_size']

        self.state = None
        self.previous_state = None
        self.available_actions = None
        self.previous_available_actions = None

        # # Compute node embedding
        # things_list = self.get_thing_list()
        # description_node_iterator = chain.from_iterable([things_list] + [t.get_channels() for t in things_list])
        # for node in description_node_iterator:
        #     node.embed_node_description(self.description_embedder.get_description_embedding)
        # print(1)

    def preprocess_raw_observation(self, observation, pytorch=True):
        new_obs = dict()
        for thing_name, thing in observation.items():
            thing_obs = dict()
            for channel_name, channel in thing.items():
                if channel['item_type'] == 'goal_string':
                    raise NotImplementedError
                try:
                    description_embedding = channel['embedding']
                except KeyError:
                    description_embedding = self.description_embedder.get_description_embedding(channel['description'])
                item_embedding = self.item_type_embedder.transform(
                    np.array(channel['item_type']).reshape(-1, 1)).flatten()
                state_embedding = np.zeros(self.state_embedding_size)
                state_embedding[:len(channel['state'])] = channel['state']
                channel_embedding = np.concatenate([description_embedding, item_embedding, state_embedding])
                if pytorch:
                    channel_embedding = torch.tensor(channel_embedding)
                thing_obs[channel_name] = channel_embedding
            new_obs[thing_name] = thing_obs
        return new_obs

    # TODO Normalize the ouput of step
    def step(self, action):
        assert isinstance(action, Node), 'ERROR : action should be a Node'
        self.previous_available_actions = self.available_actions
        if isinstance(action, ExecAction):
            super().step(self.running_action)
            self.previous_state = self.state
            self.state = self.preprocess_raw_observation(self.user_state)
            self.available_actions = self.get_thing_list()
            done = True
            return (self.state, self.available_actions), None, done, None
        else:
            self.save_running_action(action)
            self.available_actions = action.get_children_nodes()
            assert self.available_actions is not None
            return (self.state, self.available_actions), 0, False, None

    def save_running_action(self, action):
        if isinstance(action, Thing):
            self.running_action['thing'] = action.name
        elif isinstance(action, Channel):
            self.running_action['channel'] = action.name
        elif isinstance(action, OpenHABAction):
            self.running_action['action'] = action.name
        elif isinstance(action, Params):
            self.running_action['params'] = action.interpret_params()  # TODO Have a param interpreter
        else:
            raise NotImplementedError

    def reset(self):
        raw_state = super().reset()
        # Cache node embedding
        things_list = self.get_thing_list()
        description_node_iterator = chain.from_iterable([things_list] + [t.get_channels() for t in things_list])
        for node in description_node_iterator:
            node.embed_node_description(self.description_embedder.get_description_embedding)

        self.state = self.preprocess_raw_observation(raw_state)
        self.previous_state = None
        self.available_actions = self.get_thing_list()
        self.previous_available_actions = None
        self.running_action = {"thing": None, 'channel': None, 'action': None, 'params': None}
        return self.state, self.available_actions

    def get_state_and_action(self):
        return self.state, self.available_actions


if __name__ == "__main__":
    env2 = IoTEnv()
    initial_state2 = env2.reset()

    env = IoTEnv4ML(env_class=IoTEnv, params={})
    initial_state = env.reset()
