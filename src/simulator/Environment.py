import numpy as np
from sklearn.preprocessing import OneHotEncoder

import gym
from Items import ITEM_TYPE
from Thing import Thing, PlugSwitch, LightBulb
from Channel import Channel
from description_embedder import Description_embedder
from Action import ExecAction, OpenHABAction, Params
from TreeView import Node

class IoTEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Define object in the environment
        self.plug = PlugSwitch()
        self.lightbulb = LightBulb()

        # Connect objects and set
        # self.plug.connect_thing(self.lightbulb)

        # Compute observation_space
        self.observation_space = gym.spaces.Dict({
            "plug": self.plug.get_observation_space(),
            "lightbulb": self.lightbulb.get_observation_space(),
        })

        # Compute action_space
        self.action_space = gym.spaces.Dict({
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
        thing_list = self.get_thing_list()
        for thing in thing_list:
            thing.reset()
        state = self.build_state()
        return (state, thing_list)

    def render(self, mode='human'):
        raise NotImplementedError


class IoTEnv4ML(gym.Wrapper):
    def __init__(self, env=IoTEnv()):
        super().__init__(env=env)
        self.description_embedder = Description_embedder(embedding='glove', dimension=50, reduction='mean',
                                                         authorize_cache=True)
        self.item_type_embedder = OneHotEncoder(sparse=False)
        self.item_type_embedder.fit(np.array(ITEM_TYPE).reshape(-1, 1))
        self.running_action = dict()

    def get_overall_observation(self, observation):
        new_obs = dict()
        for thing_name, thing in observation.items():
            thing_obs = dict()
            for channel_name, channel in thing.items():
                if channel['item_type'] == 'goal_string':
                    raise NotImplementedError
                description_embedding = self.description_embedder.get_description_embedding(channel['description'])
                item_embedding = self.item_type_embedder.transform(
                    np.array(channel['item_type']).reshape(-1, 1)).flatten()
                state_embedding = np.zeros(3)
                state_embedding[:len(channel['state'])] = channel['state']
                channel_embedding = np.concatenate([description_embedding, item_embedding, state_embedding])
                thing_obs[channel_name] = channel_embedding
            new_obs[thing_name] = thing_obs
        return new_obs

    def observation(self, observation):
        overall_observation = self.get_overall_observation(observation[0])
        local_observation = self.get_local_observation(observation[1])
        return overall_observation, local_observation

    def step(self, action):
        assert isinstance(action, Node), 'ERROR : action should be a Node'
        if isinstance(action, ExecAction):
            super().step(self.running_action)
        else:
            self.save_running_action(action)
            available_actions = action.get_children_nodes()
            return available_actions

    def save_running_action(self, action):
        if isinstance(action, Thing):
            self.running_action['thing'] = action.name
        elif isinstance(action, Channel):
            self.running_action['channel'] = action.name
        elif isinstance(action, OpenHABAction):
            self.running_action['action'] = action.name
        elif isinstance(action, Params):
            self.running_action['param'] = action.name # TODO Have a param interpreter
        else:
            raise NotImplementedError

    def reset(self):
        state, thing_list = super().reset()
        self.running_action = dict()
        return state, thing_list


class IoTEnvTreeLike(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env=env)


if __name__ == "__main__":
    env = IoTEnv4ML()
    initial_state = env.reset()

    env2 = IoTEnv()
    initial_state2 = env2.reset()
