import numpy as np
from sklearn.preprocessing import OneHotEncoder

import gym
from Items import ITEM_TYPE
from Thing import Thing, PlugSwitch, Speaker, LightBulb, LGTV, Chromecast
from description_embedder import Description_embedder


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
        for thing in self.get_thing_list():
            thing.reset()
        state = self.build_state()
        return state

    def render(self, mode='human'):
        raise NotImplementedError


class IoTEnv4ML(gym.ObservationWrapper):
    def __init__(self, env=IoTEnv()):
        super().__init__(env=env)
        self.description_embedder = Description_embedder(embedding='glove', dimension=50, reduction='mean',
                                                         authorize_cache=True)
        self.item_type_embedder = OneHotEncoder(sparse=False)
        self.item_type_embedder.fit(np.array(ITEM_TYPE).reshape(-1, 1))

    def observation(self, observation):
        new_obs = dict()
        for thing_name, thing in observation.items():
            thing_obs = dict()
            for channel_name, channel in thing.items():
                if channel['item_type'] == 'string':
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


if __name__ == "__main__":
    env = IoTEnv4ML()
    initial_state = env.reset()
