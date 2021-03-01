from collections import OrderedDict, UserDict
import json

import gym

from simulator.Thing import Thing, ThingState
from simulator.Channel import Channel, ChannelState
from simulator.Action import ExecAction, OpenHABAction, Params, DoNothing
from simulator.TreeView import Node
from simulator.discrete_parameters import discrete_parameters


class State(ChannelState):
    def copy(self):
        return State(self)


class IoTEnv(gym.Env):
    def __init__(self, obj_list):
        super().__init__()
        # Define object in the environment

        self._things_lookup_table = {}
        for obj in obj_list:
            thing = obj.Class(**obj.Params)
            self._things_lookup_table[thing.name] = thing

        # TODO Fix this for gym compatibility
        # Compute observation_space
        # self.observation_space = gym.spaces.Dict(
        #     {k: v.get_observation_space() for k, v in self._things_lookup_table.items()}
        # )
        #
        # self.action_space = gym.spaces.Dict(
        #     {k: v.get_action_space() for k, v in self._things_lookup_table.items()}
        # )

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

        self.discrete_params = discrete_parameters

        self.oracle_state = self.build_state(oracle=True)
        self.previous_oracle_state = self.oracle_state

    def get_thing_list(self):
        return list(self._things_lookup_table.values())

    def get_visible_thing_list(self):
        return [t for t in self.get_thing_list() if t.is_visible]

    def get_thing(self, name):
        return self._things_lookup_table[name]

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

        self.previous_oracle_state = self.oracle_state
        self.oracle_state = self.build_state(oracle=True)

        reward = None
        done = None
        info = None
        return self.oracle_state, reward, done, info

    def build_state(self, oracle, thing=None):
        """

        :param thing: None means build oracle_state for all Things in the environment
        :return:
        """
        if thing is None:
            thing_list = self.get_visible_thing_list()
        elif isinstance(thing, Thing):
            thing_list = [thing]
        elif isinstance(thing, list):
            thing_list = thing
        else:
            raise TypeError('thing should be None, a thing object or a list of Thing object')

        state = dict()
        for thing in thing_list:
            state[thing.name] = thing.get_state(oracle=oracle)
        return State(state)

    def reset(self):
        thing_list = self.get_thing_list()
        for thing in thing_list:
            thing.reset()
        self.oracle_state = self.build_state(oracle=True)
        self.previous_oracle_state = self.oracle_state
        return self.oracle_state

    def render(self, mode='human'):
        raise NotImplementedError

    def set_all_things_visible(self):
        for thing in self.get_thing_list():
            thing.update_visibility(visibility=True)


class IoTEnv4ML(gym.Wrapper):
    def __init__(self, thing_params, ignore_exec_action, allow_do_nothing, max_episode_length,
                 filter_state_during_episode, episode_reset, env_class=IoTEnv):
        super().__init__(env=env_class(thing_params))

        self.running_action = {"thing": None, 'channel': None, 'action': None, 'params': None}

        self.ignore_exec_action = ignore_exec_action
        self.allow_do_nothing = allow_do_nothing
        self.max_episode_length = max_episode_length
        self.episode_reset = episode_reset

        self.episode_length = 0
        self.available_actions = None
        self.previous_available_actions = None
        self.state = None
        self.previous_state = None
        self.reset()

        self.filter_state_during_episode = filter_state_during_episode

        # # Compute node embedding
        # things_list = self.get_thing_list()
        # description_node_iterator = chain.from_iterable([things_list] + [t.get_channels() for t in things_list])
        # for node in description_node_iterator:
        #     node.embed_node_description(self.description_embedder.get_description_embedding)
        # print(1)

    # TODO Normalize the ouput of step
    def step(self, action):
        assert isinstance(action, Node), 'ERROR : action should be a Node'
        self.previous_available_actions = self.available_actions
        if isinstance(action, ExecAction):
            super().step(self.running_action)
            self.previous_state = self.state
            self.state = self.build_state(oracle=False)
            self.reset_running_action()
            self.episode_length += 1
            self.available_actions = self.get_root_actions()
            done = self.episode_length >= self.max_episode_length
            reward = None if done else 0
            info = 'exec_action'
        elif isinstance(action, DoNothing):
            done = True
            reward = None
            info = 'do_nothing'
        else:
            self.save_running_action(action)
            self.available_actions = action.get_children_nodes()
            if self.filter_state_during_episode:
                self.filter_state(action)
            assert self.available_actions is not None
            reward = 0
            done = False
            info = ""

            if self.ignore_exec_action and any([isinstance(a, ExecAction) for a in self.available_actions]):
                return self.step(ExecAction())

        available_actions = self.available_actions if not done else []
        # state = self.state if not done else []
        return (self.state, available_actions), reward, done, info

    def filter_state(self, action):
        self.previous_state = self.state
        if isinstance(action, Thing):
            self.state = State({action.name: self.state[action.name]})
        elif isinstance(action, Channel):
            assert len(self.state) == 1
            thing_name = list(self.state.keys())[0]
            channel_state = self.state[thing_name][action.name]
            thing_state = ThingState({action.name : channel_state}, description=self.state[thing_name]['description'])
            self.state = State({thing_name: thing_state})

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
        super().reset()
        self.previous_state = self.state
        self.state = self.build_state(oracle=False)
        self.episode_length = 0
        self.available_actions = self.get_root_actions()
        self.previous_available_actions = None
        self.reset_running_action()
        return self.state, self.available_actions

    def get_state_and_action(self):
        return self.state, self.available_actions

    def get_root_actions(self):
        available_things = self.get_visible_thing_list()
        if self.allow_do_nothing:
            available_things.append(DoNothing())
        return available_things

    def reset_running_action(self):
        self.running_action = {"thing": None, 'channel': None, 'action': None, 'params': None}

