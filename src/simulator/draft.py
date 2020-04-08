import gym
import numpy as np


class DiscreteToBoxWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete), \
            "Should only be used to wrap Discrete envs."
        self.n = self.observation_space.n
        self.observation_space = gym.spaces.Box(0, 1, (self.n,))

    def observation(self, obs):
        print(obs)
        new_obs = np.zeros(self.n)
        new_obs[obs] = 1
        return new_obs


class Parent(object):
    def __init__(self, name='papa', **kwargs):
        # del self.initial_value['__class__']
        self.parent = 1
        pass

    def reset(self):
        self.__init__(**self.initial_value)

class Child(Parent):
    def __init__(self, value=2):
        self.initial_value = locals().copy()
        del self.initial_value['self']
        del self.initial_value['__class__']
        hello = 3
        print(locals())
        super().__init__(value=2)
        self.child = value
        print(locals())


if __name__ == "__main__":
    c = Child()
    print(isinstance(c, Parent))
    d = dict()
    d['a'] = 1
    print(d)