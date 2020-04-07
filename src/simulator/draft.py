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
    def __init__(self):
        self.parent = 1
        pass

class Child(Parent):
    def __init__(self, value=2):
        super().__init__()
        self.child = value


if __name__ == "__main__":
    c = Child()
    print(isinstance(c, Parent))
    d = dict()
    d['a'] = 1
    print(d)