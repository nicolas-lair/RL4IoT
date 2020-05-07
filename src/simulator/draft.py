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

    import numpy as np
    n = np.random.randint(1,5,1000)


    # for i in range(1000):
    #     print(i, n[i])
    #     a = np.random.rand(n[i])
    #     a_relu = np.maximum(np.zeros(n[i]), a - 0.5)
    #     sum = np.sum(a_relu)
    #     or_ = int(sum > 0)
    #     assert or_ == (max(a) > 0.5)
    # print('end')
    import torch
    for i in range(10000):
        t = torch.rand(3)
        temp = (t - 0.5).relu() # met toutes les proba < 0.5 à 0
        temp = temp.sum() # stricly positive only if at least one proba > 0.5
        temp *= 100000 # define sensibility to 0.00001, make temp > 1 if at least one proba > 0.50001
        bool = temp.clamp(0, 1)
        assert (t.max() > 0.5) == bool