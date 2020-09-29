import random
from abc import ABC, abstractmethod

class X(ABC):
    @abstractmethod
    def get(self):
        pass

class Y(object):
    def __init__(self):
        self.y = 0

class A(X, Y):
    def get(self):
        return 1

class B(Y, X):
    def get(self):
        return 1

if __name__ == '__main__':
    d = dict(a=1)
    try:
        print(d['a'])
    except AttributeError as e:
        raise NotImplementedError(e)

    try:
        print(d['b'])
    except KeyError as e:
        raise NotImplementedError(e)

