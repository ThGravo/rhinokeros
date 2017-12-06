import numpy as np
from gym import spaces


# TODO this is just a partial policy
class Policy(object):
    def __init__(self, action_space, model_predict):
        self.action_space = action_space
        self.model_predict = model_predict

    def get_action(self, state):
        return self.model_predict(np.reshape(state, [1, -1]))


class RandomPolicy(Policy):
    def __init__(self, action_space, model_predict):
        super().__init__(action_space, model_predict)

    def get_action(self, state):
        return self.action_space.sample()


'''Greedy policies'''


class DirectPolicy(Policy):
    def __init__(self, action_space, model):
        super().__init__(action_space, model.predict)
        self.model = model

    def get_action(self, state):
        if isinstance(self.action_space, spaces.Discrete):
            if self.action_space.n is 2:
                return int(1 - np.round(self.model.predict(np.reshape(state, [1, -1])))[0, 0])
            else:
                return np.argmax(self.model.predict(np.reshape(state, [1, -1])))

        return self.model.predict(np.reshape(state, [1, -1]))


class ValuePolicy(Policy):
    def __init__(self, action_space, model):
        super().__init__(action_space, model.predict)
        self.model = model

    def get_action(self, state):
        if isinstance(self.action_space, spaces.Discrete):
            return np.argmax(self.model.predict(np.reshape(state, [1, -1])))

        return self.model.predict(np.reshape(state, [1, -1]))
