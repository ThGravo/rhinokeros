import numpy as np
from gym import spaces


class Policy(object):
    def __init__(self, action_space, model_predict):
        self.action_space = action_space
        self.model_predict = model_predict

    def get_action(self, state):
        if isinstance(self.action_space, spaces.Discrete):
            if self.action_space.n is 2:
                return int(np.round(self.model_predict(np.reshape(state, [1, -1])))[0, 0])
            else:
                return np.argmax(self.model_predict(np.reshape(state, [1, -1])))

        return self.model_predict(np.reshape(state, [1, -1]))


class RandomPolicy(Policy):
    def __init__(self, action_space, model_predict):
        super().__init__(action_space, model_predict)

    def get_action(self, state):
        return self.action_space.sample()
