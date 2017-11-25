from policies import policy
import numpy as np


class EpsilonGreedyPolicy(policy):
    def __init__(self, action_space, model, initial_epsilon=.1, minimal_epsilon=0, epsilon_decay_factor=1):
        super().__init__(action_space)
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.minimal_epsilon = minimal_epsilon
        self.epsilon_decay_factor = epsilon_decay_factor
        self.model = model

    def get_action(self, state, time_step=None):
        self.epsilon = self.epsilon * self.epsilon_decay_factor if time_step is None else self.initial_epsilon * self.epsilon_decay_factor ** time_step
        self.epsilon = max((self.epsilon, self.minimal_epsilon))
        if np.random.rand(1) < self.epsilon:
            return self.action_space.sample()
        else:
            return self.model(state)
