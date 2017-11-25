from collections import deque as DequeMemory
from policy import RandomPolicy


class Agent(object):
    def __init__(self, observation_space, action_space):
        self.policy = RandomPolicy(action_space, action_space.sample)
        # self.memory = DequeMemory(maxlen=1)  # just to have the previous state, action and reward
        self.observation_space = observation_space
        self.last_state = None
        self.last_action = None

    def act(self, state, prev_reward, prev_done):
        action = self.policy.get_action(state)
        # self.memory.append((self.last_state, self.last_action, prev_reward, prev_done))
        self.last_state = state
        self.last_action = action
        return action, False
