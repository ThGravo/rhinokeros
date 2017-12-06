import numpy as np
from collections import deque
import gym
from gym import spaces
import cv2


class Observ2OneHotWrapper(gym.Wrapper):
    def __init__(self, env):
        """Make categorical observations a one-hot-encoding
        """
        gym.Wrapper.__init__(self, env)

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        one_hot_obs = np.zeros((1, self.env.observation_space.n))
        one_hot_obs[0, obs] = 1
        return one_hot_obs, reward, done, info

    def _reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        one_hot_obs = np.zeros((1, self.env.observation_space.n))
        one_hot_obs[0, obs] = 1
        return one_hot_obs
