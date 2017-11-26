from classification_model import build_classification_model, build_binary_classification_model
from regression_model import build_regression_model
import numpy as np
from gym import spaces


def build_policy_model(observation_space, action_space,
                       dim_multipliers=(6,),
                       activations=('tanh',),
                       lr=.001):
    if isinstance(action_space, spaces.Discrete):
        if action_space.n is 2:
            return build_binary_classification_model(np.prod(observation_space.shape),
                                                     dim_multipliers=dim_multipliers, activations=activations, lr=lr)
        else:
            return build_classification_model(np.prod(observation_space.shape), action_space.n,
                                              dim_multipliers=dim_multipliers, activations=activations, lr=lr)
    elif isinstance(action_space, spaces.Box):
        return build_regression_model(np.prod(observation_space.shape), np.prod(action_space.shape),
                                      dim_multipliers=dim_multipliers, activations=activations, lr=lr)
    else:
        raise ValueError("The action_space is of type: {} - which is not supported!".format(type(action_space)))
