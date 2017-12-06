from classification_model import build_classification_model, build_binary_classification_model
from regression_model import build_regression_model
import numpy as np
from gym import spaces


def build_policy_model(observation_space, action_space,
                       dim_multipliers=(6,),
                       activations=('tanh',),
                       lr=.001):
    if isinstance(observation_space, spaces.Discrete):
        input_dim = observation_space.n
    else:
        input_dim = np.prod(observation_space.shape)

    if isinstance(action_space, spaces.Discrete):
        if action_space.n is 2:
            return build_binary_classification_model(input_dim, dim_multipliers=dim_multipliers,
                                                     activations=activations, lr=lr)
        else:
            return build_classification_model(input_dim, action_space.n, dim_multipliers=dim_multipliers,
                                              activations=activations, lr=lr)
    elif isinstance(action_space, spaces.Box):
        return build_regression_model(input_dim, np.prod(action_space.shape), dim_multipliers=dim_multipliers,
                                      activations=activations, lr=lr)
    else:
        raise ValueError("The action_space is of type: {} - which is not supported!".format(type(action_space)))


def build_q_model(observation_space, action_space,
                  dim_multipliers=(6,),
                  activations=('tanh',),
                  lr=.001):
    if isinstance(observation_space, spaces.Discrete):
        input_dim = observation_space.n
    else:
        input_dim = np.prod(observation_space.shape)

    if isinstance(action_space, spaces.Discrete):
        return build_regression_model(input_dim, action_space.n, dim_multipliers=dim_multipliers,
                                      activations=activations, lr=lr)
    elif isinstance(action_space, spaces.Box):
        # For continuous action spaces it's tricky to represent a q-function
        #  - would need to return a actual function, e.g. a Mixture of Gaussians
        return build_regression_model(input_dim + np.prod(action_space.shape), 1, dim_multipliers=dim_multipliers,
                                      activations=activations, lr=lr)
    else:
        raise ValueError("The action_space is of type: {} - which is not supported!".format(type(action_space)))
