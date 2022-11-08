"""
Adapted from SB3
"""

from cmath import isinf
import warnings
from typing import Dict, Tuple, Union

import numpy as np
import torch
from gym import spaces
from torch.nn import functional as F


def get_obs_shape(
    observation_space: spaces.Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).
    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return (int(observation_space.n),)
    elif isinstance(observation_space, spaces.Dict):
        return {
            key: get_obs_shape(subspace)
            for (key, subspace) in observation_space.spaces.items()
        }


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.
    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def to_torch(x, device=torch.device("cpu"), copy=False):
    """
    converts x to a torch tensor
    """
    if isinstance(x, list):
        # if list, and items are dicts
        if len(x) == 0: return x
        if isinstance(x[0], dict):
            # leave alone as a list if its a list of dicts
            return [to_torch(e, device=device, copy=copy) for e in x]
        else:
            raise NotImplementedError("not implemented")
    if isinstance(x, dict):
        data = {}
        for k, v in x.items():
            data[k] = to_torch(v, device=device, copy=copy)
        return data
    else:
        if (isinstance(x, torch.Tensor)):
            if copy:
                return x.clone().to(device)
            else:
                return x.to(device)
        elif isinstance(x, np.ndarray):
            data = torch.from_numpy(x)
            if copy:
                return data.clone().to(device)
            else:
                return data.to(device)