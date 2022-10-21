"""
Adapted from SB3
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import torch
from gym import spaces

from paper_rl.common.utils import get_action_dim, get_obs_shape


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        device: Union[torch.device, str] = "cpu",
        n_envs: int = 1,
    ):
        super(BaseBuffer, self).__init__()
        self.buffer_size = buffer_size
        
        self.ptr = 0
        self.full = False
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.n_envs = n_envs

    def store(self, *args, **kwargs) -> None:
        """
        store elements to the buffer.
        """
        raise NotImplementedError()

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.ptr

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.ptr = 0
        self.full = False


class GenericBuffer(BaseBuffer):
    """
    Generic buffer that stores key value items for vectorized environment outputs.
    """

    def __init__(
        self,
        buffer_size: int,
        device: Union[torch.device, str] = "cpu",
        n_envs: int = 1,
        config=dict() 
    ):
        """
        
        config - dict(k->v) where k is buffer name and v[0] is shape, v[1] is numpy dtype, v[2] is data is dict or not. if is_dict, then shape and dtype should be a dict of shapes and dtypes
        """
        super().__init__(
            buffer_size=buffer_size,
            device=device,
            n_envs=n_envs,
        )
        self.is_dict = dict()
        self.config = config
        self.buffers = dict()
        for k in config.keys():
            shape, dtype = config[k]
            is_dict = False
            if isinstance(shape, dict):
                is_dict = True
            self.is_dict[k] = is_dict
            if is_dict:
                self.buffers[k] = dict()
                for part_key in shape.keys():
                    self.buffers[k][part_key] = np.zeros((self.buffer_size, self.n_envs) + shape[part_key], dtype=dtype[part_key])
            else:
                self.buffers[k] = np.zeros((self.buffer_size, self.n_envs) + shape, dtype=dtype)
        self.ptr, self.path_start_idx, self.max_size = 0, [0]*n_envs, self.buffer_size
        
        self.batch_idx = None
        self.batch_inds = None
        self.batch_env_inds = None

    def store(self, **kwargs):
        """
        store one timestep of agent-environment interaction to the buffer. If full, replaces the oldest entry
        """
        for k in kwargs.keys():
            data = kwargs[k]
            if self.is_dict[k]:
                for data_k in data.keys():
                    d = np.array(data[data_k]).copy()
                    d = d.reshape(self.buffers[k][data_k][self.ptr].shape)
                    self.buffers[k][data_k][self.ptr] = d
            else:
                d = np.array(data).copy()
                d = d.reshape(self.buffers[k][self.ptr].shape)
                self.buffers[k][self.ptr] = d
        self.ptr += 1
        if self.ptr == self.buffer_size:
            # wrap pointer around to start replacing items
            self.full = True
            self.ptr = 0

    def _get_batch_by_ids(self, batch_ids, env_ids):
        batch_data = dict()
        for k in self.buffers.keys():
            data = self.buffers[k]
            if self.is_dict[k]:
                batch_data[k] = dict()
                for data_k in data.keys():
                    batch_data[k][data_k] = torch.as_tensor(data[data_k][batch_ids, env_ids])
            else:
                batch_data[k] = torch.as_tensor(data[batch_ids, env_ids])
        return batch_data
    def _prepared_for_sampling(self, batch_size, drop_last_batch=True):
        if self.batch_idx == None: return False
        if drop_last_batch and self.batch_idx + batch_size > self.buffer_size * self.n_envs: return False
        if self.batch_idx > self.buffer_size * self.n_envs: return False
        return True

    def sample_batch(self, batch_size: int, drop_last_batch=True):
        if not self._prepared_for_sampling(batch_size, drop_last_batch):
            self.batch_idx = 0            
            if self.full:
                inds = np.arange(0, self.buffer_size).repeat(self.n_envs)
            else:
                inds = np.arange(0, self.ptr).repeat(self.n_envs)
            env_inds = np.tile(np.arange(self.n_envs), len(inds) // self.n_envs)
            inds = np.vstack([inds, env_inds]).T
            np.random.shuffle(inds)
            self.batch_inds = inds[:, 0]
            self.batch_env_inds = inds[:, 1]
        batch_ids = self.batch_inds[self.batch_idx: self.batch_idx + batch_size]
        env_ids = self.batch_env_inds[self.batch_idx: self.batch_idx + batch_size]
        self.batch_idx = self.batch_idx + batch_size
        return self._get_batch_by_ids(batch_ids=batch_ids, env_ids=env_ids)
            
    def sample_random_batch(self, batch_size: int):
        if self.full:
            batch_ids = (np.random.randint(0, self.buffer_size, size=batch_size) + self.ptr) % self.buffer_size
        else:
            batch_ids = np.random.randint(0, self.ptr, size=batch_size)
        env_ids = np.random.randint(0, high=self.n_envs, size=(len(batch_ids),))

        return self._get_batch_by_ids(batch_ids=batch_ids, env_ids=env_ids)
    def get(self):
        all_data = dict()
        for k in self.buffers.keys():
            data = self.buffers[k]
            if self.is_dict[k]:
                all_data[k] = dict()
                for data_k in data.keys():
                    all_data[k][data_k] = torch.as_tensor(data[data_k].reshape(-1))
            else:
                all_data[k] = torch.as_tensor(data.reshape(-1))
        return all_data