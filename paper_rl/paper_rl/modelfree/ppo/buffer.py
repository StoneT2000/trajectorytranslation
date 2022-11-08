from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import torch
from gym import spaces

from paper_rl.common.buffer import GenericBuffer
from paper_rl.common.stats import discount_cumsum
from paper_rl.common.utils import get_action_dim, get_obs_shape


class PPOBuffer(GenericBuffer):
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_envs: int = 1,
        gamma=0.99,
        lam=0.95
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)

        self.action_dim = get_action_dim(action_space)
        buffer_config = dict(
            act_buf = ((self.action_dim,), action_space.dtype),
            adv_buf = ((), np.float32),
            rew_buf = ((), np.float32),
            ret_buf = ((), np.float32),
            val_buf = ((), np.float32),
            logp_buf = ((), np.float32),
            done_buf = ((), np.bool8)
        )
        if isinstance(self.obs_shape, dict):
            buffer_config["obs_buf"] = (self.obs_shape, {k: self.observation_space[k].dtype for k in self.observation_space})
        else:
            buffer_config["obs_buf"] = (self.obs_shape, np.float32)
        super().__init__(
            buffer_size=buffer_size,
            n_envs=n_envs,
            config=buffer_config
        )

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, [0] * n_envs, self.buffer_size
        self.next_batch_idx = 0

    def finish_path(self, env_id, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx[env_id], self.ptr)
        if self.ptr < self.path_start_idx[env_id]:
            path_slice = slice(self.path_start_idx[env_id],self.buffer_size)
        rews = np.append(self.buffers["rew_buf"][path_slice, env_id], last_val)
        vals = np.append(self.buffers["val_buf"][path_slice, env_id], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]

        self.buffers["adv_buf"][path_slice, env_id] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.buffers["ret_buf"][path_slice, env_id] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx[env_id] = self.ptr

    def reset(self) -> None:
        super().reset()
        self.ptr, self.path_start_idx = 0, [0] * self.n_envs