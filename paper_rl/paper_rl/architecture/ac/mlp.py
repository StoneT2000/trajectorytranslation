import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from paper_rl.architecture.ac.core import Actor, ActorCritic, mlp


class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        act = act.reshape(-1)
        return pi.log_prob(act)

    def act(self, obs):
        logits = self.logits_net(obs)
        return logits.argmax(1)


class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, log_std_scale=-0.5):
        super().__init__()
        log_std = log_std_scale * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(
            axis=-1
        )  # Last axis sum needed for Torch Normal distribution

    def act(self, obs):
        mu = self.mu_net(obs)
        return mu


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        out = self.v_net(obs)
        return torch.squeeze(
            out, -1
        )  # Critical to ensure v has right shape.


class MLPActorCritic(ActorCritic):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(64, 64),
        activation=nn.Tanh,
        log_std_scale=-0.5,
    ):
        super().__init__()
        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(
                obs_dim, action_space.shape[0], hidden_sizes, activation, log_std_scale
            )
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(
                obs_dim, action_space.n, hidden_sizes, activation
            )

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs, deterministic=False):
        if deterministic:
            return self.pi.act(obs).cpu().numpy()
        return self.step(obs)[0]

    def get_value(self, x):
        return self.v(x)