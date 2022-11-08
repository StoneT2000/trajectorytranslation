import math
import time
import warnings
from typing import Any, Dict, Optional, Type, Union

import gym
import numpy as np
import torch
from gym import spaces
from torch import optim
from torch.nn import functional as F

from paper_rl.architecture.ac.core import ActorCritic, count_vars
from paper_rl.common.buffer import GenericBuffer
from paper_rl.common.rollout import Rollout
from paper_rl.common.utils import to_torch
from paper_rl.logger.logger import Logger
from paper_rl.modelfree.ppo.buffer import PPOBuffer


class PPO:
    def __init__(
        self,
        ac: ActorCritic,
        env: gym.Env,
        num_envs: int,
        observation_space,
        action_space,
        steps_per_epoch: int = 10000,
        train_iters: int = 80,
        gamma: float = 0.99,
        gae_lambda: float = 0.97,
        clip_ratio: float = 0.2,
        ent_coef: float = 0.0,
        pi_coef: float = 1.0,
        vf_coef: float = 1.0,
        dapg_lambda: float = 0.1,
        dapg_damping: float = 0.99,

        target_kl: Optional[float] = 0.01,
        logger: Logger = None,

        seed: Optional[int] = None,
        device: Union[torch.device, str] = "cpu",

    ) -> None:
        # Random seed
        if seed is None:
            seed = 0
        # seed += 10000 * proc_id()
        # torch.manual_seed(seed)
        # np.random.seed(seed)

        self.n_envs = num_envs
        self.env = env  # should be vectorized
        self.device = device
        self.observation_space = observation_space
        self.action_space = action_space

        # hparams
        self.target_kl = target_kl
        self.pi_coef = pi_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.clip_ratio = clip_ratio
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.dapg_lambda = dapg_lambda
        self.dapg_damping = dapg_damping

        # self.pi_optimizer = optim.Adam(ac.pi.parameters(), lr=pi_lr)
        # self.vf_optimizer = optim.Adam(ac.v.parameters(), lr=vf_lr)

        # exp params
        self.train_iters = train_iters
        self.steps_per_epoch = steps_per_epoch

        self.logger = logger
        self.buffer = PPOBuffer(
            buffer_size=self.steps_per_epoch,
            observation_space=observation_space,
            action_space=action_space,
            n_envs=self.n_envs,
            gamma=self.gamma,
            lam=self.gae_lambda,
        )
        self.ac = ac.to(self.device)
        var_counts = tuple(count_vars(module) for module in [ac.pi, ac.v])
        self.logger.print(
            "\nNumber of parameters: \t pi: %d, \t v: %d\n" % var_counts,
            color="green",
            bold=True,
        )

    def to_state_dict(self):
        state_dict = dict(
            ac_state_dict=self.ac.state_dict(),
        )
        return state_dict

    def train(
        self,
        train_callback=None,
        rollout_callback=None,
        obs_to_tensor=None,
        max_ep_len=None,
        start_epoch: int = 0,
        n_epochs: int = 10,
        critic_warmup_epochs: int = 0, # number of epochs to collect rollouts and update critic only, freezing policy
        pi_optimizer: torch.optim.Optimizer = None,
        vf_optimizer: torch.optim.Optimizer = None,
        batch_size=1000,
        accumulate_grads=False,
        demo_trajectory_sampler = None,
        dapg_nll_loss=False,
        verbose=1,
    ):
        """
        Parameters
        ----------

        max_ep_len - max episode length. Can be set to infinity if environment returns a done signal to cap max episode length    

        demo_trajectory_sampler - optional function that returns sampled demo trajectories to provide for DAPG training
        """
        dapg = False
        if demo_trajectory_sampler is not None: dapg = True
        if max_ep_len is None:
            raise ValueError("max_ep_len is missing")
        ac = self.ac
        env = self.env
        buf = self.buffer
        logger = self.logger
        clip_ratio = self.clip_ratio
        train_iters = self.train_iters
        target_kl = self.target_kl
        n_envs = self.n_envs
        device = self.device
        rollout = Rollout()
        def policy(o):
            if obs_to_tensor is None:
                o = torch.as_tensor(o, dtype=torch.float32)
            else:
                o = obs_to_tensor(o)
            return ac.step(o)

        def update(update_pi=True):
            all_demo_trajectories = None
            if dapg:
                all_demo_trajectories = []
                steps_per_train_iter = int(math.ceil(buf.buffer_size * buf.n_envs / batch_size))
                ac.eval()
                with torch.no_grad():
                    for _ in range(steps_per_train_iter):
                        demo_trajectories = demo_trajectory_sampler(batch_size)
                        if not dapg_nll_loss:
                            # collect old logp
                            pi, logp = ac.pi(to_torch(demo_trajectories["observations"], device=device), to_torch(demo_trajectories["actions"],  device=device))
                            demo_trajectories["logp"] = logp
                        all_demo_trajectories.append(demo_trajectories)
                ac.train()


            update_res = ppo_update(
                buffer=buf,
                ac=ac,
                pi_optimizer=pi_optimizer,
                vf_optimizer=vf_optimizer,
                pi_coef=self.pi_coef,
                vf_coef=self.vf_coef,
                clip_ratio=clip_ratio,
                train_iters=train_iters,
                batch_size=batch_size,
                target_kl=target_kl,
                logger=logger,
                device=device,
                accumulate_grads=accumulate_grads,
                update_pi=update_pi,
                demo_trajectories=all_demo_trajectories,
                dapg_lambda=self.dapg_lambda,
                dapg_nll_loss=dapg_nll_loss
            )
            pi_info, loss_pi, loss_v, update_step = (
                update_res["pi_info"],
                update_res["loss_pi"],
                update_res["loss_v"],
                update_res["update_step"],
            )
            logger.store(tag="train", StopIter=update_step, append=False)
            
            if loss_v is not None:
                logger.store(
                    tag="train",
                    LossV=loss_v.cpu().item(),
                )
            if pi_info is not None:
                kl, ent, cf, loss_pi, ppo_loss = pi_info["kl"], pi_info["ent"], pi_info["cf"], pi_info["loss_pi"], pi_info["ppo_loss"]
                logger.store(
                    tag="train",
                    LossPi=loss_pi,
                    PPOLoss=ppo_loss,
                    KL=kl,
                    Entropy=ent,
                    ClipFrac=cf,
                )
                if "dapg_actor_loss" in pi_info:
                    logger.store("train", LossDAPGActor=pi_info["dapg_actor_loss"])

        for epoch in range(start_epoch, start_epoch + n_epochs):

            # rollout
            rollout_start_time = time.time_ns()
            buf.reset()
            ac.eval()
            rollout.collect(policy=policy, env=env, n_envs=n_envs, buf=buf, steps=self.steps_per_epoch, rollout_callback=rollout_callback, max_ep_len=max_ep_len, logger=logger, verbose=verbose)
            ac.train()
            rollout_end_time = time.time_ns()
            rollout_delta_time = (rollout_end_time - rollout_start_time) * 1e-9
            logger.store("train", RolloutTime=rollout_delta_time, critic_warmup_epochs=critic_warmup_epochs, append=False)

            update_start_time = time.time_ns()
            # advantage normalization before update
            update_pi = epoch >= critic_warmup_epochs
            update(update_pi = update_pi)
            update_end_time = time.time_ns()

            if dapg:
                logger.store("train", dapg_lambda=self.dapg_lambda, append=False)
                if update_pi: self.dapg_lambda *= self.dapg_damping

            logger.store("train", UpdateTime=(update_end_time - update_start_time) * 1e-9, append=False)
            logger.store("train", Epoch=epoch, append=False)
            logger.store("train", TotalEnvInteractions=self.steps_per_epoch * self.n_envs * (epoch + 1), append=False)
            if train_callback is not None:
                early_stop = train_callback(epoch=epoch)
                if early_stop is not None and early_stop:
                    break

def compute_gae(
    buffer: GenericBuffer,
):  
    buffer.buffers["adv_buf"] = (buffer.buffers["adv_buf"] - buffer.buffers["adv_buf"].mean()) / (buffer.buffers["adv_buf"].std())



def ppo_update(
    buffer: GenericBuffer,
    ac: ActorCritic,
    pi_optimizer,
    vf_optimizer,
    pi_coef,
    vf_coef,
    clip_ratio,
    train_iters,
    batch_size,
    target_kl,
    logger=None,
    device=torch.device("cpu"),
    accumulate_grads=False,
    update_pi=True,
    demo_trajectories=None,
    dapg_lambda=0.1,
    dapg_nll_loss=False,
):
    """
    
    demo_trajectories - dict of ndarray or list of dicts
        if list, treats each element as a batch


    dapg_nll_loss - bool
        if true, uses negative log likelihood as the demo augmented loss. Otherwise uses PPO loss
    """
    dapg = False
    if demo_trajectories is not None: dapg = True
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data["obs_buf"], data["act_buf"].to(device), data["adv_buf"].to(device), data["logp_buf"].to(device)
        obs = to_torch(obs, device=device)

        # Policy loss)
        ac.pi.eval()
        pi, logp = ac.pi(obs, act)
        ac.pi.train()
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean() * pi_coef
        
        # Entropy loss for some basic extra exploration
        entropy = pi.entropy()
        with torch.no_grad():
            # Useful extra info
            logr = logp - logp_old
            approx_kl = (torch.exp(logr) - 1 - logr).mean().cpu().item()
            clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
            clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=entropy.mean().item(), cf=clipfrac, ppo_loss=loss_pi.cpu().item())

        # add dapg loss, the negative log likelihood in particular
        if dapg:
            demo_trajectories = data["demo_trajectories"]
            ac.pi.eval()
            demo_pi, demo_logp = ac.pi(to_torch(demo_trajectories["observations"], device=device), to_torch(demo_trajectories["actions"], device=device))
            ac.pi.train()

            if dapg_nll_loss:
                dapg_actor_loss = -demo_logp.mean()
            else:
                # mark advantage as 1 since advantage is normalized
                demo_logp_old = demo_trajectories["logp"]
                ratio = torch.exp(demo_logp - demo_logp_old)
                clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
                dapg_actor_loss = -(torch.min(ratio, clip_adv)).mean()
            dapg_actor_loss = dapg_actor_loss * dapg_lambda
            pi_info["dapg_actor_loss"] = dapg_actor_loss.cpu().item()
            loss_pi = loss_pi + dapg_actor_loss
        pi_info["loss_pi"] = loss_pi.cpu().item()

        return dict(loss_pi=loss_pi, logp=logp, entropy=entropy, pi_info=pi_info)

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data["obs_buf"], data["ret_buf"].to(device)
        if isinstance(obs, dict):
            for k in obs.keys():
                obs[k] = obs[k].to(device)
        else:
            obs = obs.to(device)
        return ((ac.v(obs) - ret) ** 2).mean()

    # Train policy with multiple steps of gradient descent
    update_step = 0
    early_stop_update = False
    loss_pi = None
    pi_info = None

    # compute gae here
    compute_gae(buffer=buffer)

    for _ in range(train_iters):
        if early_stop_update:
            break
        N = buffer.buffer_size * buffer.n_envs

        steps_per_train_iter = int(math.ceil(N / batch_size))
        if accumulate_grads:
            if update_pi: pi_optimizer.zero_grad()
            vf_optimizer.zero_grad()
            average_kl = 0
        for batch_idx in range(steps_per_train_iter):
            batch_data = buffer.sample_batch(batch_size=batch_size)
            loss_v = compute_loss_v(batch_data) * vf_coef
            if dapg: 
                batch_demo_trajectories = demo_trajectories[batch_idx]
                batch_data["demo_trajectories"] = batch_demo_trajectories
            if update_pi:
                loss_pi_data = compute_loss_pi(batch_data)
                loss_pi, logp, entropy, pi_info = loss_pi_data["loss_pi"], loss_pi_data["logp"], loss_pi_data["entropy"], loss_pi_data["pi_info"]
                kl = pi_info["kl"]
                if accumulate_grads:
                    average_kl += kl
                if not accumulate_grads and target_kl is not None:
                    if kl > 1.5 * target_kl:
                        logger.print("Early stopping at step %d due to reaching max kl." % update_step)
                        early_stop_update = True
                        break
            
            if not accumulate_grads:
                if update_pi:
                    pi_optimizer.zero_grad()
                vf_optimizer.zero_grad()
                if update_pi:
                    loss_pi.backward()
                    pi_optimizer.step()
                loss_v.backward()
                vf_optimizer.step()
                update_step += 1
            if accumulate_grads:
                # scale loss down
                if update_pi: loss_pi /= steps_per_train_iter
                loss_v /= steps_per_train_iter
                if update_pi: loss_pi.backward()
                loss_v.backward()
        
        if accumulate_grads and target_kl is not None:
            average_kl /= steps_per_train_iter
            if average_kl > 1.5 * target_kl:
                logger.print("Early stopping at step %d due to reaching max kl." % update_step)
                early_stop_update = True
                break
        if accumulate_grads:
            if update_pi: pi_optimizer.step()
            vf_optimizer.step()
            update_step += 1
        if early_stop_update:
            break
    return dict(
        pi_info=pi_info,
        update_step=update_step,
        loss_v=loss_v,
        loss_pi=loss_pi,
    )
