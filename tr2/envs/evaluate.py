import gym
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from stable_baselines3.common.vec_env import SubprocVecEnv

from paper_rl.common.rollout import Rollout

eval_envs = None

def evaluate_online(env_name, n_envs, env_cfg, ids, policy, device, use_teacher, noise=None, render=False, verbose=False):
    ids_per_env = len(ids) // n_envs
    def make_env(idx):
        def _init():
            import tr2.envs
            env_kwargs = OmegaConf.to_container(env_cfg)
            env_kwargs["trajectories"] = ids[idx * ids_per_env: (idx + 1) * ids_per_env]
            env = gym.make(env_name, **env_kwargs)
            env.seed(10000*idx)
            return env
        return _init

    rollout = Rollout()
    noise_generator = np.random.default_rng(0)
    def obs_to_tensor(o):
        tensor_o = {}
        for k in o:
            tensor_o[k] = torch.as_tensor(o[k], device=device)
            if tensor_o[k].dtype == torch.float64:
                tensor_o[k] = tensor_o[k].float()
        return tensor_o
    def wrapped_policy(o):
        with torch.no_grad():
            o = obs_to_tensor(o)
            if not use_teacher:
                o["teacher_attn_mask"][:] = False
                o["teacher_frames"] = o["teacher_frames"] * 0
            a = policy(o)
            if noise is not None and noise != "None":
                a = a + noise_generator.normal(0, noise, size=a.shape)
        return a
    def format_trajectory(t_obs, t_act, t_rew, t_info):
        ep_len = len(t_act)
        t_info = t_info[-1]
        return dict(
            returns=np.sum(t_rew),
            traj_match=t_info["stats"]["farthest_traj_match_frac"],
            match_left=t_info["traj_len"] - t_info["stats"]["farthest_traj_match"] - 1,
            traj_len=t_info["traj_len"],
            ep_len=ep_len,
            success=t_info['task_complete'],
            traj_id=t_info["traj_id"]
        )
    global eval_envs
    if eval_envs is None:
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
        eval_envs = env
    else:
        env = eval_envs
    trajectories = rollout.collect_trajectories(
        policy=wrapped_policy,
        env=env,
        n_trajectories=(len(ids) // n_envs) * n_envs,
        n_envs=n_envs,
        pbar=verbose,
        even_num_traj_per_env=True,
        format_trajectory=format_trajectory,
        render=render,
    )
    val_results = []
    for traj_set in trajectories:
        for traj in traj_set:
            val_results.append(traj)
    val_df = pd.DataFrame(val_results)
    avg_return = val_df["returns"].mean()
    success_rate = (val_df['success'] == True).sum() / len(val_df)
    # env.close()
    return dict(
        df=val_df,
        avg_return=avg_return,
        success_rate=success_rate,
    )