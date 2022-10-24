import os
from pathlib import Path
import pickle
import time

import gym
import numpy as np
import sapien.core as sapien
import tqdm
import os.path as osp
import skilltranslation.envs
from omegaconf import OmegaConf
from tqdm import tqdm
from skilltranslation.planner.boxpusherteacher import BoxPusherReacherPlanner, BoxPusherTaskPlanner
from paper_rl.common.rollout import Rollout
from stable_baselines3.common.vec_env import SubprocVecEnv

from skilltranslation.utils.sampling import resample_teacher_trajectory
from skilltranslation.envs.xmagical.register_envs import register_envs
from skilltranslation.envs.xmagical.env import SweepToTopEnv
register_envs()

def watch_traj(env, traj):
    obs = env.reset()
    for obs in traj["observations"]:
        env.set_state_xy(obs)
        time.sleep(0.02)
        env.render()
if __name__ == "__main__":
    
    def make_env(idx):
        def _init():
            import skilltranslation.envs.boxpusher.traj_env
            embodiment = 'Shortstick'
            env = gym.make(f'SweepToTop-{embodiment}-State-Allo-TestLayout-v0')
            env.seed(idx)
            return env
        return _init
    speed = 4e-2
    cfg = OmegaConf.from_cli()
    traj_id = cfg['id']
    env: SweepToTopEnv = make_env(0)()
    with open("datasets/xmagical/dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    watch_traj(env, dataset['teacher'][str(traj_id)])