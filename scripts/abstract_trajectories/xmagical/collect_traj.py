import os
from pathlib import Path
import pickle

import gym
import numpy as np
import sapien.core as sapien
import tqdm
import os.path as osp
import skilltranslation.envs

from tqdm import tqdm
from skilltranslation.planner.boxpusherteacher import BoxPusherReacherPlanner, BoxPusherTaskPlanner
from paper_rl.common.rollout import Rollout
from stable_baselines3.common.vec_env import SubprocVecEnv

from skilltranslation.utils.sampling import resample_teacher_trajectory
from skilltranslation.envs.xmagical.register_envs import register_envs
from skilltranslation.envs.xmagical.env import SweepToTopEnv
register_envs()
save_folder = osp.join("./datasets", "xmagical")
if not osp.isdir(save_folder):
    os.mkdir(save_folder)
def get_abstract_obs(obs):
    return obs[:8]
def get_traj(env, render):
    obs = env.reset()
    env_init_state = obs
    traj_obs = []
    debrisorder = np.arange(3)
    np.random.shuffle(debrisorder)
    for debris in debrisorder:
        grasped_debris = None
        subdone = False
        while not subdone:
            # if render: time.sleep(0.03)
            obs = env.get_state()

            dist_to_goal = obs[13 + debris]
            if dist_to_goal < 0.02:
                subdone = True
                break
            traj_obs += [get_abstract_obs(obs)]
            agent_xy = obs[:2]
            debris_xy = obs[2 + 2*debris : 2 + 2 * (debris + 1)]
            if grasped_debris is None:
                # move to grasp debris
                agent_debris_delta = debris_xy - agent_xy
                if np.linalg.norm(agent_debris_delta) < 3e-2:
                    grasped_debris = debris
                move_delta = (agent_debris_delta / np.linalg.norm(agent_debris_delta)) * speed
                
            else:
                # once grasping, move debris to the goal by going up
                move_delta = np.array([0, speed])
            new_agent_xy = agent_xy + move_delta
            obs[:2] = new_agent_xy
            if grasped_debris is not None:
                new_debris_xy = debris_xy + move_delta
                obs[2 + 2*debris : 2 + 2* (debris+1)] = new_debris_xy
            env.set_state_xy(obs)
            if render: env.render()
        subdone = False
        while not subdone:
            obs = env.get_state()
            agent_xy = obs[:2]
            dist_to_goal = np.abs(agent_xy[1] + 0.5)
            if dist_to_goal < 0.02:
                subdone = True
                break
            traj_obs += [get_abstract_obs(obs)]
            move_delta = np.array([0, -speed])
            new_agent_xy = agent_xy + move_delta
            obs[:2] = new_agent_xy
            env.set_state_xy(obs)
            if render: env.render()
    traj_obs += [get_abstract_obs(env.get_state())]
    traj = {"observations": traj_obs, "env_init_state": env_init_state}
    return traj
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
    render = True
    N = 4000

    env: SweepToTopEnv = make_env(0)()
    
    done = False
    import time
    np.random.seed(0)


    dataset = {'teacher': dict()}
    for i in tqdm(range(N)):
        env.seed(i)
        traj = get_traj(env, render=render)
        new_obs = traj['observations']
        new_obs = resample_teacher_trajectory(new_obs, max_dist=0.1)
        traj['observations'] = new_obs
        dataset['teacher'][str(i)] = traj

    Path("datasets/xmagical").mkdir(parents=True, exist_ok=True)
    with open("datasets/xmagical/dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)
    ids = sorted(list(dataset["teacher"].keys()))
    np.save("datasets/xmagical/dataset_train_ids.npy", ids)