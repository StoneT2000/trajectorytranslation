import os
from pathlib import Path
import pickle

import gym
import numpy as np
import sapien.core as sapien
import tqdm
import os.path as osp
import tr2.envs

from tqdm import tqdm
from tr2.planner.boxpusherteacher import BoxPusherReacherPlanner, BoxPusherTaskPlanner
from paper_rl.common.rollout import Rollout
from stable_baselines3.common.vec_env import SubprocVecEnv

from tr2.utils.sampling import resample_teacher_trajectory

save_folder = osp.join("./datasets", "boxpusher")
# save_folder = osp.join("./data", "trajectories")
if not osp.isdir(save_folder):
    os.mkdir(save_folder)
    
if __name__ == "__main__":
    
    cpu = 4
    def make_env(idx):
            def _init():
                import tr2.envs.boxpusher.traj_env
                env = gym.make("BoxPusher-v0", obs_mode="dict", balls=1, magic_control=True)
                env.seed(idx)
                return env
            return _init

    env = SubprocVecEnv([make_env(i) for i in range(cpu)])
    all_trajs = []
    pbar = tqdm(range(4000))
    
    for agent_type in range(4):
        agents = []
        for _ in range(cpu):
            agent = BoxPusherTaskPlanner()
            agent.set_type(agent_type)
            agents.append(agent)
        for traj_id in range(0, 1000, cpu):
            rollout = Rollout()
            for x in range(cpu):
                env.env_method("seed", traj_id+x, indices=[x])
            env.reset()
            def policy(obs):
                acts = []
                for idx in range(cpu):
                    o = {}
                    for k in obs.keys():
                        o[k] = obs[k][idx]
                    a, _ = agents[idx].act(o)
                    acts.append(a)
                acts = np.stack(acts)
                return acts
            trajs = rollout.collect_trajectories(policy=policy, env=env, n_envs=cpu, n_trajectories=cpu, render=False, pbar=False, even_num_traj_per_env=True)
            for t in trajs:
                all_trajs += t
            pbar.update(cpu)
    print(f"Trajectories collected, now saving {len(all_trajs)} of them...")
    dataset = dict(teacher=dict())
    for idx, traj in tqdm(enumerate(all_trajs), total=len(all_trajs)):
        
        N = len(traj["observations"])
        new_obs = np.zeros((N, 4))
        for i in range(N):
            o = traj["observations"][i]
            a_xy = o["agent_ball"]
            b_xy = o["target_ball"]
            # t_xy = o["target"]
            o = np.stack([a_xy, b_xy]).flatten()
            new_obs[i] = o
        env_init_obs = traj["observations"][0]
        env_init_state = np.stack([env_init_obs["target"], env_init_obs["agent_ball"], env_init_obs["target_ball"]]).flatten()
        
        new_obs = resample_teacher_trajectory(new_obs, max_dist=0.1)
        dataset["teacher"][str(idx)] = dict(
            observations=new_obs,
            env_init_state=env_init_state
        )
    Path("datasets/boxpusher_v2").mkdir(parents=True, exist_ok=True)
    with open("datasets/boxpusher_v2/dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)
    ids =sorted(list(dataset["teacher"].keys()))
    np.save("datasets/boxpusher_v2/dataset_train_ids.npy", ids)