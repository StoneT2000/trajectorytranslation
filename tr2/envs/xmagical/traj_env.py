import os.path as osp
import pickle
import time

import gym
import numpy as np
import sapien.core as sapien
from gym import spaces
from matplotlib import pyplot as plt

from tr2.envs.trajectory_env import TrajectoryEnv
from tr2.envs.xmagical.env import SweepToTopEnv
from tr2.envs.xmagical.register_envs import register_envs

register_envs()

class XMagicalTrajectory(TrajectoryEnv):
    def __init__(
        self,
        trajectories=[],
        max_trajectory_length=100,
        max_ep_len_factor=2,
        trajectories_dataset=None,
        stack_size=1,
        fixed_max_ep_len=None,
        max_trajectory_skip_steps=999,
        max_stray_dist=3e-1,
        give_traj_id=False,
        reward_type="lcs_dense",
        task_agnostic=True,
        trajectory_sample_skip_steps=0,
        randomize_trajectories=True,
        early_success=False,
        exclude_target_state=True,
        env_rew_weight=0.2,
        planner_cfg = dict(
            planner=None,
            planning_env=None,
            render_plan = False,
            max_plan_length=64,
            save_plan_videos=False
        ),
        embodiment = "Shortstick",
        **kwargs,
    ):
        """
        Parameters
        ----------
        fixed_max_ep_len : int
            max episode length
        task_agnostic : bool
            If true, task success is determined by whether agent matches teacher's trajectory, not the environments original success signal
        """
        self.reward_type = reward_type
        self.plan_id = 0
        self.match_traj_count = 0
        self.env_rew_weight = env_rew_weight
        env = gym.make(f'SweepToTop-{embodiment}-State-Allo-TestLayout-v0')
        if trajectories_dataset is not None:
            if osp.splitext(trajectories_dataset)[1] == ".pkl":
                with open(trajectories_dataset, "rb") as f:
                    raw_dataset = pickle.load(f)
            self.raw_dataset = raw_dataset
        state_dims = 16
        act_dims = 2
        if embodiment == "Gripper":
            act_dims = 3
            state_dims = 17
        super().__init__(
            env=env,
            state_dims=state_dims,
            act_dims=act_dims,
            teacher_dims=2 + 2 * 3, # x,y of agent and 3 debris
            early_success=early_success,
            trajectories=trajectories,
            max_trajectory_length=max_trajectory_length,
            trajectories_dataset=trajectories_dataset,
            max_ep_len_factor=max_ep_len_factor,
            stack_size=stack_size,
            fixed_max_ep_len=fixed_max_ep_len,
            max_trajectory_skip_steps=max_trajectory_skip_steps,
            give_traj_id=give_traj_id,
            trajectory_sample_skip_steps=trajectory_sample_skip_steps,
            task_agnostic=task_agnostic,
            randomize_trajectories=randomize_trajectories,
            planner_cfg=planner_cfg
        )
        self.orig_observation_space = self.env.observation_space
        self.max_stray_dist = max_stray_dist

    def format_ret(self, obs, ret):
        if self.reward_type == "lcs_dense" or self.reward_type == 'lcs_dense2':
            reward = 0
            # if "dense" in self.reward_type:
            #     ret = (1 - np.tanh(-self.env.dense_reward()))
            reward += ret * self.env_rew_weight
            if self.improved_farthest_traj_step:
                look_ahead_idx = int(min(self.farthest_traj_step, len(self.weighted_traj_diffs) - 1))
                prog_frac = self.farthest_traj_step / (len(self.weighted_traj_diffs) - 1)
                dist_to_next = self.weighted_traj_diffs[look_ahead_idx]
                reward += (10 + 50*prog_frac) * (1 - np.tanh(dist_to_next*30))
            else:
                if self.farthest_traj_step < len(self.weighted_traj_diffs) - 1:
                    pass
                else:
                    dist_to_next = self.weighted_traj_diffs[-1]
                    reward += 10 * (1- np.tanh(dist_to_next*30)) # add big reward if agent did reach end and encourage agent to stay close to end
            return reward
    
    def format_obs(self, obs):
        obs, _ = super().format_obs(obs)
        dense_obs = obs["observation"]
        agent_xy = dense_obs[:2]
        debris_xy = dense_obs[2:8]
        teacher_agent = self.orig_trajectory_observations[:, :2]
        teacher_world = self.orig_trajectory_observations[:, 2:]
        self.teacher_student_agent_diff = np.linalg.norm(
            teacher_agent - agent_xy, axis=1
        )
        if self.reward_type == 'lcs_dense2':
            done_debris = self.env.get_done_debris()
            teacher_world_not_done = []
            debris_xy_not_done = []
            for i in range(len(done_debris)):
                if done_debris[i]:
                    debris_xy_not_done += [teacher_world[-1][i*2: (i+1) * 2]]
                else:
                    debris_xy_not_done += [debris_xy[i*2: (i+1) * 2]]
            debris_xy_not_done = np.stack(debris_xy_not_done).ravel()
            self.teacher_student_world_diff = np.linalg.norm(
                teacher_world - debris_xy_not_done, axis=1
            )
        else:
            self.teacher_student_world_diff = np.linalg.norm(
                teacher_world - debris_xy, axis=1
            )



        agent_within_dist = self.teacher_student_agent_diff < 0.4
        world_within_dist = self.teacher_student_world_diff < 0.2
        
        idxs = np.where(world_within_dist & agent_within_dist)[0] # determine which teacher frames are within distance

        self.weighted_traj_diffs = self.teacher_student_agent_diff * 0.1 + self.teacher_student_world_diff * 0.9
        self.closest_traj_step = 0
        self.improved_farthest_traj_step = False
        if len(idxs) > 0:
            closest_traj_step = idxs[(self.weighted_traj_diffs)[idxs].argmin()]
            if closest_traj_step - self.farthest_traj_step < self.max_trajectory_skip_steps:
                self.closest_traj_step = closest_traj_step
                if closest_traj_step > self.farthest_traj_step:
                    self.improved_farthest_traj_step = True
                    self.match_traj_count += 1
                self.farthest_traj_step = max(closest_traj_step, self.farthest_traj_step)
        return obs, {
            "stats": {
                "match_traj_count": self.match_traj_count,
                "match_traj_frac": self.match_traj_count / (len(self.weighted_traj_diffs) - 1)
            }
        }
    def get_trajectory(self, t_idx):
        trajectory = self.raw_dataset["teacher"][str(t_idx)]
        return trajectory
    def reset_env(self, seed=None, **kwargs):
        if seed is None:
            self.env.reset()
            return
        self.env.seed(seed)
        self.env.reset()
    def reset_to_start_of_trajectory(self):
        env: SweepToTopEnv = self.env
        env.reset()
        env.set_state_xy(self.trajectory["env_init_state"])
    def get_obs(self):
        return self.env.get_state()
gym.register(
    id="XMagicalTrajectory-v0",
    entry_point="skilltranslation.envs.xmagical.traj_env:XMagicalTrajectory",
)
if __name__ == "__main__":
    env = XMagicalTrajectory(trajectories_dataset="datasets/xmagical/dataset.pkl", trajectories=[1])
    env.seed(0)
    env.reset()
    while True:
        a = env.action_space.sample()
        o,r,d,i=env.step(a)
        time.sleep(0.05)
        env.render()
        print(r)