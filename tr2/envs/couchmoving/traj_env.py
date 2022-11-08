import os.path as osp
import pickle

import gym
import numpy as np
import sapien.core as sapien
from gym import spaces
from matplotlib import pyplot as plt

from tr2.envs.boxpusher.env import BoxPusherEnv
from tr2.envs.couchmoving.building import add_walls
from tr2.envs.couchmoving.env import CouchmovingEnv
from tr2.envs.trajectory_env import TrajectoryEnv
from tr2.envs.world_building import (add_ball, add_target,
                                                  create_box)


class CouchmovingTrajectory(TrajectoryEnv):
    def __init__(
        self,
        trajectories=[],
        max_trajectory_length=100,
        max_ep_len_factor=2,
        trajectories_dataset=None,
        stack_size=1,
        fixed_max_ep_len=None,
        max_trajectory_skip_steps=15,
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
        sub_goals = False,
        sub_goal_nstep = 10,
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
        # assert len(trajectories) > 0
        self.attn_blocks = []
        self.reward_type = reward_type
        self.teacher_balls = []
        self.plan_id = 0
        self.sub_goal_nstep = sub_goal_nstep
        self.env_rew_weight = env_rew_weight
        reward_type = "dense"
        env = CouchmovingEnv(obs_mode="dense", agent_type="couch",exclude_target_state=exclude_target_state,skip_map_gen=True, reward_type=reward_type, **kwargs)
        if trajectories_dataset is not None:
            if osp.splitext(trajectories_dataset)[1] == ".npy":
                raw_dataset  = np.load(trajectories_dataset, allow_pickle=True).reshape(1)[0]
            elif osp.splitext(trajectories_dataset)[1] == ".pkl":
                with open(trajectories_dataset, "rb") as f:
                    raw_dataset = pickle.load(f)
            self.raw_dataset = raw_dataset
        state_dims = 13
        if not exclude_target_state:
            state_dims = 15
        super().__init__(
            env=env,
            state_dims=state_dims + 2 + (2 if sub_goals else 0),
            act_dims=3,
            teacher_dims=2,
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
            planner_cfg=planner_cfg,
            sub_goals=sub_goals
        )
        self.orig_observation_space = self.env.observation_space
        self.max_stray_dist = max_stray_dist

    def format_ret(self, obs, ret):
        if self.reward_type == "trajectory":
            reward = 0
            reward += ret * self.env_rew_weight
            if self.improved_farthest_traj_step:
                look_ahead_idx = int(min(self.farthest_traj_step, len(self.teacher_student_coord_diff) - 1))
                prog_frac = self.farthest_traj_step / (len(self.teacher_student_coord_diff) - 1)
                dist_to_next = self.weighted_traj_diffs[look_ahead_idx]
                reward += (10 + 50*prog_frac) * (1 - np.tanh(dist_to_next*30))
            else:
                if self.farthest_traj_step < len(self.teacher_student_coord_diff) - 4:
                    # checks len(...) - 4 due to bug of the environment where agent can actually never match the last state due to morphology.
                    pass
                else:
                    dist_to_next = self.weighted_traj_diffs[-1]
                    reward += 10 * (1- np.tanh(dist_to_next*30)) # add big reward if agent did reach end and encourage agent to stay close to end
            return reward
        elif self.reward_type == "dense":
            # copied reward from original env
            return ret
        elif self.reward_type == "sparse":
            reward = -1
            return reward


    
    def format_obs(self, obs):
        obs, _ = super().format_obs(obs)
        obs["observation"] = obs["observation"][:]
        dense_obs = obs["observation"]
        agent_xy = dense_obs[:2]
        
        teacher_agent = self.orig_trajectory_observations[:, :2]
        self.teacher_student_coord_diff = np.linalg.norm(
            teacher_agent - agent_xy, axis=1
        )
        
        agent_within_dist = self.teacher_student_coord_diff < 0.3
        idxs = np.where(agent_within_dist)[0] # determine which teacher frames are within distance

        self.weighted_traj_diffs = self.teacher_student_coord_diff
        self.closest_traj_step = 0
        self.improved_farthest_traj_step = False
        if len(idxs) > 0:
            closest_traj_step = idxs[(self.weighted_traj_diffs)[idxs].argmin()]
            if closest_traj_step - self.farthest_traj_step < 10:
                self.closest_traj_step = closest_traj_step
                if closest_traj_step > self.farthest_traj_step:
                    self.improved_farthest_traj_step = True
                    self.match_traj_count += 1
                self.farthest_traj_step = max(closest_traj_step, self.farthest_traj_step)
        N = len(self.orig_trajectory_observations)
        dir_to_goal = self.orig_trajectory_observations[min(self.closest_traj_step + 1, N - 1)] - self.orig_trajectory_observations[min(self.closest_traj_step, N - 2)]
        obs["observation"] = np.hstack([obs["observation"], dir_to_goal])
        return obs, {
            "stats": {
                "match_traj_count": self.match_traj_count,
                "match_traj_frac": self.match_traj_count / (len(self.weighted_traj_diffs) - 1)
            }
        }
    def next_sub_goal(self):
        return self.orig_trajectory_observations[min(self.farthest_traj_step + self.sub_goal_nstep, len(self.orig_trajectory_observations) - 1)]

    def get_trajectory(self, t_idx):
        trajectory = self.raw_dataset["teacher"][t_idx]
        trajectory["observations"] = trajectory["observations"]
        return trajectory
    def reset_env(self, **kwargs):
        pass
    def reset_to_start_of_trajectory(self):
        env: CouchmovingEnv = self.env
        self.env.seed(self.trajectory["seed"])
        if "env_cfg" in self.trajectory:
            env.max_walks = self.trajectory["env_cfg"]["max_walks"]
            env.walk_dist_range = self.trajectory["env_cfg"]["walk_dist_range"]
            env.world_size = self.trajectory["env_cfg"]["world_size"]
        self.env.reset()
        self.env._set_state(self.trajectory["env_init_state"])
    def seed(self, seed, *args, **kwargs):
        from gym.utils import seeding
        self.np_random, seed = seeding.np_random(seed)

    def draw_teacher_trajectory(self, skip=4):
        for ball in self.teacher_balls:
            self.env._scene.remove_actor(ball)
        self.teacher_balls = []
        world_size_scale = 50/ self.env.world_size
        obs = self.orig_trajectory_observations[:-1:skip+1]
        obs = np.vstack([obs, self.orig_trajectory_observations[-1]])
        for i, o in enumerate(obs):
            frac = i/len(obs)
            pose = sapien.Pose([o[0], o[1], 0])
            ball = add_target(
                self.env._scene,
                pose=pose,
                radius=0.01 * world_size_scale,
                color=[(159/255)*frac,(51/255)*frac,frac*214/255],
                target_id=f"shadow_{i}",
            )
            self.teacher_balls.append(ball)
    def _plan_trajectory(self, start_state):
        pass
gym.register(
    id="CouchmovingTrajectory-v0",
    entry_point="tr2.envs.couchmoving.traj_env:CouchmovingTrajectory",
)