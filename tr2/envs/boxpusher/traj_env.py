import os.path as osp
import pickle

import gym
import numpy as np
import sapien.core as sapien
from gym import spaces

from tr2.envs.boxpusher.env import BoxPusherEnv
from tr2.envs.trajectory_env import TrajectoryEnv
from tr2.envs.world_building import add_ball, add_target


class BoxPusherTrajectory(TrajectoryEnv):
    def __init__(
        self,
        trajectories=[],
        max_trajectory_length=100,
        max_ep_len_factor=2,
        trajectories_dataset=None,
        stack_size=1,
        fixed_max_ep_len=None,
        max_trajectory_skip_steps=15,
        dense_obs_only=False,
        max_stray_dist=3e-1,
        give_traj_id=False,
        exclude_target_state=False,
        reward_type="trajectory",
        task_agnostic=True,
        trajectory_sample_skip_steps=0,
        randomize_trajectories=True,
        seed_by_dataset=True,
        early_success=True,
        re_center=False,
        env_rew_weight=0.5,
        planner_cfg = dict(
            planner=None,
            planning_env=None,
            render_plan = False,
            max_plan_length=64,
            save_plan_videos=False,
            min_student_execute_length=64,
            re_center=False
        ),
        raw_obs_only=False,
        sub_goals=False,
        silo_window_mask=-1,
        **kwargs,
    ):
        """
        Parameters
        ----------
        dense_obs_only : bool
            If true, will set observation space to a flat vector of values, which includ only the current state
        fixed_max_ep_len : int
            max episode length
        task_agnostic : bool
            If true, task success is determined by whether agent matches teacher's trajectory, not the environments original success signal
        """
        # assert len(trajectories) > 0
        self.exclude_target_state = exclude_target_state
        self.reward_type = reward_type
        self.teacher_balls = []
        self.goal_balls = []
        self.goal_balls_idx = 0
        self.plan_id = 0
        self.re_center = re_center
        self.env_rew_weight = env_rew_weight
        self.last_center = np.zeros(2)
        env = BoxPusherEnv(obs_mode="dense", reward_type="sparse", **kwargs, disable_ball_removal=task_agnostic)

        if trajectories_dataset is not None:
            if osp.splitext(trajectories_dataset)[1] == ".npy":
                raw_dataset  = np.load(trajectories_dataset, allow_pickle=True).reshape(1)[0]
            elif osp.splitext(trajectories_dataset)[1] == ".pkl":
                with open(trajectories_dataset, "rb") as f:
                    raw_dataset = pickle.load(f)
            self.raw_dataset = raw_dataset

        super().__init__(
            env=env,
            state_dims=(4 if exclude_target_state else 6) + (4 if sub_goals else 0),
            act_dims=2,
            teacher_dims=4,
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
            seed_by_dataset=seed_by_dataset,
            early_success=early_success,
            sub_goals=sub_goals,
            raw_obs_only=raw_obs_only,
            silo_window_mask=silo_window_mask
        )
        self.orig_observation_space = self.env.observation_space
        self.dense_obs_only = dense_obs_only
        if (self.dense_obs_only):
            size = 7
            if self.exclude_target_state:
                size = 5
            if self.give_traj_id:
                # hack to test one hot encoding
                size += 4
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=[size],
            )

        self.max_stray_dist = max_stray_dist

    def reset(self, **kwargs):
        self.clear_past_obs()
        obs = super().reset(**kwargs)
        self.target_loc = self.env.target_locs[self.env.target]
        return obs

    def format_ret(self, obs, ret):
        if self.dense_obs_only:
            dense_obs = obs
        else:
            dense_obs = obs["observation"]
        if self.exclude_target_state:
            agent_xy = dense_obs[:2]
            ball_xy = dense_obs[2:4]
        else:
            target_xy = dense_obs[:2]
            agent_xy = dense_obs[2:4]
            ball_xy = dense_obs[4:6]
        if self.reward_type == "lcs_dense":
            reward = 0
            control_to_ball = np.linalg.norm(
                agent_xy - ball_xy
            )
            ball_to_target = np.linalg.norm(self.target_loc - ball_xy)
            ret = -control_to_ball*1*0.1 - ball_to_target*1*0.9 # about -0.6 at start usually
            # if ball_to_target < self.env.target_radius + self.env.ball_radius:
            #     ret += 1 # task complete signal
            reward += ret * self.env_rew_weight
            if self.improved_farthest_traj_step:
                look_ahead_idx = int(min(self.farthest_traj_step, len(self.weighted_traj_diffs) - 1))
                dist_to_next = self.weighted_traj_diffs[look_ahead_idx]
                prog_frac = self.farthest_traj_step / (len(self.weighted_traj_diffs) - 1)
                reward += (10 + 50*prog_frac) * (1 - np.tanh(dist_to_next*10))
            else:
                if self.farthest_traj_step < len(self.weighted_traj_diffs) - 1:
                    reward -= 0 # add step wise penalty if agent hasn't reached the end yet
                else:
                    dist_to_next = self.weighted_traj_diffs[-1]
                    reward += 5 * (1- np.tanh(dist_to_next*10)) # add big reward if agent did reach end

            return reward
        elif self.reward_type == "dense":
            # copied reward from original env
            reward = 0
            control_to_ball = np.linalg.norm(
                agent_xy - ball_xy
            )
            ball_to_target = np.linalg.norm(self.target_loc - ball_xy)
            reward = -control_to_ball*1*0.1 - ball_to_target*1*0.9 
            return reward
        elif self.reward_type == "sparse":
            reward = -1
            return reward
        elif self.reward_type == "test":
            return -np.linalg.norm(agent_xy - np.array([0.5,-0.5])) / 10


    
    def format_obs(self, obs):
        obs, _ = super().format_obs(obs)
        dense_obs = obs["observation"]
        target_xy = dense_obs[:2]
        agent_xy = dense_obs[2:4]
        ball_xy = dense_obs[4:6]

        teacher_agent = self.orig_trajectory_observations[:, :2]
        teacher_world = self.orig_trajectory_observations[:, 2:4]

        self.teacher_student_agent_diff = np.linalg.norm(
            teacher_agent - agent_xy, axis=1
        )
        # distances from world state (just one ball atm) to the intended trajectory
        self.teacher_student_world_diff = np.linalg.norm(
            teacher_world - ball_xy, axis=1
        )

        control_to_ball = np.linalg.norm(
            agent_xy - ball_xy
        )

        agent_within_dist = self.teacher_student_agent_diff < 0.4
        world_within_dist = self.teacher_student_world_diff < 0.1
        if self.env.task == 'silo_obstacle_push':
            agent_within_dist = self.teacher_student_agent_diff < 0.4
            world_within_dist = self.teacher_student_world_diff < 0.3
        
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
                self.farthest_traj_step = max(closest_traj_step, self.farthest_traj_step)

        if self.exclude_target_state:
            obs["observation"] = obs["observation"][2:]
        obs["observation"][2:4] = obs["observation"][2:4] - self.last_center
        obs["observation"][:2] = obs["observation"][:2] - self.last_center
        if self.dense_obs_only:
            obs = np.concatenate([obs["observation"]])
        if self.dense_obs_only and self.give_traj_id:
            # HACK for one hot encodes
            ohe = np.zeros(4)
            teacher_type = self.curr_traj_id // 1000
            ohe[teacher_type] = 1
            obs = np.concatenate([obs, ohe])
        return obs, {}

    def next_sub_goal(self):
        return self.orig_trajectory_observations[min(self.farthest_traj_step + 15, len(self.orig_trajectory_observations) - 1)]

    def get_trajectory(self, t_idx):
        trajectory = self.raw_dataset["teacher"][str(t_idx)]

        trajectory["observations"] #= trajectory["observations"][:-1]
        trajectory["observations"][-1][2:] = trajectory["env_init_state"][:2]
        # add fix by replacing last frame with the true goal location as planner doesn't go to the exact spot all the time.
        return trajectory
    def get_state(self):
        return self.env._get_state()
    def reset_env(self, seed=None, **kwargs):
        if seed is None:
            self.env.reset()
            return
        self.env.seed(seed)
        self.env.reset()
    def reset_to_start_of_trajectory(self):
        self.env._set_state(self.trajectory["env_init_state"])
    def render_goal(self, g1, g2):
        # for HAC code
        # if self.goal_balls_idx == 0:
            # for ball in self.goal_balls:
            #     self.env._scene.remove_actor(ball)
        for i, o in enumerate([g1]):
            frac = 1
            i = self.goal_balls_idx + i
            if self.goal_balls_idx > 0:
                pose = sapien.Pose([o[0], o[1], 0])
                self.goal_balls[0].set_pose(pose)
                pose = sapien.Pose([o[2], o[3], 0])
                self.goal_balls[1].set_pose(pose)
                continue

            pose = sapien.Pose([o[0], o[1], 0])
            ball = add_target(
                self.env._scene,
                pose=pose,
                radius=0.025,
                color=[(14/255)*frac,(14/255)*frac,frac*32/255],
                target_id=f"shadow_goal_agent_{i}",
            )
            self.goal_balls.append(ball)
            pose = sapien.Pose([o[2], o[3], 0])
            ball = add_target(
                self.env._scene,
                pose=pose,
                radius=0.025,
                color=[(232/255)*frac,(233/255)*frac,frac*32/255],
                target_id=f"shadow_goal_box_{i}",
            )
            self.goal_balls.append(ball)
            self.goal_balls_idx += 2
    def draw_teacher_trajectory(self, skip=4):
        for ball in self.teacher_balls:
            self.env._scene.remove_actor(ball)
        self.teacher_balls = []
        obs = self.orig_trajectory_observations[::skip+1]
        obs = np.vstack([obs, self.orig_trajectory_observations[-1]])
        for i, o in enumerate(obs):
            frac = i/len(obs)
            
            pose = sapien.Pose([o[0], o[1], 0])
            ball = add_target(
                self.env._scene,
                pose=pose,
                radius=0.025,
                color=[(51/255)*frac,(159/255)*frac,frac*214/255],
                target_id=f"shadow_{i}",
            )
            self.teacher_balls.append(ball)
            pose = sapien.Pose([o[2], o[3], 0])
            ball = add_target(
                self.env._scene,
                pose=pose,
                radius=0.025,
                color=[(251/255)*frac,(15/255)*frac,frac*214/255],
                target_id=f"shadow_box_{i}",
            )
            self.teacher_balls.append(ball)
    def _plan_trajectory(self, start_state, start_obs):
        self.env: BoxPusherEnv
        self.clear_past_obs()
        self.planning_env.reset()
        self.planning_env._set_state(start_state)
        obs = self.planning_env._get_obs()
        done = False,
        if self.render_plan:
            self.planning_env.render()
            viewer = self.planning_env.viewer
            viewer.paused=True
        
        center = obs["agent_ball"]
        if self.re_center:
            self.last_center = center
        if not self.planner_cfg["re_center"]:
            center = np.zeros_like(center)
        def teacher_obs(obs):
            return np.hstack([
                obs["agent_ball"] - center,
                obs["target_ball"] - center
            ])
        new_init_state = None
        if self.env.task == 'train':
            positions = start_state
        else:
            positions = start_state["positions"]
        positions[:2] -= center
        positions[2:4] -= center
        positions[4:6] -= center
        if self.env.task == 'train':
            new_init_state = positions
        else:
            obstacles = start_state["obstacles"]
            for i in range(len(obstacles)):
                obstacles[i] -= center
            new_init_state = dict(obstacles=obstacles, positions=positions)
        observations = [teacher_obs(obs)]
        imgs = []
        for i in range(self.max_plan_length):
            if self.render_plan:
                self.planning_env.render()
            if self.planner_cfg["save_plan_videos"]:
                img = self.planning_env.render(mode="rgb_array")
                imgs.append(img)
            a, cutoff = self.planner.act(obs)
            obs, reward, done, info = self.planning_env.step(a)
            observations.append(teacher_obs(obs))
            if cutoff:
                break # planner says to stop, then stop
        observations = np.stack(observations).copy()
        from tr2.utils.sampling import resample_teacher_trajectory
        observations = resample_teacher_trajectory(observations, 0.1)
        if self.planner_cfg["save_plan_videos"]:
            print("animate")
            from tr2.utils.animate import animate
            animate(imgs, filename=f"plan_{self.plan_id}.mp4", _return=False, fps=24)
        self.plan_id += 1

        observations[-1][2:] = positions[:2]  
        return {
            "observations": observations,
            "env_init_state": new_init_state
        }
gym.register(
    id="BoxPusherTrajectory-v0",
    entry_point="tr2.envs.boxpusher.traj_env:BoxPusherTrajectory",
)