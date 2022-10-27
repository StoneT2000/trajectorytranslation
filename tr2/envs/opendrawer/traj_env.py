import os.path as osp
import pickle
from tracemalloc import start

import gym
import numpy as np
from mani_skill.env.open_cabinet_door_drawer import (
    OpenCabinetDrawerEnv_CabinetSelection,
    OpenCabinetDrawerMagicEnv_CabinetSelection)

from tr2.envs.trajectory_env import TrajectoryEnv
from tr2.planner.opendrawerplanner import OpenDrawerPlanner
from tr2.utils.sampling import resample_teacher_trajectory

# from mani_skill import register_gym_env
try:
    import mani_skill.env
    from mani_skill.env.open_cabinet_door_drawer import \
        OpenCabinetDrawerMagicEnv
except:
    print( "#" * 15, "no Maniskill 1", "#" * 15,)
import sapien.core as sapien

from tr2.envs.world_building import add_target


# @register_gym_env("OpenDrawerTrajectory-v0")
class OpenDrawerTrajectory(TrajectoryEnv):
    def __init__(
        self,
        trajectories=[],
        max_trajectory_length=100,
        max_ep_len_factor=2,
        trajectories_dataset=None,
        stack_size=1,
        fixed_max_ep_len=200,
        max_trajectory_skip_steps=15,
        give_traj_id=False,
        reward_type="original",
        task_agnostic=True,
        trajectory_sample_skip_steps=0,
        randomize_trajectories=True,
        max_world_state_stray_dist=0.1,
        max_coord_stray_dist=0.1,
        early_success=False,
        controller="arm",
        env_rew_weight=0,
        partial_trajectories=False,
        obs_mode='state',
        max_plans=-1,
        mode=1,
        planner_cfg = dict(
            planner=None,
            planning_env=None,
            render_plan = False,
            max_plan_length=64,
            save_plan_videos=False,
            min_student_execute_length=64,
            re_center=False
        ),
        sub_goals=False,
        **kwargs,
    ):
        self.reward_type = reward_type
        self.max_world_state_stray_dist = max_world_state_stray_dist
        self.max_coord_stray_dist = max_coord_stray_dist
        self.plan_id = 0
        self.partial_trajectories = partial_trajectories
        self.obs_mode = obs_mode
        env = gym.make("OpenCabinetDrawer_CabinetSelection-v0", max_episode_steps=fixed_max_ep_len + 100, mode=mode)
        env.set_env_mode(obs_mode=obs_mode, reward_type='dense')
        self.grasp_count = 0
        self.match_traj_count = 0
        self.match_traj_frac = 0
        self.env_rew_weight=env_rew_weight
        if trajectories_dataset is not None:
            with open(trajectories_dataset, "rb") as f:
                raw_dataset = pickle.load(f)
            self.raw_dataset = raw_dataset
        state_dims=39
        teacher_dims=9
        if obs_mode == 'pointcloud':
            state_dims = 33 + 60*3 # remove BBOX info and add in PCD data
            teacher_dims = 3 + 3
        super().__init__(
            env=env,
            state_dims=state_dims + (teacher_dims if sub_goals else 0), #[9] [9] [7] [3] [4] [3] [3]
            act_dims=env.action_space.shape[0],
            teacher_dims=teacher_dims, #[5] [5] [7]
            trajectories=trajectories,
            max_trajectory_length=max_trajectory_length,
            trajectories_dataset=trajectories_dataset,
            max_ep_len_factor=max_ep_len_factor,
            stack_size=stack_size,
            fixed_max_ep_len=fixed_max_ep_len,
            give_traj_id=give_traj_id,
            trajectory_sample_skip_steps=trajectory_sample_skip_steps,
            task_agnostic=task_agnostic,
            randomize_trajectories=randomize_trajectories,
            max_trajectory_skip_steps=max_trajectory_skip_steps,
            planner_cfg=planner_cfg,
            early_success=early_success,
            max_plans=max_plans,
            sub_goals=sub_goals
        )
        self.orig_observation_space = self.env.observation_space
        self.teacher_balls = []
    def reset_env(self, seed=None, **kwargs):
        if seed is None:
            self.env.reset()
            return
        self.env.seed(seed)
        self.env.reset()
    def reset(self, *args, **kwargs):
        self.grasp_count = 0
        self.match_traj_count = 0
        self.match_traj_frac = 0
        super_reset = super().reset(*args, **kwargs)
        return super_reset

    def format_ret(self, obs, ret):
        if self.reward_type == "original" or self.reward_type == "dense":
            return ret
        elif self.reward_type == "trajectory":
            reward = 0
            reward += ret * self.env_rew_weight
            if self.improved_farthest_traj_step:
                look_ahead_idx = int(min(self.farthest_traj_step, len(self.teacher_student_coord_diff) - 1))
                prog_frac = self.farthest_traj_step / (len(self.teacher_student_coord_diff) - 1)
                dist_to_next = self.weighted_traj_diffs[look_ahead_idx]
                reward += (10 + 50*prog_frac) * (1 - np.tanh(dist_to_next*30))
            else:
                if self.farthest_traj_step < len(self.teacher_student_coord_diff) - 1:
                    pass
                else:
                    dist_to_next = self.weighted_traj_diffs[-1]
                    reward += 10 * (1- np.tanh(dist_to_next*30)) # add big reward if agent did reach end and encourage agent to stay close to end
            return reward
        else:
            raise NotImplementedError()


    def format_obs(self, obs):
        obs, _ =super().format_obs(obs)
        dense_obs = obs['observation']
        # 13 qpos, 13 qvel, 7 hand pose, 6D bbox min and max
        if self.obs_mode == 'pointcloud':
            dense_obs = obs['observation']['agent']
            agent_obs = dense_obs[:33]
            obs_pcd = obs['observation']['pointcloud']['xyz']
            student_world_state = obs_pcd.mean(0)
            student_hand_xyz = agent_obs[26:29]
            # import pdb;pdb.set_trace()
            dense_obs = np.hstack([agent_obs, obs_pcd.flatten()])
            obs["observation"] = dense_obs
            
        else:
            obs['observation'] = dense_obs
            student_world_state = dense_obs[33:39]
            student_hand_xyz = dense_obs[26:29]
        
        
        assert len(student_hand_xyz) == 3
        # assert len(student_world_state) == 6 # TODO switch out for PCDs?
        teacher_agent_xyz = self.orig_trajectory_observations[:, :3]
        if self.obs_mode == 'pointcloud':
            teacher_world_states = self.orig_trajectory_observations[:, 3:]
        else:
            teacher_world_states= self.orig_trajectory_observations[:, -6:]

        self.teacher_student_coord_diff = np.linalg.norm(
            teacher_agent_xyz - student_hand_xyz , axis=1
        )

        self.teacher_student_world_diff = np.linalg.norm(
            teacher_world_states - student_world_state, axis=1
        )
        # measure rot, range from 1 (180 deg diff) to 0 (same)
        grasped = False #self.env.agent.check_grasp(self.env.blocks[0])
        if grasped:
            self.grasp_count += 1


        agent_within_dist = self.teacher_student_coord_diff < self.max_coord_stray_dist
        world_within_dist = self.teacher_student_world_diff < self.max_world_state_stray_dist

        idxs = np.where(world_within_dist & agent_within_dist)[0] # determine which teacher frames are within distance

        self.weighted_traj_diffs = self.teacher_student_coord_diff * 0.1 + self.teacher_student_world_diff * 0.9
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
        stats = {
            "grasp": self.grasp_count,
            "match_traj_count": self.match_traj_count,
            "match_traj_frac": self.match_traj_count / (len(self.weighted_traj_diffs) - 1)
        }
        robot = self.env.agent.robot
        r_qpos = robot.get_qpos()
        r_qpos[3] = min(r_qpos[3], 0.35)
        robot.set_qpos(r_qpos)
        if self.env.task == "close":
            stats["farthest_traj_frac_close"] = self.farthest_traj_step / (len(self.weighted_traj_diffs) - 1)
        elif self.env.task == "open":
            stats["farthest_traj_frac_open"] = self.farthest_traj_step / (len(self.weighted_traj_diffs) - 1)
        return obs, {
            "stats": stats
        }
    def next_sub_goal(self):
        return self.orig_trajectory_observations[min(self.farthest_traj_step + 10, len(self.orig_trajectory_observations) - 1)]

    def get_trajectory(self, t_idx):
        trajectory=self.raw_dataset["teacher"][str(t_idx)]
        return trajectory

    def seed(self, seed, *args, **kwargs):
        self.env.seed(seed)
        return super().seed(seed, *args, **kwargs)

    def get_obs(self):
        return self.env.get_obs()
    def get_state(self):
        return self.env.get_state()

    def reset_to_start_of_trajectory(self):
        self.env: OpenCabinetDrawerEnv_CabinetSelection
        cfg = self.trajectory["config"]
        self.env.reset(level=int(cfg['level']), cabinet_id=cfg['cabinet_id'], target_link_id=cfg['target_link_id'], task=cfg['task'])
        self.env.set_state(cfg["env_init_state"])
    def draw_teacher_trajectory(self, skip=4, **kwargs):
        for ball in self.teacher_balls:
            self.env.scene.remove_actor(ball)
        self.teacher_balls = []
        obs = self.orig_trajectory_observations[:-1:skip+1]
        obs = np.vstack([obs, self.orig_trajectory_observations[-1]])
        for i, o in enumerate(obs):
            frac = i/len(obs)
            pose = sapien.Pose([o[0], o[1], o[2]])
            ball = add_target(
                self.env.scene,
                pose=pose,
                radius=0.01,
                color=[(159/255)*frac,(51/255)*frac,frac*214/255],
                target_id=f"shadow_{i}",
            )
            self.teacher_balls.append(ball)
    def _plan_trajectory(self, start_state, start_obs):
        self.clear_past_obs()
        self.planning_env: OpenCabinetDrawerMagicEnv_CabinetSelection
        self.planner: OpenDrawerPlanner
        target_index = self.env.target_index
        # check if we should change target
        print(f"PLANNING plan {self.plan_id}")
        self.plan_id += 1
        for i, open_enough in enumerate(self.env.compute_eval_flag_dict()["open_enoughs"]):
            if not open_enough:
                target_index = i
        self.env.select_target(target_index)
        self.planning_env.reset(level=None, cabinet_id=self.env.cabinet_id, target_link_id=self.env.target_index, task='open')

        # apply mapping function f to low-level state start_obs
        qpos = self.env.cabinet.get_qpos()
        self.planning_env.cabinet.set_qpos(qpos)
        robot = self.planning_env.agent.robot
        robot_qpos = robot.get_qpos()
        robot_qpos[:3] = start_obs[26:29] # gripper pos
        robot.set_qpos(robot_qpos)

        # run high level agent
        observations = self.planner.generate_magic_teacher_traj_for_env(obs=dict(planning_env=self.planning_env), render=self.planner_cfg["render_plan"])
        max_dist = 0.05
        while self._traj_len(dict(observations=observations)) > self.max_trajectory_length:
            observations = resample_teacher_trajectory(observations, max_dist=max_dist)
            max_dist += 1e-2
            if max_dist > 0.2:
                observations = np.zeros((10, 9))
        return {
            "observations": np.array(observations),
        }
gym.register(
    id="OpenDrawerTrajectory-v0",
    entry_point="tr2.envs.opendrawer.traj_env:OpenDrawerTrajectory",
)