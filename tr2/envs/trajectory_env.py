import os.path as osp
from typing import List

import gym
import numpy as np
import sapien.core as sapien
from attr import has
from gym import spaces
from gym.utils import seeding

from tr2.envs.sapien_env import SapienEnv
from tr2.planner.base import HighLevelPlanner


class TrajectoryEnv(gym.Env):
    """
    base class for any abstract-trajectory following environment
    """
    def __init__(
        self,
        env: gym.Env,
        state_dims,
        act_dims,
        teacher_dims,
        trajectories=[],
        max_trajectory_length=100,
        trajectories_dataset=None,
        max_ep_len_factor=2,
        fixed_max_ep_len=None,
        stack_size=1,
        stack_with_past_actions=True,
        max_trajectory_skip_steps=15,
        give_traj_id=False,
        trajectory_sample_skip_steps=0,
        task_agnostic=True,
        randomize_trajectories=True,
        seed_by_dataset=True,
        early_success=True,
        max_plans=-1,
        sub_goals=False,
        planner_cfg = dict(
            planner=None,
            planning_env=None,
            render_plan = False,
            max_plan_length=30,
            min_student_execute_length=30,
            save_plan_videos=False,
            re_center=False,
        ),
        raw_obs_only=False,
        categorical=False,
        skip_long_trajectories=False,
        silo_window_mask=-1,
        **kwargs) -> None:
        """
        Parameters
        ----------
        env: gym.Env
            the env to turn into a trajectory following env. Must inherit gym.Env and implement a _get_obs function and _set_state function
        state_dims: int
            number of dimensions in the env's observation space. Currently only support box observation spaces
        act_dims:
            number of dimensions in the env's action space. Currently only support box action spaces
        trajectories: list[int]
            list of trajectory ids this trajectory env can use
        max_trajectory_length:
            max teacher trajectory length possible. Env will pad the teacher trajectory to the maxlength with 0 vectors
            and also return an appropriate masking vector
        stack_size: int
            number of past frames to stack. stack_size = 1 would be returning the current observation only
        max_trajectory_skip_steps: int
            the max number of steps in the teacher trajectory the student is allowed to skip forward to. 
            Agent skipping too much will not progress and cannot receive an early done signal
        give_traj_id: bool
            whether to return the trajectory id loaded
        trajectory_sample_skip_steps: int
            how many frames in the trajectory to skip before returning it in the observation. 0 means no frames are dropped. 1. means half are dropped.
            always keeps the first and last frame
        task_agnostic: bool
            if true, uses trajectory following success signal. If false, uses given envs success signal
        randomize_trajectories: bool
            if true, selects a random trajectory from the given trajectories to use as the teacher. 
            if false, sequentially goes through each trajectory in order.
        teacher_dims : int
            number of dimensions in teacher observations. default is None, value used will be same as student (state_dims)
        early_success : bool
            whether to return done=True if success is detected

        seed_by_dataset : bool
            if seed_by_dataset is true, then abstract trajectory is pulled from dataset using traj_ids and resetting 
                the environment resets to the initial state of one of the abstract trajectories
            if false, then traj_ids are just environment seeds

        sub_goals : bool
            if True, then observation will additionally include a sub_goal via the next_sub_goal function. If False, then it is not provided

        silo_window_mask : int
            if -1, not used. Otherwise, this si the window size parameter used in selective imitation learning from observations.
            In particular, it will mask out any part of the abstract trajectory that has been matched already and any part that is more than
            silo_window_mask steps far in the future 

        planner related
        planner_cfg

        can provide a custom planner and an associated planner environment (if needed) to perform replanning and execution

        """
        self.env: SapienEnv = env
        
        self.trajectory: np.ndarray = None
        self.trajectory_observations: np.ndarray = None
        self.orig_trajectory_observations: np.ndarray = None
        self.trajectory_attn_mask: np.ndarray = None
        self.trajectory_time_steps: np.ndarray = None
        self.trajectory_sample_skip_steps = trajectory_sample_skip_steps
        self.sub_goals = sub_goals
        self.max_trajectory_length = max_trajectory_length
        self.skip_long_trajectories = skip_long_trajectories
        self.silo_window_mask = silo_window_mask

        self.seed_by_dataset = seed_by_dataset

        self.planner_cfg = planner_cfg
        self.use_planner = planner_cfg["planner"] is not None
        self.total_plans = 0
        self.max_plans = max_plans
        if self.use_planner:
            self.min_student_execute_length = self.planner_cfg["min_student_execute_length"]
            self.max_student_execute_length = self.planner_cfg["max_student_execute_length"]
            self.planner: HighLevelPlanner = planner_cfg["planner"]
            self.planning_env: gym.Env = planner_cfg["planning_env"]
            self.render_plan: bool = planner_cfg["render_plan"]
            self.max_plan_length = planner_cfg["max_plan_length"]
            self.last_plan_step = 0


        self.state_dims = state_dims
        if teacher_dims is not None:
            self.teacher_dims = teacher_dims
        else:
            self.teacher_dims = state_dims
        self.act_dims = act_dims

        self.task_agnostic = task_agnostic
        self.early_success = early_success

        self.trajectory_ids = trajectories
        self.trajectories: List = []
        self.randomize_trajectories = randomize_trajectories
        self.curr_traj_idx = -1
        self.curr_traj_id = -1
        self.trajectories_dataset = trajectories_dataset

        self.current_trajectory_id = -1
        self.match_traj_count = 0
        self.farthest_traj_step = 0 # farthest teacher trajectory step matched so far since reset
        self.closest_traj_step = 0 # step of current teacher trajectory that most closesly matches the current state
        self.traj_len = 0 # how long the current teacher trajectory is after trajectory sample skipping.
        self.orig_traj_len = 0 # original trajectory length
        self.env_return = 0

        self.max_trajectory_skip_steps = max_trajectory_skip_steps

        self.last_clear_past_step = 0
        self.fixed_max_ep_len = fixed_max_ep_len
        self.max_ep_len_factor = max_ep_len_factor
        self.max_ep_len = -1
        self.lcs_dp_so_far_percentage = 0
        
        self.env_steps = 0

        self.action_space = self.env.action_space
        self.give_traj_id = give_traj_id
        # frame stacking related
        self.stack_size = stack_size
        self.stack_with_past_actions = stack_with_past_actions
        self.past_obs = []
        self.categorical = categorical
        shared = {}
        self.raw_obs_only = raw_obs_only
        if stack_size == 1:
            shared = {
                "step": spaces.Discrete(1000),
                "teacher_frames": spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_trajectory_length, self.teacher_dims), dtype=np.float32),
                "teacher_attn_mask": spaces.Box(low=0, high=1, shape=(self.max_trajectory_length,), dtype=bool),
                "teacher_time_steps": spaces.Box(low=0, high=self.max_trajectory_length-1, shape=(self.max_trajectory_length,), dtype=int),
                "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dims,), dtype=np.float32),
                "farthest_traj_match_frac": spaces.Box(low=0., high=1, shape=())
            }
            
        else:
            shared = {
                "step": spaces.Discrete(1000),
                "teacher_frames": spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_trajectory_length, self.teacher_dims), dtype=np.float32),
                "teacher_attn_mask": spaces.Box(low=0, high=1, shape=(self.max_trajectory_length,), dtype=bool),
                "teacher_time_steps": spaces.Box(low=0, high=self.max_trajectory_length-1, shape=(self.max_trajectory_length,), dtype=int),
                "observation_attn_mask": spaces.Box(low=0, high=1, shape=(stack_size,), dtype=bool),
                "observation_time_steps": spaces.Box(low=0, high=np.inf, shape=(stack_size,), dtype=int),
                "farthest_traj_match_frac": spaces.Box(low=0., high=1, shape=(), dtype=np.float32)
            }
            if self.stack_with_past_actions:
                shared["observation"] = spaces.Box(low=-np.inf, high=np.inf, shape=(stack_size, self.state_dims + self.act_dims), dtype=np.float32)
        if self.give_traj_id:
            shared["traj_id"] = spaces.Discrete(1000000)
        if self.raw_obs_only:
            self.observation_space = shared["observation"]
        else:
            self.observation_space = spaces.Dict(shared)
        # print("Trajectories dataset: ", trajectories_dataset)
        if self.use_planner:
            print("Planning mode on")
        if self.seed_by_dataset:
            self.load_trajectories(self.trajectory_ids)
        else:
            self.trajectories = self.trajectory_ids

    def load_trajectories(self, trajectories):
        self.trajectory_ids = trajectories
        self.trajectories = []
        loaded_trajectory_ids = []
        for t_id in self.trajectory_ids:
            traj = self.get_trajectory(t_id)
            if not self.skip_long_trajectories: 
                if self._traj_len(traj) > self.max_trajectory_length: continue
            self.trajectories.append(traj)
            loaded_trajectory_ids.append(t_id)
        # print(f"Out of {len(self.trajectory_ids)} trajectories, loaded {len(loaded_trajectory_ids)} that are within max_trajectory_length={self.max_trajectory_length} with trajectory_sample_skip_steps={self.trajectory_sample_skip_steps}")
        self.trajectory_ids = loaded_trajectory_ids


    def step(self, action=...):
        def merge(source, destination):
            for key, value in source.items():
                if isinstance(value, dict):
                    # get node or create one
                    node = destination.setdefault(key, {})
                    merge(value, node)
                else:
                    destination[key] = value

            return destination
        self.env_steps += 1
        obs, ret, done, info = self.env.step(action)
        obs, add_info = self._format_obs(obs)
        info["stats"] = {}
        info['task_complete'] = done
        info['stats']['env_rew'] = ret
        self.env_return += ret
        info = merge(add_info, info)

        if isinstance(obs, dict):
            obs_observation = obs["observation"]
        else:
            obs_observation = obs
        ret = self.format_ret(obs, ret)
        if self.stack_size > 1:
            self.past_obs[-1][self.state_dims:] = action
            self.past_obs.append(np.hstack([obs_observation, np.ones(self.act_dims) * -10.]))
        
        if self.stack_size > 1:
            self.stack_obs(obs)

        # record stats
        
        info['stats']['task_complete'] = done
        info["stats"]["farthest_traj_match"] = self.farthest_traj_step
        info["stats"]["farthest_traj_match_frac"] = (
            (self.farthest_traj_step) / (self.orig_traj_len - 1)
        )
        info["stats"]["closest_traj_match"] = self.closest_traj_step
        info["stats"]["closest_traj_match_frac"] = (self.closest_traj_step) / (self.orig_traj_len - 1)
        info["traj_id"] = self.curr_traj_id
        info["traj_len"] = self.orig_traj_len
        info["stats"]["plans"] = self.total_plans
        
        

        traj_done = False
        past_timelimit = False
        # make "task agnostic" by ignoring normal done signals and only adding done if we mimic the whole trajectory
        if self.orig_traj_len == self.closest_traj_step + 1:
            traj_done = True
        if self.task_agnostic:
            done = traj_done
        
        replanned = False
        if self.use_planner and not done:
            if traj_done or self.env_steps > self.last_plan_step + self.min_student_execute_length:
                curr_state = self.get_state()
                curr_obs = self.get_obs()
                # check if we need to replan now using planner
                need_replan = self.planner.need_replan(curr_state, obs, self.orig_trajectory_observations, self.env)
                if need_replan or self.env_steps - self.last_plan_step > self.max_student_execute_length:
                    # normally with planning you can't use the full current state. Provided for simplicity for programming, unused for different obs modes
                    planned_trajectory = self._plan_trajectory(curr_state, curr_obs)
                    # load the planned trajectory
                    self._select_trajectory(planned_trajectory)
                    self.total_plans += 1
                    replanned = True
                    
                    self.last_plan_step = self.env_steps
        info['replanned'] = replanned
        if self.env_steps >= self.max_ep_len:
            past_timelimit = True
        if "failed" in info and info["failed"]:
            done = False
            info['task_complete'] = False
        
        if past_timelimit:
            done = True
            info["stats"]["env_return"] = self.env_return
            if hasattr(self, 'lcs_dp_so_far'):
                info["stats"]["lcs_dp"] = self.lcs_dp_so_far
                info["stats"]["lcs_dp_percent"] = self.lcs_dp_so_far_percentage
        else:
            if not self.early_success:
                done = False
        if self.max_plans != -1 and self.total_plans > self.max_plans:
            info["failed"] = True
            info['task_complete'] = False
            done = True
        if self.raw_obs_only == True:
            obs = obs["observation"]
        return obs, ret, done, info
    def clear_past_obs(self):
        self.last_clear_past_step = self.env_steps
    def stack_obs(self, obs):
        step = obs["step"] + 1 - self.last_clear_past_step
        prepend_start = step - self.stack_size
        frame_start = max(0, prepend_start)
        obs_time_steps = np.arange(frame_start, step)
        obs_attn_mask = np.ones((self.stack_size), dtype=bool)
        if prepend_start < 0:
            obs_time_steps = np.hstack([np.zeros(-prepend_start, dtype=int), obs_time_steps])
            obs_attn_mask[:-prepend_start] = False
        obs["observation_time_steps"] = obs_time_steps
        obs["observation_attn_mask"] = obs_attn_mask
        obs["observation"] = np.vstack(self.past_obs[-self.stack_size:])
    def format_obs(self, obs):
        """
        should be overriden to additionally compute farthest_traj_step and closest_traj_step, and/or change obs as needed
        """
        trajectory_attn_mask = self.trajectory_attn_mask
        if self.silo_window_mask != -1:
            trajectory_attn_mask = self.trajectory_attn_mask.copy()
            # if self.farthest_traj_step is 0, nothing additional is masked out so no one off error here
            trajectory_attn_mask[self.max_trajectory_length - self.traj_len:self.max_trajectory_length - self.traj_len + self.farthest_traj_step] = False
            # if self.farthest_traj_step is 0, and self.silo_window_mask is 5, then steps 0, 1, ... , 5 are not masked
            trajectory_attn_mask[self.max_trajectory_length - self.traj_len + self.farthest_traj_step + self.silo_window_mask + 1:] = False
            
        obs = dict(
            observation=obs,
            teacher_frames=self.trajectory_observations,
            teacher_time_steps=self.trajectory_time_steps,
            teacher_attn_mask=trajectory_attn_mask,
            step=self.env_steps,
            farthest_traj_match_frac=0,
        )
        if (self.give_traj_id):
            obs["traj_id"]=self.curr_traj_id
        return obs, {}
    def _format_obs(self, obs):
        obs, info = self.format_obs(obs)
        if self.sub_goals:
            obs["observation"] = np.concatenate([obs["observation"], self.next_sub_goal()])
        return obs, info
    def next_sub_goal(self):
        raise NotImplementedError("Next sub goal is not implemented")

    def format_ret(self, obs, ret):
        # Can be overridden
        return ret
    def reset_env(self, seed=None, **kwargs):
        self.env.reset(seed, **kwargs)
    def reset_to_start_of_trajectory(self):
        # Can be overridden
        self.env._set_state(self.trajectory_observations[self.max_trajectory_length - self.traj_len])
    def get_obs(self):
        # Can be overridden
        return self.env._get_obs()
    def get_state(self):
        # Can be overridden
        return self.env._get_state()
    def set_state(self, state):
        # Can be overridden
        return self.env._set_state(state)
    def reset(self, init_plan_state=None, **kwargs):
        self.reset_env(**kwargs)
        self.last_clear_past_step = 0
        self.env_return = 0
        if init_plan_state is not None:
            # used for plan + replanning. # set env state to the plan's initial state (which should be the end of the last execution)
            # self.set_state(init_plan_state)
            if self.use_planner:
                # generate the plan (teacher) from the same initil state as student
                planned_trajectory = self._plan_trajectory(init_plan_state)
                # load the planned trajectory
                self._select_trajectory(planned_trajectory)
                # reset the environment appropriately
                # self.reset_to_start_of_trajectory()
        else:
            self.total_plans = 0
            self.env_steps = 0
            self.last_plan_step = 0
            if self.randomize_trajectories:
                self.curr_traj_idx = self.np_random.randint(0, len(self.trajectories))
            else:
                self.curr_traj_idx = (self.curr_traj_idx + 1) % len(self.trajectories)
            # load original teacher. In planning mode, we only do this to load the initial state
            self.current_trajectory_id = self.trajectory_ids[self.curr_traj_idx]
            self.curr_traj_id = self.trajectory_ids[self.curr_traj_idx]
            if self.seed_by_dataset:
                traj = self.trajectories[self.curr_traj_idx].copy()
                if "attns" in traj.keys():
                    traj["observations"] = traj["observations"][np.where(traj["attns"] == 0)]
                self._select_trajectory(traj)
                self.reset_to_start_of_trajectory()
            else:
                self.reset_env(int(self.curr_traj_id))
            
            if self.use_planner:
                planned_trajectory = self._plan_trajectory(self.get_state(), self.get_obs())
                self.total_plans += 1
                self._select_trajectory(planned_trajectory)
                # self.reset_to_start_of_trajectory()

        
        obs = self.get_obs()
        obs, add_info = self._format_obs(obs)
        if self.stack_size > 1:
            self.past_obs = []
            for i in range(self.stack_size):
                if self.stack_with_past_actions:
                    null_frame = np.zeros(self.state_dims + self.act_dims)
                    null_frame[self.state_dims:] = -10.
                else:
                    null_frame = np.zeros(self.state_dims)
                self.past_obs.append(
                    null_frame
                )
            if isinstance(obs, dict):
                self.past_obs[-1][:self.state_dims] = obs["observation"]
            else:
                self.past_obs[-1][:self.state_dims] = obs
        if self.stack_size > 1:
            self.stack_obs(obs)
        if self.raw_obs_only == True:
            obs = obs["observation"]
        return obs
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
    def seed(self, seed, *args, **kwargs):
        self.np_random, seed = seeding.np_random(seed)
        self.seed_val = seed
        return self.env.seed(seed, *args, **kwargs)
    @property
    def viewer(self):
        return self.env.viewer
    def get_trajectory(self, t_idx):
        raise NotImplementedError()

    def _traj_len(self, traj):
        return len(traj['observations'][0:-1:self.trajectory_sample_skip_steps+1]) + 1

    def _plan_trajectory(self, start_state, start_obs):
        raise NotImplementedError()

    def _select_trajectory(self, trajectory):
        """
        select trajectory `t_idx` and set self.trajectory, self.reward_trajectory, reset values
        """
        self.trajectory = trajectory
        
        self.trajectory_observations = np.zeros((self.max_trajectory_length, self.teacher_dims), dtype=np.float32)
        self.match_traj_count = 0
        self.farthest_traj_step = 0
        self.closest_traj_step = 0
        self.lcs_farthest_step = 0
        self.trajectory['observations']
        self.orig_trajectory_observations = self.trajectory["observations"].copy()
        self.orig_traj_len = len(self.orig_trajectory_observations)
        if self.trajectory_sample_skip_steps > 0:
            skip = self.trajectory_sample_skip_steps + 1
            new_obs = self.trajectory['observations'][0:-1:skip].copy()
            new_obs = np.vstack([new_obs, self.trajectory['observations'][-1]])
            self.trajectory['observations'] = new_obs
        self.traj_len = len(self.trajectory['observations'])
        self.trajectory_observations[self.max_trajectory_length - self.traj_len:] = self.trajectory['observations']

        self.trajectory_attn_mask = np.zeros((self.max_trajectory_length), dtype=bool)
        self.trajectory_attn_mask[self.max_trajectory_length - self.traj_len:] = True
        self.trajectory_time_steps = np.zeros((self.max_trajectory_length), dtype=int)
        self.trajectory_time_steps[self.max_trajectory_length - self.traj_len:] = np.arange(0, self.traj_len, dtype=int)

        self.max_ep_len = int(self.orig_traj_len * self.max_ep_len_factor)
        if self.fixed_max_ep_len is not None:
            self.max_ep_len = self.fixed_max_ep_len