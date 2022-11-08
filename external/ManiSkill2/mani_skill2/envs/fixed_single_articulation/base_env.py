import copy
from collections import OrderedDict
from typing import Dict, List, Tuple, Union

import attr
import numpy as np
import sapien.core as sapien
import yaml
from gym import spaces
from transforms3d.euler import euler2quat

from mani_skill2 import AGENT_CONFIG_DIR, ASSET_DIR
from mani_skill2.agents.fixed_xmate3_robotiq import FixedXmate3Robotiq
from mani_skill2.envs.fixed_single_articulation.utils import ignore_collision
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.common import (
    clip_and_scale_action,
    convert_observation_to_space,
    normalize_action_space,
)
from mani_skill2.utils.geometry import angle_distance
from mani_skill2.utils.sapien_utils import (
    get_actor_state,
    get_articulation_state,
    get_entity_by_name,
    set_actor_state,
    set_articulation_state,
)


@attr.s(auto_attribs=True, kw_only=True)
class ArticulationConfig:
    name: str
    partnet_mobility_id: int  # Note (rayc): not sure we still need it
    urdf_path: str
    multiple_collisions: bool
    scale: float
    target_joint_idx: int
    bbox_min: List[float]
    bbox_max: List[float]


class FixedXmate3RobotiqEnv(BaseEnv):
    SUPPORTED_OBS_MODES: Tuple[str] = ()
    SUPPORTED_REWARD_MODES: Tuple[str] = ()
    _agent: FixedXmate3Robotiq

    def __init__(
        self,
        articulation_config_path="",
        obs_mode=None,
        reward_mode=None,
        sim_freq=500,
        control_freq=20,
    ):
        # articulation
        with open(articulation_config_path, "r") as f:
            config_dict = yaml.safe_load(f)
            self._articulation_config = ArticulationConfig(**config_dict)
        self._max_dof = 8  # actually, it is 6 for our data
        self._target_indicator = np.zeros(self._max_dof)

        self.step_in_ep = 0
        self._prev_actor_pose = sapien.Pose()
        self._cache_obs_state_dict: OrderedDict = (
            OrderedDict()
        )  # save obs state dict to save reward computation time
        self._cache_info = {}
        super().__init__(obs_mode, reward_mode, sim_freq, control_freq)

    @property
    def agent(self):
        return self._agent

    @property
    def articulation(self):
        return self._articulation

    @property
    def target_link(self):
        return self._target_link

    @property
    def control_mode(self):
        return self._agent._control_mode

    # ---------------------------------------------------------------------------- #
    # Observation
    # ---------------------------------------------------------------------------- #
    @property
    def obs_mode(self):
        return self._obs_mode

    def get_obs(self):
        raise NotImplementedError

    # -------------------------------------------------------------------------- #
    # Agent
    # -------------------------------------------------------------------------- #

    def step_action(self, action):
        if action is None:
            pass
        elif isinstance(action, np.ndarray):
            self._agent.set_action(action)
        elif isinstance(action, dict):
            if action["control_mode"] != self._agent.control_mode:
                self._agent.set_control_mode(action["control_mode"])
            self._agent.set_action(action["action"])
        else:
            raise TypeError(type(action))
        for _ in range(self._sim_step_per_control):
            self._agent.simulation_step()
            self._scene.step()

        self._agent.update_generalized_external_forces()

    # -------------------------------------------------------------------------- #
    # Reward mode
    # -------------------------------------------------------------------------- #
    @property
    def reward_mode(self):
        return self._reward_mode

    @reward_mode.setter
    def reward_mode(self, mode: str):
        if mode not in self.SUPPORTED_REWARD_MODES:
            raise NotImplementedError("Unsupported reward mode: {}".format(mode))
        self._reward_mode = mode

    def get_reward(self):
        raise NotImplementedError

    # -------------------------------------------------------------------------- #
    # Reset
    # -------------------------------------------------------------------------- #
    def _clear(self):
        self._agent = None
        self._camera = None
        super()._clear()

    def _setup_physical_materials(self):
        self.add_physical_material("default", 1.0, 1.0, 0.0)

    def _setup_render_materials(self):
        self.add_render_material(
            "ground",
            color=[0.5, 0.5, 0.5, 1],
            metallic=1.0,
            roughness=0.7,
            specular=0.04,
        )
        self.add_render_material(
            "default",
            color=[0.8, 0.8, 0.8, 1],
            metallic=0,
            roughness=0.9,
            specular=0.0,
        )

    def _load_articulation(self):
        loader = self._scene.create_urdf_loader()
        loader.load_multiple_collisions_from_file = (
            self._articulation_config.multiple_collisions
        )

        loader.scale = self._articulation_config.scale
        loader.fix_root_link = True

        config = {"material": self._physical_materials["default"]}
        self._articulation = loader.load(
            str(ASSET_DIR / self._articulation_config.urdf_path)
        )
        self._articulation.set_name(self._articulation_config.name)
        self._target_joint_idx = self._articulation_config.target_joint_idx

        self._target_joint = self._articulation.get_active_joints()[
            self._target_joint_idx
        ]
        self._target_link = self._target_joint.get_child_link()

        self._target_indicator = np.zeros(self._max_dof)
        self._target_indicator[self._target_joint_idx] = 1.0

    def _load_agent(self):
        self._agent = FixedXmate3Robotiq.from_config_file(
            AGENT_CONFIG_DIR / "fixed_xmate3_robotiq.yml",
            self._scene,
            self._control_freq,
        )
        self.grasp_site: sapien.Link = get_entity_by_name(
            self._agent._robot.get_links(), "grasp_convenient_link"
        )

    def _initialize_articulation(self):
        raise NotImplementedError

    def _initialize_agent(self):
        raise NotImplementedError

    def _load_table(self):
        loader = self._scene.create_actor_builder()
        loader.add_visual_from_file(
            str(ASSET_DIR / "descriptions/optical_table/visual/optical_table.dae")
        )
        loader.add_collision_from_file(
            str(ASSET_DIR / "descriptions/optical_table/visual/optical_table.dae")
        )
        self._table = loader.build_static(name="table")
        self._table.set_pose(sapien.Pose([0.0, 0.0, -0.04]))

    def reconfigure(self):
        self.step_in_ep = 0
        self._prev_actor_pose = sapien.Pose()
        self._clear()

        self._setup_scene()
        self._setup_physical_materials()
        self._setup_render_materials()
        self._load_table()
        self._load_articulation()
        ignore_collision(self._articulation)
        self._load_agent()

        self._setup_lighting()
        self._setup_camera()
        if self._viewer is not None:
            self._setup_viewer()
        # cache actors and articulations for sim state
        self._actors = self.get_actors()
        self._articulations = self.get_articulations()
        # Cache initial simulation state
        self._initial_sim_state = self.get_sim_state()

    def reset(self, seed=None, reconfigure=False):
        if seed is None:
            self._episode_seed = self._main_rng.randint(2 ** 32)
        else:
            self._episode_seed = seed
        self._episode_rng = np.random.RandomState(self._episode_seed)
        if reconfigure:
            self.reconfigure()
        else:
            self.set_sim_state(self._initial_sim_state)
        self._initialize_articulation()
        self._initialize_agent()
        self._cache_obs_state_dict.clear()
        self._cache_info.clear()
        return self.get_obs()

    # -------------------------------------------------------------------------- #
    # Step
    # -------------------------------------------------------------------------- #
    def check_success(self) -> bool:
        raise NotImplementedError

    def get_done(self):
        # return self.check_success()
        return False

    def get_info(self):
        return dict(success=self.check_success())

    def step(self, action: Union[None, np.ndarray, Dict]):
        self.step_in_ep += 1
        self.step_action(action)
        obs = self.get_obs()
        reward = self.get_reward()
        done = self.get_done()
        info = self.get_info()
        return obs, reward, done, info

    # ---------------------------------------------------------------------------- #
    # Visualization
    # ---------------------------------------------------------------------------- #
    def _setup_lighting(self):
        self._scene.set_ambient_light([0.3, 0.3, 0.3])
        self._scene.add_point_light([2, 2, 2], [1, 1, 1])
        self._scene.add_point_light([2, -2, 2], [1, 1, 1])
        self._scene.add_point_light([-2, 0, 2], [1, 1, 1])
        self._scene.add_directional_light([1, -1, -1], [0.3, 0.3, 0.3])
        self._scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _setup_viewer(self):
        super()._setup_viewer()
        self._viewer.set_camera_xyz(1.0, 0.0, 1.2)
        self._viewer.set_camera_rpy(0, -0.5, 3.14)

    def _setup_camera(self):
        self._camera = self._scene.add_camera("frontview", 512, 512, 1, 0.01, 10)
        self._camera.set_local_pose(sapien.Pose([1, 0, 1.2], euler2quat(0, 0.5, 3.14)))

    def render(self, mode="human"):
        if mode == "rgb_array":
            self._scene.update_render()
            self._camera.take_picture()
            rgb = self._camera.get_color_rgba()[..., :3]
            # rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            return rgb
        else:
            return super().render(mode=mode)

    # utilization
    def check_actor_static(self, actor, max_v=None, max_ang_v=None):
        if self.step_in_ep <= 1:
            flag_v = (max_v is None) or (np.linalg.norm(actor.get_velocity()) <= max_v)
            flag_ang_v = (max_ang_v is None) or (
                np.linalg.norm(actor.get_angular_velocity()) <= max_ang_v
            )
        else:
            pose = actor.get_pose()
            t = 1.0 / self._control_freq
            flag_v = (max_v is None) or (
                np.linalg.norm(pose.p - self._prev_actor_pose.p) <= max_v * t
            )
            flag_ang_v = (max_ang_v is None) or (
                angle_distance(self._prev_actor_pose, pose) <= max_ang_v * t
            )
        self._prev_actor_pose = actor.get_pose()
        return flag_v and flag_ang_v

    # -------------------------------------------------------------------------- #
    # Simulation state (required for MPC)
    # -------------------------------------------------------------------------- #
    def get_actors(self):
        return [x for x in self._scene.get_all_actors()]

    def get_articulations(self):
        return [self._agent._robot, self._articulation]

    def get_sim_state(self) -> np.ndarray:
        state = []
        # for actor in self._scene.get_all_actors():
        for actor in self._actors:
            state.append(get_actor_state(actor))
        # for articulation in self._scene.get_all_articulations():
        for articulation in self._articulations:
            state.append(get_articulation_state(articulation))
        return np.hstack(state)

    def set_sim_state(self, state: np.ndarray):
        KINEMANTIC_DIM = 13  # [pos, quat, lin_vel, ang_vel]
        start = 0
        for actor in self._actors:
            set_actor_state(actor, state[start : start + KINEMANTIC_DIM])
            start += KINEMANTIC_DIM
        for articulation in self._articulations:
            ndim = KINEMANTIC_DIM + 2 * articulation.dof
            set_articulation_state(articulation, state[start : start + ndim])
            start += ndim

    def get_state(self):
        return self.get_sim_state()

    def set_state(self, state: np.ndarray):
        return self.set_sim_state(state)
