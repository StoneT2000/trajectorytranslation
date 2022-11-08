from collections import OrderedDict
from pathlib import Path
from typing import List

import attr
import numpy as np
import sapien.core as sapien
from sapien.core import Pose

from mani_skill2 import AGENT_CONFIG_DIR, ASSET_DIR
from mani_skill2.agents.fixed_xmate3_robotiq import FixedXmate3Robotiq
from mani_skill2.utils.common import register_gym_env
from mani_skill2.utils.io import load_json
from mani_skill2.utils.sapien_utils import (
    get_articulation_max_impulse_norm,
    get_entity_by_name,
    vectorize_pose,
)

from .base_env import FixedXmate3RobotiqEnv, PandaEnv
from .utils import OCRTOC_DIR


class EpisodeIterator:
    def __init__(self, episodes, cycle=True, shuffle=False, seed=None) -> None:
        self.episodes = episodes
        self.cycle = cycle
        self.shuffle = shuffle

        self.seed(seed)
        self._iterator = iter(self._indices)

    def __iter__(self):
        return self

    def __next__(self):
        next_ind = next(self._iterator, None)
        if next_ind is None:
            if not self.cycle:
                raise StopIteration

            if self.shuffle:
                self._indices = self.np_random.permutation(len(self.episodes))
            self._iterator = iter(self._indices)

            next_ind = next(self._iterator)

        return self.episodes[next_ind]

    def reset(self):
        if self.shuffle:
            self._indices = self.np_random.permutation(len(self.episodes))
        self._iterator = iter(self._indices)

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        if self.shuffle:
            self._indices = self.np_random.permutation(len(self.episodes))
        else:
            self._indices = np.arange(len(self.episodes))

    @classmethod
    def from_directory(cls, dirname, **kwargs):
        episode_dir = Path(dirname)
        assert episode_dir.is_dir(), episode_dir
        episode_paths = list(episode_dir.glob("*.json"))
        episodes = [load_json(p) for p in episode_paths]
        return cls(episodes, **kwargs)

    @classmethod
    def from_json(cls, json_path, **kwargs):
        episodes = load_json(json_path)
        return cls(episodes, **kwargs)


DEFAULT_EPISODE_JSON = OCRTOC_DIR / "episodes" / "train_ycb_box_v0_1000.json.gz"


@register_gym_env("PickClutterPanda-v0", max_episode_steps=200)
class PickClutterPandaEnv(PandaEnv):
    def __init__(self, json_path=DEFAULT_EPISODE_JSON, **kwargs):
        # self.episode_config = load_json(json_path)
        # self.episode_iterator = EpisodeIterator.from_directory(json_path, seed=0)
        self.episode_iterator = EpisodeIterator.from_json(json_path, seed=0)
        super().__init__(**kwargs)

    def seed(self, seed=None):
        self.episode_iterator.seed(seed)
        return super().seed(seed=seed)

    def _build_actor_from_json(self, config, physical_material=None):
        builder = self._scene.create_actor_builder()
        scale = np.broadcast_to(config["scale"], 3)
        model_id = config["name"]
        density = config.get("density", 1000.0)

        collision_file = str(
            OCRTOC_DIR / "models" / model_id / "collision_meshes" / "collision.obj"
        )
        builder.add_multiple_collisions_from_file(
            filename=collision_file,
            scale=scale,
            material=physical_material,
            density=density,
        )

        visual_file = str(
            OCRTOC_DIR / "models" / model_id / "visual_meshes" / "visual.dae"
        )
        builder.add_visual_from_file(
            filename=visual_file,
            scale=scale,
        )

        actor = builder.build(name=model_id)

        pose = config["pose"]
        actor.set_pose(Pose(pose[:3], pose[3:]))

        return actor

    def reconfigure(self):
        self.episode_config = next(self.episode_iterator)
        return super().reconfigure()

    def reset(self, seed=None, reconfigure=True):
        return super().reset(seed=seed, reconfigure=reconfigure)

    def _load_actors(self):
        self._add_ground()

        self.objs: List[sapien.Actor] = []
        for actor_cfg in self.episode_config["actors"]:
            actor = self._build_actor_from_json(actor_cfg)
            self.objs.append(actor)
        self.obj_idx = self._episode_rng.choice(len(self.objs))
        self.obj = self.objs[self.obj_idx]

        # goal (visual only)
        builder = self._scene.create_actor_builder()
        builder.add_sphere_visual(radius=0.02, color=[0, 1, 0])
        self.goal_site = builder.build_static("goal_site")
        self.goal_site.hide_visual()

    def _initialize_actors(self):
        # Initialize goal
        self.goal_pos = np.array([0.0, 0.0, 0.2])
        self.goal_site.set_pose(Pose(self.goal_pos))

    def _initialize_agent(self):
        # qpos = np.array([0, -0.785, 0, -2.356, 0, 1.57, 0.785, 0, 0])
        qpos = np.array([0.027, 0.048, -0.027, -1.873, -0.006, 1.86, 0.785, 0, 0])
        qpos[:-2] += self._episode_rng.normal(0, 0.02, len(qpos) - 2)
        self._agent.reset(qpos)
        self._agent._robot.set_pose(Pose([-0.56, 0, 0]))

    def _get_obs_extra(self):
        if self._obs_mode in ["rgbd", "pointcloud"]:
            return OrderedDict(
                gripper_pose=vectorize_pose(self.grasp_site.pose),
                goal_pos=self.goal_pos,
                gripper_to_goal_pos=self.goal_pos - self.grasp_site.pose.p,
            )
        else:
            obj_pose = self.obj.pose.transform(self.obj.cmass_local_pose)
            return OrderedDict(
                agent_state=self._agent.get_proprioception(),  # proprioception
                obj_pose=vectorize_pose(obj_pose),
                gripper_pose=vectorize_pose(self.grasp_site.pose),
                gripper_to_obj_pos=obj_pose.p - self.grasp_site.pose.p,
                goal_pos=self.goal_pos,
                obj_to_goal_pos=self.goal_pos - obj_pose.p,
                gripper_to_goal_pos=self.goal_pos - self.grasp_site.pose.p,
            )

    def _get_images(self):
        obs_dict = super()._get_images(actor_seg=True)
        for cam_name, cam_images in obs_dict.items():
            obs_dict[cam_name]["obj_mask"] = cam_images["actor_seg"] == self.obj.id
        return obs_dict

    def check_success(self):
        obj_pose = self.obj.pose.transform(self.obj.cmass_local_pose)
        obj_to_goal_pos = self.goal_pos - obj_pose.p
        return np.linalg.norm(obj_to_goal_pos) < 0.02

    def compute_dense_reward(self):
        reward = 0.0

        if self.check_success():
            reward = 2.25
        else:
            obj_pose = self.obj.pose.transform(self.obj.cmass_local_pose)

            gripper_to_obj_pos = obj_pose.p - self.grasp_site.pose.p
            gripper_to_obj_dist = np.linalg.norm(gripper_to_obj_pos)
            reaching_reward = 1 - np.tanh(10.0 * gripper_to_obj_dist)
            reward += reaching_reward

            is_grasped = self._agent.check_grasp(self.obj)
            reward += 0.25 if is_grasped else 0.0

            if is_grasped:
                cube_to_goal_pos = self.goal_pos - obj_pose.p
                cube_to_goal_dist = np.linalg.norm(cube_to_goal_pos)
                reaching_reward2 = 1 - np.tanh(10 * cube_to_goal_dist)
                reward += reaching_reward2

            # contacts = self._scene.get_contacts()
            # max_impulse_norm = get_articulation_max_impulse_norm(
            #     contacts, self._agent._robot, [self.obj]
            # )
            # # print("max_impulse_norm", max_impulse_norm)
            # if max_impulse_norm > 1e-3:
            #     reward -= 0.1

        return reward

    def unhide_goal(self):
        self.goal_site.unhide_visual()
        for obj in self.objs:
            if obj == self.obj:
                continue
            for vb in self.obj.get_visual_bodies():
                vb.set_visibility(0.5)

    def hide_goal(self):
        self.goal_site.hide_visual()
        for obj in self.objs:
            if obj == self.obj:
                continue
            for vb in self.obj.get_visual_bodies():
                vb.set_visibility(1.0)

    def render(self, mode="human"):
        if mode in ["human", "rgb_array"]:
            self.unhide_goal()
            ret = super().render(mode=mode)
            self.hide_goal()
        else:
            ret = super().render(mode=mode)
        return ret

    def get_state(self) -> np.ndarray:
        state = super().get_state()
        return np.hstack([state, self.goal_pos])

    def set_state(self, state):
        self.goal_pos = state[-3:]
        super().set_state(state[:-3])


@register_gym_env("PickClutterFixedXmate3Robotiq-v0", max_episode_steps=200)
class PickClutterFixedXmate3RobotiqEnv(FixedXmate3RobotiqEnv, PickClutterPandaEnv):
    pass
