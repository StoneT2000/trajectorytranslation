from collections import OrderedDict

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.euler import euler2quat

from mani_skill2 import ASSET_DIR
from mani_skill2.utils.common import register_gym_env
from mani_skill2.utils.sapien_utils import look_at, vectorize_pose

from .base_env import PandaEnv
from .pick_clutter import EpisodeIterator


@register_gym_env("CollisionAvoidancePanda-v0", max_episode_steps=200)
class CollisionAvoidancePandaEnv(PandaEnv):
    def _build_cube(
        self, half_size=(0.02, 0.02, 0.02), color=(1, 0, 0), name="cube", static=False
    ):
        builder = self._scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size, color=color)
        if static:
            return builder.build_static(name)
        else:
            return builder.build(name)

    def _build_goal_site(self, radius=0.02, color=(0, 1, 0), name="goal_site"):
        builder = self._scene.create_actor_builder()
        builder.add_sphere_visual(radius=radius, color=color)
        goal_site = builder.build_static(name)
        goal_site.hide_visual()
        return goal_site

    def _load_actors(self):
        self._add_ground()
        self.cube = self._build_cube([0.02, 0.02, 0.02])
        self.obstacle = self._build_cube(
            [0.02, 0.1, 0.05], color=[0, 0, 1], name="obstacle", static=True
        )

        self.goal_pos = np.array([-0.1, 0, 0.02], np.float32)
        self.goal_site = self._build_goal_site()
        self.goal_site.set_pose(Pose(self.goal_pos))

    def _initialize_actors(self):
        self.cube.set_pose(Pose([0.1, 0, 0.02]))
        self.obstacle.set_pose(Pose([0.0, 0, 0.05]))

    def _get_obs_extra(self):
        return OrderedDict(
            cube_pose=vectorize_pose(self.cube.pose),
            gripper_pose=vectorize_pose(self.grasp_site.pose),
            gripper_to_cube_pos=self.cube.pose.p - self.grasp_site.pose.p,
        )

    def check_success(self):
        cube_to_goal_pos = self.goal_pos - self.cube.pose.p
        is_reached = np.linalg.norm(cube_to_goal_pos) < 0.02
        return is_reached

    def compute_dense_reward(self):
        reward = 0.0

        if self.check_success():
            reward = 2.25
        else:
            # reaching reward
            gripper_pos = self.grasp_site.get_pose().p
            cube_pos = self.cube.get_pose().p
            dist = np.linalg.norm(gripper_pos - cube_pos)
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward

            is_grasped = self._agent.check_grasp(self.cube)

            # grasp reward
            if is_grasped:
                reward += 0.25

            # reaching (goal) reward
            if is_grasped:
                cube_to_goal_pos = self.goal_pos - self.cube.pose.p
                cube_to_goal_dist = np.linalg.norm(cube_to_goal_pos)
                reaching_reward2 = 1 - np.tanh(10 * cube_to_goal_dist)
                reward += reaching_reward2

        return reward

    def render(self, mode="human"):
        if mode in ["human", "rgb_array"]:
            self.goal_site.unhide_visual()
            ret = super().render(mode=mode)
            self.goal_site.hide_visual()
        else:
            ret = super().render(mode=mode)
        return ret


@register_gym_env("CollisionAvoidancePanda-v1", max_episode_steps=200)
class CollisionAvoidancePandaEnvV1(PandaEnv):
    DEFAULT_EPISODE_JSON = (
        ASSET_DIR / "collision_avoidance/episodes/eval_panda_100.json.gz"
    )

    def __init__(self, json_path=None, **kwargs):
        if json_path is None:
            json_path = self.DEFAULT_EPISODE_JSON
        self.episode_iterator = EpisodeIterator.from_json(json_path, seed=0)
        super().__init__(**kwargs)

    def reconfigure(self):
        self.episode_config = next(self.episode_iterator)
        return super().reconfigure()

    def reset(self, seed=None, reconfigure=True):
        return super().reset(seed=seed, reconfigure=reconfigure)

    def _build_cube(self, half_size, color, name="cube", static=False):
        builder = self._scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size, color=color)
        if static:
            return builder.build_static(name)
        else:
            return builder.build(name)

    def _load_actors(self):
        self.obstacles = []
        for i, actor_cfg in enumerate(self.episode_config["actors"]):
            actor = self._build_cube(
                actor_cfg["half_size"],
                actor_cfg["color"],
                name=f"obstacle_{i}",
                static=True,
            )
            self.obstacles.append(actor)

        self.goal_site = self._build_goal_site()

    def _initialize_actors(self):
        for i, actor_cfg in enumerate(self.episode_config["actors"]):
            pose = actor_cfg["pose"]
            self.obstacles[i].set_pose(Pose(pose[:3], pose[3:]))

        self._agent._robot.set_qpos(self.episode_config["end_qpos"])
        self.goal_pos = self.grasp_site.pose.p
        self.goal_site.set_pose(Pose(self.goal_pos))

    def _initialize_agent(self):
        qpos = self.episode_config["start_qpos"]
        self._agent.reset(qpos)
        self._agent._robot.set_pose(Pose([0, 0, 0]))

    def check_success(self):
        return False

    def compute_dense_reward(self):
        reward = 0.0
        return reward

    def _setup_cameras(self):
        # Camera only for rendering, not included in `_cameras`
        self.render_camera = self._scene.add_camera(
            "render_camera", 512, 512, 1, 0.001, 10
        )
        self.render_camera.set_local_pose(
            sapien.Pose([1.5, 0, 1.5], euler2quat(0, 0.6, 3.14))
        )

        third_view_camera = self._scene.add_camera(
            "third_view_camera", 128, 128, np.pi / 2, 0.001, 10
        )
        third_view_camera.set_local_pose(look_at([1.0, 0, 1.0], [0, 0, 0]))
        self._cameras["third_view_camera"] = third_view_camera

    def _setup_viewer(self):
        super()._setup_viewer()
        self._viewer.set_camera_xyz(1.5, 0.0, 1.5)
        self._viewer.set_camera_rpy(0, -0.6, 3.14)

    def render(self, mode="human"):
        if mode in ["human", "rgb_array"]:
            self.goal_site.unhide_visual()
            ret = super().render(mode=mode)
            self.goal_site.hide_visual()
        else:
            ret = super().render(mode=mode)
        return ret
