from collections import OrderedDict

import numpy as np
import sapien.core as sapien
from sapien.core import Pose

from mani_skill2.utils.common import register_gym_env
from mani_skill2.utils.sapien_utils import vectorize_pose
from mani_skill2.utils.tmu import register_gym_env_for_tmu

from .base_env import FixedXmate3RobotiqEnv, PandaEnv


@register_gym_env_for_tmu("LiftCubePanda_@-v0")
@register_gym_env("LiftCubePanda-v0", max_episode_steps=200)
class LiftCubePandaEnv(PandaEnv):
    def _build_cube(self, half_size=(0.02, 0.02, 0.02), color=(1, 0, 0), name="cube"):
        builder = self._scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size, color=color)
        return builder.build(name)

    def _load_actors(self):
        self._add_ground()
        self.cube = self._build_cube()

    def _initialize_actors(self):
        cube_xy = self._episode_rng.uniform(-0.03, 0.03, [2])
        cube_xyz = np.hstack([cube_xy, 0.02])
        self.cube.set_pose(Pose(cube_xyz))
        self._cube_xy = cube_xy

    def _get_obs_extra(self):
        return OrderedDict(
            cube_pose=vectorize_pose(self.cube.pose),
            gripper_pose=vectorize_pose(self.grasp_site.pose),
            gripper_to_cube_pos=self.cube.pose.p - self.grasp_site.pose.p,
        )

    def check_success(self):
        is_lifted = self.cube.get_pose().p[2] >= 0.2
        is_grasped = self._agent.check_grasp(self.cube)
        return is_lifted and is_grasped

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

            # lifting reward
            if is_grasped:
                lifting_reward = (self.cube.pose.p[2] - 0.02) / (0.2 - 0.02)
                lifting_reward = min(lifting_reward, 1.0)
                reward += lifting_reward

        return reward


@register_gym_env_for_tmu("LiftCubeFixedXmate3Robotiq_@-v0")
@register_gym_env("LiftCubeFixedXmate3Robotiq-v0", max_episode_steps=200)
class LiftCubeFixedXmate3RobotiqEnv(FixedXmate3RobotiqEnv, LiftCubePandaEnv):
    pass


@register_gym_env_for_tmu("LiftCubePanda_@-v1")
@register_gym_env("LiftCubePanda-v1", max_episode_steps=200)
class LiftCubePandaEnvV1(LiftCubePandaEnv):
    def compute_dense_reward(self):
        reward = 0.0

        if self.check_success():
            reward = 2.25 + self.drift_penalty()
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

            # lifting reward
            if is_grasped:
                lifting_reward = 1.25 - 5 * np.abs(self.cube.pose.p[2] - 0.25)
                lifting_reward = max(lifting_reward, 1.0)

                reward += lifting_reward
                reward += self.drift_penalty()

        return reward

    def drift_penalty(self):
        curr_xy = self.cube.pose.p[:2]
        offset = np.abs(curr_xy - self._cube_xy).sum()
        drift_penalty = (0.05 - offset) * 10
        return drift_penalty


@register_gym_env_for_tmu("LiftCubeFixedXmate3Robotiq_@-v1")
@register_gym_env("LiftCubeFixedXmate3Robotiq-v1", max_episode_steps=200)
class LiftCubeFixedXmate3RobotiqEnvV1(FixedXmate3RobotiqEnv, LiftCubePandaEnvV1):
    pass
