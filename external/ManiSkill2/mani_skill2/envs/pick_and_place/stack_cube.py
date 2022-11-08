from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import sapien.core as sapien
from sapien.core import Pose

from mani_skill2.utils.common import register_gym_env
from mani_skill2.utils.sapien_utils import get_pairwise_contact_impulse, vectorize_pose

from .base_env import FixedXmate3RobotiqEnv, PandaEnv


class UniformSampler:
    """Uniform placement sampler.

    Args:
        ranges: ((low1, low2, ...), (high1, high2, ...))
        rng (np.random.RandomState): random generator
    """

    def __init__(
        self, ranges: Tuple[List[float], List[float]], rng: np.random.RandomState
    ) -> None:
        assert len(ranges) == 2 and len(ranges[0]) == len(ranges[1])
        self._ranges = ranges
        self._rng = rng
        self._fixtures = []

    def sample(self, radius, max_trials, append=True):
        """Sample a position.

        Args:
            radius (float): collision radius.
            max_trials (int): maximal trials to sample.
            append (bool, optional): whether to append the new sample to fixtures. Defaults to True.
            verbose (bool, optional): whether to print verbosely. Defaults to False.

        Returns:
            np.ndarray: a sampled position.
        """
        if len(self._fixtures) == 0:
            pos = self._rng.uniform(*self._ranges)
        else:
            fixture_pos = np.array([x[0] for x in self._fixtures])
            fixture_radius = np.array([x[1] for x in self._fixtures])
            for i in range(max_trials):
                pos = self._rng.uniform(*self._ranges)
                dist = np.linalg.norm(pos - fixture_pos, axis=1)
                if np.all(dist > fixture_radius + radius):
                    # print(f"Found a valid sample at {i}-th trial")
                    break
        if append:
            self._fixtures.append((pos, radius))
        return pos


@register_gym_env("StackCubePanda-v0", max_episode_steps=200)
class StackCubePandaEnv(PandaEnv):
    def _load_actors(self):
        self._add_ground()

        # cubeA
        builder = self._scene.create_actor_builder()
        half_size = [0.02, 0.02, 0.02]
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size, color=[1, 0, 0])
        self.cubeA = builder.build("cubeA")

        # cubeB
        builder = self._scene.create_actor_builder()
        half_size = [0.025, 0.025, 0.025]
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size, color=[0, 1, 0])
        # self.cubeB = builder.build("cubeB")
        self.cubeB = builder.build_static("cubeB")

    def _initialize_actors(self):
        ranges = [[-0.08, -0.08], [0.08, 0.08]]
        sampler = UniformSampler(ranges, self._episode_rng)
        cubeA_pos = sampler.sample(0.02 * 1.414, 100)
        cubeB_pos = sampler.sample(0.025 * 1.414, 100)

        self.cubeA.set_pose(Pose([cubeA_pos[0], cubeA_pos[1], 0.02]))
        self.cubeB.set_pose(Pose([cubeB_pos[0], cubeB_pos[1], 0.025]))

    def _initialize_agent(self):
        qpos = np.array(
            [0, np.pi / 16, 0, -np.pi * 5 / 6, 0, np.pi - 0.2, np.pi / 4, 0, 0]
        )
        qpos[:-2] += self._episode_rng.normal(0, 0.02, len(qpos) - 2)
        self._agent.reset(qpos)
        self._agent._robot.set_pose(Pose([-0.56, 0, 0]))

    def _get_obs_extra(self):
        if self._obs_mode in ["rgbd", "pointcloud"]:
            return OrderedDict(
                gripper_pose=vectorize_pose(self.grasp_site.pose),
            )
        else:
            return OrderedDict(
                cubeA_pose=vectorize_pose(self.cubeA.pose),
                cubeB_pose=vectorize_pose(self.cubeB.pose),
                gripper_pose=vectorize_pose(self.grasp_site.pose),
                gripper_to_cubeA_pos=self.grasp_site.pose.p - self.cubeA.pose.p,
                gripper_to_cubeB_pos=self.grasp_site.pose.p - self.cubeB.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
            )

    def _check_cubeA_contact_cubeB(self):
        impulse = get_pairwise_contact_impulse(
            self._scene.get_contacts(), self.cubeA, self.cubeB
        )
        has_contact = np.linalg.norm(impulse) > 1e-6
        return has_contact

    def _check_cubeA_above_cubeB(self):
        # NOTE(jigu): allow some tolerance for numerical error
        return self.cubeA.pose.p[2] - 0.02 > self.cubeB.pose.p[2] + 0.02

    def _check_cubeA_on_cubeB(self, dist_thresh=0.01):
        cubeA_pos = self.cubeA.pose.p
        cubeB_pos = self.cubeB.pose.p
        goal_xyz = np.hstack([cubeB_pos[0:2], cubeB_pos[2] + 0.025 + 0.02])
        cubeA_to_goal_dist = np.linalg.norm(goal_xyz - cubeA_pos)
        return cubeA_to_goal_dist < dist_thresh

    def check_success(self):
        is_cubeA_grasped = self._agent.check_grasp(self.cubeA)
        # is_cubeA_above_cubeB = self._check_cubeA_above_cubeB()
        is_cubeA_on_cubeB = self._check_cubeA_on_cubeB()
        has_cubeA_contact_cubeB = self._check_cubeA_contact_cubeB()

        # print("is_cubeA_grasped", is_cubeA_grasped)
        # print("is_cubeA_on_top_of_cubeB", is_cubeA_above_cubeB)
        # print("has_cubeA_contact_cubeB", has_cubeA_contact_cubeB)

        # return not is_cubeA_grasped and is_cubeA_above_cubeB and has_cubeA_contact_cubeB
        return not is_cubeA_grasped and is_cubeA_on_cubeB and has_cubeA_contact_cubeB

    def compute_dense_reward(self):
        reward = 0.0
        if self.check_success():
            reward = 2.25
        else:
            # reaching object reward
            gripper_pos = self.grasp_site.pose.p
            cubeA_pos = self.cubeA.pose.p
            cubeA_to_gripper_dist = np.linalg.norm(gripper_pos - cubeA_pos)
            reaching_reward = 1 - np.tanh(10.0 * cubeA_to_gripper_dist)
            reward += reaching_reward

            # grasping reward
            is_cubeA_grasped = self._agent.check_grasp(self.cubeA)
            if is_cubeA_grasped:
                reward += 0.25

            # reaching goal reward
            if is_cubeA_grasped:
                cubeA_pos = self.cubeA.pose.p
                cubeB_pos = self.cubeB.pose.p
                goal_xyz = np.hstack([cubeB_pos[0:2], cubeB_pos[2] + 0.025 + 0.02])
                cubeA_to_goal_dist = np.linalg.norm(goal_xyz - cubeA_pos)
                reaching_reward2 = 1 - np.tanh(10.0 * cubeA_to_goal_dist)
                reward += reaching_reward2

        return reward


@register_gym_env("StackCubeFixedXmate3Robotiq-v0", max_episode_steps=200)
class StackCubeFixedXmate3RobotiqEnv(FixedXmate3RobotiqEnv, StackCubePandaEnv):
    pass
