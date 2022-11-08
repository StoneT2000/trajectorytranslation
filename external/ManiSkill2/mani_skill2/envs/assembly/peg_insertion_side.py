from collections import OrderedDict

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.euler import euler2quat

from mani_skill2.utils.common import register_gym_env
from mani_skill2.utils.sapien_utils import vectorize_pose

from .base_env import FixedXmate3RobotiqEnv, PandaEnv


@register_gym_env(name="PegInsertionSidePanda-v0", max_episode_steps=200)
class PegInsertionSidePandaEnv(PandaEnv):
    def _build_box_with_hole(
        self, inner_radius, outer_radius, height, name="box_with_hole"
    ):
        builder = self._scene.create_actor_builder()
        thickness = (outer_radius - inner_radius) * 0.5
        half_size = [thickness, outer_radius, height]
        offset = thickness + inner_radius
        poses = [
            Pose([offset, 0, 0], [1, 0, 0, 0]),
            Pose([-offset, 0, 0], [1, 0, 0, 0]),
            Pose([0, offset, 0], [0.7071068, 0, 0, 0.7071068]),
            Pose([0, -offset, 0], [0.7071068, 0, 0, 0.7071068]),
        ]
        for pose in poses:
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, color=[0, 1, 0])
        return builder.build_static(name)

    def _load_actors(self):
        self._add_ground()
        # peg
        builder = self._scene.create_actor_builder()
        half_size = [0.02, 0.1, 0.02]
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size, color=[1, 0, 0])
        self.peg = builder.build("peg")
        self.peg_head_offset = [0, half_size[1], 0]

        # box with hole
        self.box = self._build_box_with_hole(0.025, 0.1, 0.1)

    def _initialize_actors(self):
        self.peg.set_pose(Pose([0.0, 0.0, 0.02]))
        self.box.set_pose(Pose([0.0, 0.3, 0.1], [0.7071068, 0.7071068, 0, 0]))

    def _initialize_agent(self):
        qpos = np.array(
            [0, np.pi / 16, 0, -np.pi * 5 / 6, 0, np.pi - 0.2, np.pi / 4, 0, 0]
        )
        qpos[6] += np.pi / 2
        qpos[:-2] += self._episode_rng.normal(0, 0.02, len(qpos) - 2)
        self._agent.reset(qpos)
        self._agent._robot.set_pose(Pose([-0.56, 0, 0]))

    def _get_obs_state_dict(self):
        state_dict = OrderedDict()
        peg_head_pos = self.peg.pose.transform(Pose(self.peg_head_offset)).p
        state_dict.update(
            agent_state=self._agent.get_proprioception(),  # proprioception
            peg_pose=vectorize_pose(self.peg.pose),
            box_pose=vectorize_pose(self.box.pose),
            gripper_pose=vectorize_pose(self.grasp_site.pose),
            gripper_to_peg_pos=self.peg.pose.p - self.grasp_site.pose.p,
            box_to_peg_pos=self.peg.pose.p - self.box.pose.p,
            box_to_peg_head_pos=peg_head_pos - self.box.pose.p,
        )
        return state_dict

    def _get_obs_rgbd(self):
        obs_dict = super()._get_obs_rgbd()
        obs_dict.update(
            agent_state=self._agent.get_proprioception(),  # proprioception
            gripper_pose=vectorize_pose(self.grasp_site.pose),
        )
        return obs_dict

    def check_success(self):
        peg_head_pos = self.peg.pose.transform(Pose(self.peg_head_offset)).p
        box_pos = self.box.get_pose().p
        distance = np.linalg.norm(peg_head_pos - box_pos)
        return distance < 0.01

    def compute_dense_reward(self):
        reward = 0.0

        if self.check_success():
            reward = 2.25
        else:
            # reaching reward
            gripper_pos = self.grasp_site.get_pose().p
            peg_pos = self.peg.get_pose().p
            gripper_to_peg_dist = np.linalg.norm(gripper_pos - peg_pos)
            reaching_reward = 1 - np.tanh(10.0 * gripper_to_peg_dist)
            reward += reaching_reward

            # grasp reward
            is_grasped = self._agent.check_grasp(self.peg)
            if is_grasped:
                reward += 0.25

            # insertion reward
            if is_grasped:
                peg_head_pos = self.peg.pose.transform(Pose(self.peg_head_offset)).p
                box_to_peg_head_dist = np.linalg.norm(peg_head_pos - self.box.pose.p)
                insertion_reward = 1 - np.tanh(10.0 * box_to_peg_head_dist)
                reward += insertion_reward

        return reward

    def _setup_camera(self):
        self._camera = self._scene.add_camera("frontview", 512, 512, 1, 0.01, 10)
        self._camera.set_local_pose(
            sapien.Pose([0.5, -0.3, 0.5], euler2quat(0, 0.5, 2.5))
        )


@register_gym_env(name="PegInsertionSideFixedXmate3Robotiq-v0", max_episode_steps=200)
class PegInsertionSideFixedXmate3RobotiqEnv(
    FixedXmate3RobotiqEnv, PegInsertionSidePandaEnv
):
    def _initialize_agent(self):
        qpos = np.array([0, 0.6, 0, 1.3, 0, 1.3, 0, 0, 0])
        qpos[:-2] += self._episode_rng.normal(0, 0.02, len(qpos) - 2)
        self._agent.reset(qpos)
        self._agent._robot.set_pose(Pose([-0.56, 0, 0]))
