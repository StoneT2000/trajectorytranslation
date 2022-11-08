from collections import OrderedDict

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qinverse, qmult, quat2axangle

from mani_skill2.utils.common import register_gym_env
from mani_skill2.utils.sapien_utils import look_at, vectorize_pose

from .base_env import FixedXmate3RobotiqEnv, PandaEnv


@register_gym_env(name="PlugChargerPanda-v0", max_episode_steps=200)
class PlugChargerPandaEnv(PandaEnv):
    _base_size = [2e-2, 1.5e-2, 1e-2]  # charger base half size
    _peg_size = [8e-3, 0.75e-3, 3.2e-3]  # charger peg half size
    _peg_gap = 6e-3  # charger peg gap
    _clearance = 2e-3  # single side clearance
    _receptacle_size = [1e-2, 5e-2, 5e-2]  # receptacle half size

    def _build_charger(self, peg_size, base_size, gap):
        builder = self._scene.create_actor_builder()

        # peg
        color = [1, 0.5, 0]
        builder.add_box_collision(Pose([peg_size[0], gap, 0]), peg_size)
        builder.add_box_visual(Pose([peg_size[0], gap, 0]), peg_size, color=color)
        builder.add_box_collision(Pose([peg_size[0], -gap, 0]), peg_size)
        builder.add_box_visual(Pose([peg_size[0], -gap, 0]), peg_size, color=color)

        # base
        color = [0, 0.5, 1]
        builder.add_box_collision(Pose([-base_size[0], 0, 0]), base_size)
        builder.add_box_visual(Pose([-base_size[0], 0, 0]), base_size, color=color)

        return builder.build(name="charger")

    def _build_receptacle(self, peg_size, receptacle_size, gap):
        builder = self._scene.create_actor_builder()

        sy = 0.5 * (receptacle_size[1] - peg_size[1] - gap)
        sz = 0.5 * (receptacle_size[2] - peg_size[2])
        dx = -receptacle_size[0]
        dy = peg_size[1] + gap + sy
        dz = peg_size[2] + sz
        poses = [
            Pose([dx, 0, dz]),
            Pose([dx, 0, -dz]),
            Pose([dx, dy, 0]),
            Pose([dx, -dy, 0]),
        ]
        half_sizes = [
            [receptacle_size[0], receptacle_size[1], sz],
            [receptacle_size[0], receptacle_size[1], sz],
            [receptacle_size[0], sy, receptacle_size[2]],
            [receptacle_size[0], sy, receptacle_size[2]],
        ]

        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(
                pose, half_size, material=self._render_materials["default"]
            )

        pose = Pose([-receptacle_size[0], 0, 0])
        half_size = [receptacle_size[0], gap - peg_size[1], peg_size[2]]
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(
            pose, half_size, material=self._render_materials["default"]
        )

        return builder.build_static(name="receptacle")

    def _load_actors(self):
        self._add_ground()
        self.charger = self._build_charger(
            self._peg_size,
            self._base_size,
            self._peg_gap,
        )
        self.receptacle = self._build_receptacle(
            [
                self._peg_size[0],
                self._peg_size[1] + self._clearance,
                self._peg_size[2] + self._clearance,
            ],
            self._receptacle_size,
            self._peg_gap,
        )

    def _initialize_actors(self):
        self.charger.set_pose(Pose([-0.05, 0.0, 1e-2]))
        self.receptacle.set_pose(Pose([0.05, 0, 1e-1], euler2quat(0, 0, np.pi)))
        self.goal_pose = Pose([0.05, 0.0, 1e-1])
        # self.charger.set_pose(self.goal_pose)

    def _get_obs_state_dict(self):
        env_state = OrderedDict()
        charger_pos = self.charger.pose.p
        charger_pos[0] -= self._base_size[0]  # to the center of charger base
        goal_pos = self.goal_pose.p
        goal_pos[0] -= self._base_size[0]
        env_state["charger_pos"] = charger_pos
        env_state["goal_pos"] = goal_pos

        state_dict = OrderedDict()
        state_dict.update(
            agent_state=self._agent.get_proprioception(),  # proprioception
            env_state=env_state,
        )
        return state_dict

    def check_success(self):
        obj_to_goal_dist, obj_to_goal_angle = self._compute_distance()
        # print(obj_to_goal_dist, obj_to_goal_angle)
        return obj_to_goal_dist <= 5e-3 and obj_to_goal_angle <= 0.2

    def _compute_distance(self):
        obj_pose = self.charger.pose
        obj_to_goal_pos = self.goal_pose.p - obj_pose.p
        obj_to_goal_dist = np.linalg.norm(obj_to_goal_pos)

        obj_to_goal_quat = qmult(qinverse(self.goal_pose.q), obj_pose.q)
        _, obj_to_goal_angle = quat2axangle(obj_to_goal_quat)
        assert obj_to_goal_angle >= 0.0, obj_to_goal_angle

        return obj_to_goal_dist, obj_to_goal_angle

    def compute_dense_reward(self):
        if self.check_success():
            reward = 5.0
        else:
            reward = 0.0

            cmass_pose = self.charger.pose.transform(self.charger.cmass_local_pose)
            gripper_to_obj_pos = cmass_pose.p - self.grasp_site.pose.p
            gripper_to_obj_dist = np.linalg.norm(gripper_to_obj_pos)
            reaching_reward = 1 - np.tanh(10.0 * gripper_to_obj_dist)
            reward += reaching_reward

            is_grasped = self._agent.check_grasp(self.charger)
            reward += 0.25 if is_grasped else 0.0

            if is_grasped:
                # coarse
                obj_to_goal_dist, obj_to_goal_angle = self._compute_distance()
                reward += 1 - np.tanh(10.0 * obj_to_goal_dist)
                if obj_to_goal_dist < 0.05:
                    reward += max(0.3 - obj_to_goal_angle, 0)

                # # fine-grained
                # obj_to_goal_dist, obj_to_goal_angle = self._compute_distance()
                # reward += 1 - np.tanh(10.0 * max(obj_to_goal_dist - 0.05, 0))
                # if obj_to_goal_dist < 0.05:
                #     reward += max(0.3 - obj_to_goal_angle, 0)
                #     obj_to_goal_pos = self.goal_pose.p - self.charger.pose.p
                #     dist_yz = np.linalg.norm(obj_to_goal_pos[1:], ord=1)
                #     # reward += 1 - np.tanh(50.0 * max(dist_yz - 1e-3, 0))
                #     reward += max(0.05 - dist_yz, 0) / 0.05
                #     if dist_yz < 1e-3:
                #         # reward += 1 - np.tanh(50.0 * np.abs(obj_to_goal_pos[0]))
                #         reward += max(0.05 - np.abs(obj_to_goal_pos[0]), 0) / 0.05

        return reward

    def _setup_cameras(self):
        self.render_camera = self._scene.add_camera(
            "render_camera", 512, 512, 1, 0.001, 10
        )
        self.render_camera.set_local_pose(look_at([0, -0.5, 0.5], [0, 0, 0]))

        self.third_view_camera = self._scene.add_camera(
            "third_view_camera", 128, 128, np.pi / 2, 0.001, 10
        )
        self.third_view_camera.set_local_pose(look_at([0, 0.5, 0.5], [0, 0, 0]))


@register_gym_env(name="PlugChargerFixedXmate3Robotiq-v0", max_episode_steps=200)
class PlugChargerFixedXmate3RobotiqEnv(FixedXmate3RobotiqEnv, PlugChargerPandaEnv):
    def _initialize_agent(self):
        qpos = np.array([0, 0.06, 0, 1.76, 0, 1.3, 1.57, 0, 0])
        qpos[:-2] += self._episode_rng.normal(0, 0.02, len(qpos) - 2)
        self._agent.reset(qpos)
        self._agent._robot.set_pose(Pose([-0.56, 0, 0]))
