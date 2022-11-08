from collections import OrderedDict

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.euler import euler2quat

from mani_skill2.utils.common import register_gym_env
from mani_skill2.utils.sapien_utils import vectorize_pose

from .base_env import FixedXmate3RobotiqEnv, PandaEnv


@register_gym_env("PickCubePanda-v0", max_episode_steps=200)
class PickCubePandaEnv(PandaEnv):
    def _build_cube(self, half_size=(0.02, 0.02, 0.02), color=(1, 0, 0), name="cube"):
        builder = self._scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size, color=color)
        return builder.build(name)

    def _load_actors(self):
        self._add_ground()
        self.cube = self._build_cube()
        self.goal_site = self._build_goal_site()

    def _initialize_goal(self, max_trials=100, verbose=False):
        cube_pos = self.cube.pose.p
        cube_z = cube_pos[2]

        for i in range(max_trials):
            goal_xy = self._episode_rng.uniform(-0.1, 0.1, [2])
            goal_z = self._episode_rng.uniform(0, 0.2) + cube_z
            goal_pos = np.hstack([goal_xy, goal_z])
            if np.linalg.norm(goal_pos - cube_pos) > 0.05:
                if verbose:
                    print(f"Found a valid goal at {i}-th trial")
                break

        self.goal_pos = goal_pos
        self.goal_site.set_pose(Pose(self.goal_pos))

    def _initialize_actors(self):
        cube_xy = self._episode_rng.uniform(-0.03, 0.03, [2])
        cube_xyz = np.hstack([cube_xy, 0.02])
        self.cube.set_pose(Pose(cube_xyz))

    def initialize_episode(self):
        super().initialize_episode()
        self._initialize_goal()

    def _get_obs_extra(self) -> OrderedDict:
        if self._obs_mode in ["rgbd", "pointcloud"]:
            return OrderedDict(
                gripper_pose=vectorize_pose(self.grasp_site.pose),
                goal_pos=self.goal_pos,
                gripper_to_goal_pos=self.goal_pos - self.grasp_site.pose.p,
            )
        else:
            return OrderedDict(
                cube_pose=vectorize_pose(self.cube.pose),
                gripper_pose=vectorize_pose(self.grasp_site.pose),
                gripper_to_cube_pos=self.cube.pose.p - self.grasp_site.pose.p,
                goal_pos=self.goal_pos,
                cube_to_goal_pos=self.goal_pos - self.cube.pose.p,
                gripper_to_goal_pos=self.goal_pos - self.grasp_site.pose.p,
            )

    def check_success(self):
        cube_to_goal_pos = self.goal_pos - self.cube.pose.p
        is_reached = np.linalg.norm(cube_to_goal_pos) < 0.02

        # The movement of the robot should be almost static
        qvel = self._agent._robot.get_qvel()
        is_robot_static = np.max(np.abs(qvel)) <= 0.2

        return is_reached and is_robot_static

    def compute_dense_reward(self):
        reward = 0.0

        if self.check_success():
            reward = 10
        else:
            gripper_to_cube_pos = self.cube.pose.p - self.grasp_site.pose.p
            gripper_to_cube_dist = np.linalg.norm(gripper_to_cube_pos)
            reaching_reward = 1 - np.tanh(10.0 * gripper_to_cube_dist)
            reward += reaching_reward

            is_grasped = self._agent.check_grasp(self.cube)
            reward += 0.25 if is_grasped else 0.0

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

    def get_state(self) -> np.ndarray:
        state = super().get_state()
        return np.hstack([state, self.goal_pos])

    def set_state(self, state):
        self.goal_pos = state[-3:]
        super().set_state(state[:-3])


@register_gym_env("PickCubeFixedXmate3Robotiq-v0", max_episode_steps=200)
class PickCubeFixedXmate3RobotiqEnv(FixedXmate3RobotiqEnv, PickCubePandaEnv):
    pass


@register_gym_env("PickCubeFromShelfPanda-v0", max_episode_steps=200)
class PickCubeFromShelfPandaEnv(PickCubePandaEnv):
    def _build_shelf(self, n, half_size, interval, name="shelf"):
        assert n > 1
        builder = self._scene.create_actor_builder()

        # each layer
        for i in range(n):
            pose = Pose([0, 0, i * interval])
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, color=(0.8, 0.5, 0.0))

        # back
        height = interval * (n - 1)
        pose = Pose([-half_size[0] - half_size[2], 0, height * 0.5])
        half_size2 = [half_size[2], half_size[1], 0.5 * height + half_size[2]]
        builder.add_box_collision(pose, half_size2)
        builder.add_box_visual(pose, half_size2, color=(0.8, 0.5, 0.0))

        # side
        half_size3 = [
            half_size[0] + half_size[2],
            half_size[2],
            0.5 * height + half_size[2],
        ]
        pose1 = Pose([-half_size[2], -half_size[1] - half_size[2], height * 0.5])
        builder.add_box_collision(pose1, half_size3)
        builder.add_box_visual(pose1, half_size3, color=(0.8, 0.5, 0.0))
        pose2 = Pose([-half_size[2], half_size[1] + half_size[2], height * 0.5])
        builder.add_box_collision(pose2, half_size3)
        builder.add_box_visual(pose2, half_size3, color=(0.8, 0.5, 0.0))

        return builder.build_static(name)

    def _load_actors(self):
        self._add_ground()
        self.cube = self._build_cube()
        self.n_layers = 2
        self.interval = 0.2
        self.shelf = self._build_shelf(
            self.n_layers + 1, [0.05, 0.15, 0.01], self.interval
        )
        self.goal_site = self._build_goal_site()

    def _initialize_actors(self):
        self.shelf.set_pose(Pose([0.2, 0, 0.01], euler2quat(0, 0, np.pi)))

        layer_idx = self._episode_rng.choice(self.n_layers)
        offset = layer_idx * self.interval + 0.01 * 2

        cube_xy = self._episode_rng.uniform(
            [-0.05 + 0.02, -0.15 + 0.02], [0.05 - 0.02, 0.15 - 0.02], [2]
        )
        cube_xy[0] += 0.2
        cube_xyz = np.hstack([cube_xy, 0.02 + offset])
        self.cube.set_pose(Pose(cube_xyz))

    def _initialize_goal(self):
        # TODO(jigu): should I use a fixed position?
        self.goal_pos = self.grasp_site.pose.p
        self.goal_site.set_pose(Pose(self.goal_pos))

    def _initialize_agent(self):
        qpos = np.array([0.027, 0.048, -0.027, -1.873, -0.006, 1.86, 0.785, 0, 0])
        qpos[:-2] += self._episode_rng.normal(0, 0.02, len(qpos) - 2)
        self._agent.reset(qpos)
        self._agent._robot.set_pose(Pose([-0.56, 0, 0]))
