from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import sapien.core as sapien
from sapien.core import Pose

from mani_skill2.utils.common import (
    random_choice,
    register_gym_env,
    register_gym_env_v1,
)

from .base_env import FixedXmate3RobotiqEnv, PandaEnv
from .utils import build_actor_hab_ycb, build_actor_orctoc


@register_gym_env_v1(
    "ServeFoodPanda-plate-@-v0",
    container_model_ids=["plate"],
    max_episode_steps=200,
)
@register_gym_env_v1(
    "ServeFoodPanda-bowl-@-v0",
    container_model_ids=["bowl"],
    max_episode_steps=200,
)
@register_gym_env("ServeFoodPanda-v0", max_episode_steps=200)
class ServeFoodPandaEnv(PandaEnv):
    _goal_pos = np.array([0, 0.3, 0.3])
    _all_container_model_ids = ("plate", "bowl")
    _all_food_model_ids = (
        "013_apple",
        "014_lemon",
        "015_peach",
        "016_pear",
        "017_orange",
        "018_plum",
    )

    def __init__(self, *args, container_model_ids=None, food_model_ids=None, **kwargs):
        if container_model_ids is None:
            container_model_ids = self._all_container_model_ids
        self.container_model_ids = container_model_ids
        if food_model_ids is None:
            food_model_ids = self._all_food_model_ids
        self.food_model_ids = food_model_ids
        self.container_model_id = None
        self.food_model_id = None
        self.container_model_idx = 0
        self.food_model_idx = 0

        super().__init__(*args, **kwargs)

    def reset(self, seed=None, reconfigure=False):
        self.set_episode_rng(seed)

        # -------------------------------------------------------------------------- #
        # Model selection
        # -------------------------------------------------------------------------- #
        container_model_id = random_choice(self.container_model_ids, self._episode_rng)
        food_model_id = random_choice(self.food_model_ids, self._episode_rng)
        if (
            container_model_id != self.container_model_id
            or food_model_id != self.food_model_id
        ):
            reconfigure = True

        self.container_model_id = container_model_id
        self.food_model_id = food_model_id
        self.container_model_idx = self._all_container_model_ids.index(
            container_model_id
        )
        self.food_model_idx = self._all_food_model_ids.index(food_model_id)

        # -------------------------------------------------------------------------- #

        if reconfigure:  # Reconfigure the scene if assets change
            self.reconfigure()
        else:
            self.set_sim_state(self._initial_sim_state)

        self.initialize_episode()

        return self.get_obs()

    def _build_cube(self, half_size, color=(1, 0, 0), name="cube", static=True):
        builder = self._scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size, color=color)
        if static:
            return builder.build_static(name)
        else:
            return builder.build(name)

    def _load_actors(self):
        self._add_ground()
        self.container = build_actor_orctoc(self.container_model_id, self._scene)
        self.food = build_actor_hab_ycb(self.food_model_id, self._scene)

        self.start_receptacle = self._build_cube(
            [0.05, 0.05, 0.15], [0.8, 0.5, 0], "start"
        )
        self.goal_receptacle = self._build_cube(
            [0.05, 0.05, 0.15], [0.5, 0.8, 0], "goal"
        )

    def _initialize_actors(self):
        x, y = self._episode_rng.uniform(-0.03, 0.03, [2])
        q = [1, 0, 0, 0]

        # Move obstacles away
        self.start_receptacle.set_pose(Pose([0, 10, 0]))
        self.goal_receptacle.set_pose(Pose([0, -10, 0]))
        # Move robot away
        self._agent._robot.set_pose(Pose([-10, 0, 0]))
        self.food.set_pose(Pose([-20, 0, 1.0]))
        self.food.lock_motion()

        self.container.set_pose(Pose([x, y, 0.2], q))
        self.container.lock_motion(True, True, False)
        for _ in range(self._sim_freq):
            self._scene.step()
            # self.render()
        self.container.lock_motion()

        self.food.set_pose(Pose([x, y, 0.2], q))
        self.food.lock_motion(True, True, False)
        for i in range(self._sim_freq):
            self._scene.step()
            # if i % 4 == 0:
            #     self.render()

        self.container.set_velocity(np.zeros(3))
        self.container.set_angular_velocity(np.zeros(3))
        self.container.lock_motion(*[False] * 6)
        self.food.set_velocity(np.zeros(3))
        self.food.set_angular_velocity(np.zeros(3))
        self.food.lock_motion(*[False] * 6)
        for _ in range(self._sim_freq):
            self._scene.step()

        # Place objects on receptacles
        self.start_receptacle.set_pose(Pose([0, -0.3, 0.15]))
        self.goal_receptacle.set_pose(Pose([0, 0.3, 0.15]))
        pose = self.container.pose
        self.container.set_pose(Pose(pose.p + [0, -0.3, 0.3], pose.q))
        pose = self.food.pose
        self.food.set_pose(Pose(pose.p + [0, -0.3, 0.3], pose.q))

    def check_success(self):

        # container
        offset = self.container.pose.p - self._goal_pos
        is_container_placed = (
            np.linalg.norm(offset[0:2]) < 0.05 and (offset[2] - 0.4) < 0.05
        )

        # food
        offset = self.food.pose.p - self._goal_pos
        is_food_placed = np.linalg.norm(offset[0:2]) < 0.05 and (offset[2] - 0.4) < 0.05

        return is_container_placed and is_food_placed

    def compute_dense_reward(self):
        reward = 0.0
        return reward

    def _get_obs_extra(self) -> OrderedDict:
        extra_dict = OrderedDict()
        extra_dict["goal_pos"] = self._goal_pos
        extra_dict["container_pos"] = self.container.pose.p
        extra_dict["container_idx"] = np.array([self.container_model_idx])
        extra_dict["food_idx"] = np.array([self.food_model_idx])

        return extra_dict

    def _initialize_agent(self):
        qpos = np.array(
            [0, np.pi / 16, 0, -np.pi * 2 / 6, 0, np.pi - 0.2, np.pi / 4, 0, 0]
        )
        qpos[:-2] += self._episode_rng.normal(0, 0.02, len(qpos) - 2)
        self._agent.reset(qpos)
        self._agent._robot.set_pose(Pose([-0.56, 0, 0]))


@register_gym_env("ServeFoodFixedXmate3Robotiq-v0", max_episode_steps=200)
class ServeFoodFixedXmate3RobotiqEnv(FixedXmate3RobotiqEnv, ServeFoodPandaEnv):
    def _initialize_agent(self):
        qpos = np.array([0, 0.0, 0, 0.0, 0, 1.3, -1.57, 0, 0])
        qpos[:-2] += self._episode_rng.normal(0, 0.02, len(qpos) - 2)
        self._agent.reset(qpos)
        self._agent._robot.set_pose(Pose([-0.66, 0, 0]))
