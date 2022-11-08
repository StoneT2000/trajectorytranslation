from collections import OrderedDict
from typing import Dict, List

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.euler import euler2quat

from mani_skill2.utils.common import register_gym_env, register_gym_env_v1
from mani_skill2.utils.io import load_json
from mani_skill2.utils.sapien_utils import vectorize_pose

from .base_env import FixedXmate3RobotiqEnv, PandaEnv
from .utils import (
    EGAD_DIR,
    GRASPNET_DIR,
    OCRTOC_DIR,
    build_actor_egad,
    build_actor_graspnet,
    build_actor_orctoc,
    sample_scale,
)


class PickSinglePandaEnv(PandaEnv):
    obj: sapien.Actor  # target object
    DEFAULT_MODEL_JSON: str

    def __init__(
        self,
        model_ids: List[str] = (),
        model_json: str = None,
        init_rot_z=True,
        **kwargs,
    ):
        if model_json is None:
            model_json = self.DEFAULT_MODEL_JSON
        self.model_db: Dict[str, Dict] = load_json(model_json)
        if len(model_ids) == 0:
            model_ids = sorted(self.model_db.keys())
        assert len(model_ids) > 0, model_json
        self.model_ids = model_ids
        self.model_id = model_ids[0]
        self.model_scale = None

        self.init_rot_z = init_rot_z

        super().__init__(**kwargs)

    def reset(self, seed=None, reconfigure=False):
        self.set_episode_rng(seed)

        # -------------------------------------------------------------------------- #
        # Model selection
        # -------------------------------------------------------------------------- #
        if len(self.model_ids) > 1:
            next_model_id = self.model_ids[
                self._episode_rng.choice(len(self.model_ids))
            ]
            if next_model_id != self.model_id:
                self.model_id = next_model_id
                reconfigure = True

        next_model_scale = sample_scale(
            self.model_db[self.model_id]["scales"], rng=self._episode_rng
        )
        if next_model_scale != self.model_scale:
            self.model_scale = next_model_scale
            reconfigure = True
        # -------------------------------------------------------------------------- #

        if reconfigure:  # Reconfigure the scene if assets change
            self.reconfigure()
        else:
            self.set_sim_state(self._initial_sim_state)

        self.initialize_episode()

        return self.get_obs()

    def _load_actors(self):
        self._add_ground()
        self._load_model()
        self.goal_site = self._build_goal_site()

    def _load_model(self):
        """Load the target object."""
        raise NotImplementedError

    def _initialize_actors(self):
        obj_xy = self._episode_rng.uniform(-0.03, 0.03, [2])

        p = np.hstack([obj_xy, 1.0])
        q = [1, 0, 0, 0]
        if self.init_rot_z:
            ori = self._episode_rng.uniform(0, 2 * np.pi)
            q = euler2quat(0, 0, ori)

        self.obj.set_pose(Pose(p, q))
        self._agent._robot.set_pose(Pose([-10, 0, 0]))
        self.obj.lock_motion(False, False, False, True, True, False)
        for _ in range(self._sim_freq):
            self._scene.step()
        self.obj.lock_motion(False, False, False, False, False, False)
        for _ in range(self._sim_freq):
            self._scene.step()

        self._initialize_goal()

    def _initialize_goal(self, max_trials=100, verbose=False):
        # obj_pos = self.obj.pose.p
        obj_pos = self.obj.pose.transform(self.obj.cmass_local_pose).p

        for i in range(max_trials):
            goal_xy = self._episode_rng.uniform(-0.1, 0.1, [2])
            goal_z = self._episode_rng.uniform(0, 0.2) + obj_pos[2]
            goal_pos = np.hstack([goal_xy, goal_z])
            if np.linalg.norm(goal_pos - obj_pos) > 0.05:
                if verbose:
                    print(f"Found a valid goal at {i}-th trial")
                break

        # goal_pos = np.array([0.0, 0.0, 0.2])
        self.goal_pos = goal_pos
        self.goal_site.set_pose(Pose(self.goal_pos))

    def _get_obs_extra(self) -> OrderedDict:
        obj_pose = self.obj.pose.transform(self.obj.cmass_local_pose)
        if self._obs_mode in ["rgbd", "pointcloud"]:
            return OrderedDict(
                gripper_pose=vectorize_pose(self.grasp_site.pose),
                goal_pos=self.goal_pos,
                gripper_to_goal_pos=self.goal_pos - self.grasp_site.pose.p,
            )
        else:
            return OrderedDict(
                obj_pose=vectorize_pose(obj_pose),
                gripper_pose=vectorize_pose(self.grasp_site.pose),
                gripper_to_obj_pos=obj_pose.p - self.grasp_site.pose.p,
                goal_pos=self.goal_pos,
                obj_to_goal_pos=self.goal_pos - obj_pose.p,
                gripper_to_goal_pos=self.goal_pos - self.grasp_site.pose.p,
            )

    def check_success(self):
        obj_pose = self.obj.pose.transform(self.obj.cmass_local_pose)
        obj_to_goal_pos = self.goal_pos - obj_pose.p
        is_reached = np.linalg.norm(obj_to_goal_pos) < 0.02

        # The movement of the robot should be almost static
        qvel = self._agent._robot.get_qvel()
        is_robot_static = np.max(np.abs(qvel)) <= 0.2

        return is_reached and is_robot_static

    def compute_dense_reward(self):
        reward = 0.0

        if self.check_success():
            reward = 10.0
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


class PickSingleFixedXmate3RobotiqEnv(FixedXmate3RobotiqEnv, PickSinglePandaEnv):
    pass


@register_gym_env_v1(
    "PickSingleOCRTOCPanda-plate-@-v0",
    model_json=OCRTOC_DIR / "meta/info_ycb_others_v0.json",
    model_ids=["plate"],
)
@register_gym_env_v1(
    "PickSingleOCRTOCPanda-bowl-@-v0",
    model_json=OCRTOC_DIR / "meta/info_ycb_others_v0.json",
    model_ids=["bowl"],
)
@register_gym_env("PickSingleOCRTOCPanda-v0", max_episode_steps=200)
class PickSingleOCRTOCPandaEnv(PickSinglePandaEnv):
    DEFAULT_MODEL_JSON = OCRTOC_DIR / "meta/info_ycb_box_v0.json"

    def _load_model(self):
        self.obj = build_actor_orctoc(
            self.model_id, self._scene, scale=self.model_scale
        )
        self.obj.name = self.model_id


@register_gym_env("PickSingleEGADPanda-v0", max_episode_steps=200)
class PickSingleEGADPandaEnv(PickSinglePandaEnv):
    DEFAULT_MODEL_JSON = EGAD_DIR / "meta/info_eval_v0.json"

    def _load_model(self):
        self.obj = build_actor_egad(self.model_id, self._scene, scale=self.model_scale)
        self.obj.name = self.model_id


@register_gym_env("PickSingleGraspNetPanda-v0", max_episode_steps=200)
class PickGraspNetPandaEnv(PickSinglePandaEnv):
    DEFAULT_MODEL_JSON = GRASPNET_DIR / "meta/info_v0.json"

    def _load_model(self):
        self.obj = build_actor_graspnet(
            self.model_id, self._scene, scale=self.model_scale
        )
        self.obj.name = self.model_id
