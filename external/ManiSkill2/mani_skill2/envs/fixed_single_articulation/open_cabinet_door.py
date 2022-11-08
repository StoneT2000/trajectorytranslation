import copy
from collections import OrderedDict
from typing import Dict

import numpy as np
import sapien.core as sapien
import trimesh
from sapien.core import Pose

from mani_skill2 import ASSET_DIR
from mani_skill2.envs.fixed_single_articulation.base_env import FixedXmate3RobotiqEnv
from mani_skill2.utils.common import (
    convert_np_bool_to_float,
    flatten_state_dict,
    register_gym_env,
)
from mani_skill2.utils.contrib import (
    apply_pose_to_points,
    normalize_and_clip_in_interval,
    o3d_to_trimesh,
    trimesh_to_o3d,
)
from mani_skill2.utils.geometry import angle_distance
from mani_skill2.utils.o3d_utils import merge_mesh, np2mesh


@register_gym_env("FixedOpenCabinetDoor-v0", max_episode_steps=500)
class OpenCabinetDoor(FixedXmate3RobotiqEnv):
    SUPPORTED_OBS_MODES = ("state", "state_dict", "rgbd")
    SUPPORTED_REWARD_MODES = ("dense", "sparse")

    _articulation_init_pos_min_x = 0.0
    _articulation_init_pos_max_x = 0.1
    _articulation_init_pos_min_y = 0.1
    _articulation_init_pos_max_y = 0.2
    _articulation_init_rot_min_z = 0.04
    _articulation_init_rot_max_z = 0.2
    _init_open_extent_range = (
        0.1  # the target joint is set to random open extent at reset()
    )

    _joint_friction_range = (0.05, 0.15)
    _joint_stiffness_range = (0.0, 0.0)
    _joint_damping_range = (5.0, 20.0)

    _open_extent = 0.3
    max_v = 0.1
    max_ang_v = 1.0

    _robot_init_qpos: np.ndarray

    def __init__(
        self,
        articulation_config_path="",
        obs_mode=None,
        reward_mode=None,
        sim_freq=500,
        control_freq=20,
    ):
        super().__init__(
            articulation_config_path, obs_mode, reward_mode, sim_freq, control_freq
        )

    def _initialize_articulation(self):
        pos_x = self._episode_rng.uniform(
            self._articulation_init_pos_min_x, self._articulation_init_pos_max_x
        )
        pos_y = self._episode_rng.uniform(
            self._articulation_init_pos_min_y, self._articulation_init_pos_max_y
        )
        rot_z = self._episode_rng.uniform(
            self._articulation_init_rot_min_z, self._articulation_init_rot_max_z
        )
        self._articulation.set_root_pose(
            Pose(
                [
                    pos_x,
                    pos_y,
                    -self._articulation_config.scale
                    * self._articulation_config.bbox_min[2],
                ],
                [np.sqrt(1 - rot_z ** 2), 0, 0, rot_z],
            )
        )

        [[lmin, lmax]] = self._target_joint.get_limits()
        init_open_extent = self._episode_rng.uniform(0, self._init_open_extent_range)
        qpos = np.zeros(self._articulation.dof)
        for i in range(self._articulation.dof):
            qpos[i] = self._articulation.get_active_joints()[i].get_limits()[0][0]
        qpos[self._articulation_config.target_joint_idx] = (
            lmin + (lmax - lmin) * init_open_extent
        )
        self._articulation.set_qpos(qpos)

        self.target_qpos = lmin + (lmax - lmin) * self._open_extent

        # set physical properties for all the joints
        joint_friction = self._episode_rng.uniform(
            self._joint_friction_range[0], self._joint_friction_range[1]
        )
        joint_stiffness = self._episode_rng.uniform(
            self._joint_stiffness_range[0], self._joint_stiffness_range[1]
        )
        joint_damping = self._episode_rng.uniform(
            self._joint_damping_range[0], self._joint_damping_range[1]
        )

        for joint in self._articulation.get_active_joints():
            joint.set_friction(joint_friction)
            joint.set_drive_property(joint_stiffness, joint_damping)

        self._get_handle_info_in_target_link()

    def _get_handle_info_in_target_link(self):
        """build a mesh and a point cloud of handle in target link,
        find visual_body_ids for observation
        compute grasp poses of handle
        """
        link = self._target_link
        meshes = []
        visual_body_ids = []
        for visual_body in link.get_visual_bodies():
            if "handle" not in visual_body.get_name():
                continue
            visual_body_ids.append(visual_body.get_visual_id())
            for render_shape in visual_body.get_render_shapes():
                vertices = apply_pose_to_points(
                    render_shape.mesh.vertices * visual_body.scale,
                    visual_body.local_pose,
                )
                mesh = np2mesh(vertices, render_shape.mesh.indices.reshape(-1, 3))
                meshes.append(mesh)
        assert visual_body_ids, "Can NOT find handle in the target link."
        mesh = merge_mesh(meshes)
        mesh = trimesh.convex.convex_hull(o3d_to_trimesh(mesh))
        pcd = mesh.sample(100)
        pcd_world = apply_pose_to_points(
            pcd, link.get_pose()
        )  # transform to world frame to compute grasp pose
        bbox_size = (pcd_world.max(0) - pcd_world.min(0)) / 2
        center = (pcd_world.max(0) + pcd_world.min(0)) / 2

        R = link.get_pose().to_transformation_matrix()[:3, :3]
        # choose the axis closest to X
        idx = np.argmax(np.abs(R[0]))
        forward = R[:3, idx]
        if forward[0] < 0:
            forward *= -1
        if bbox_size[1] > bbox_size[2]:
            flat = np.array([0, 0, 1])
        else:
            flat = np.cross(forward, np.array([0, 0, 1]))
        grasp_pose = (
            link.get_pose().inv() * self._agent.build_grasp_pose(forward, flat, center),
            link.get_pose().inv()
            * self._agent.build_grasp_pose(forward, -flat, center),
        )
        self._handle_info = OrderedDict()
        self._handle_info["mesh"] = trimesh_to_o3d(mesh)
        self._handle_info["pcd"] = pcd
        self._handle_info["grasp"] = grasp_pose
        self._handle_info["visual_body_ids"] = visual_body_ids

    def _initialize_agent(self):
        qpos = np.zeros(9)
        qpos[1] = 0.3
        qpos[3] = 0.3
        qpos[6] = 1.57

        self._robot_init_qpos = qpos
        self._agent.reset(qpos)
        self._agent._robot.set_pose(Pose([-0.6, 0.4, 0]))

    def _get_obs_state_dict(self) -> Dict:
        state_dict = OrderedDict()
        state_dict.update(
            agent_state=self._agent.get_proprioception(),  # proprioception
        )
        state_dict.update(eval_flag_dict=self.compute_eval_flag_dict())
        state_dict.update(other_flag_dict=self.compute_other_flag_dict())
        return state_dict

    def _get_obs_rgbd(self) -> Dict:
        obs_dict = OrderedDict()
        images = self._agent.get_images(depth=True, visual_seg=True)
        obs_dict.update(images)
        # generate handle mask
        for cam_name, cam_images in images.items():
            masks = []
            for handle_id in self._handle_info["visual_body_ids"]:
                masks.append(cam_images["visual_seg"] == handle_id)
            mask = np.logical_or.reduce(masks)
            obs_dict[cam_name]["handle_mask"] = mask

        # TODO (ruic): Do we need object mask?

        obs_dict.update(
            agent_state=self._agent.get_proprioception(),  # proprioception
        )

        return obs_dict

    def get_obs(self):
        state_dict = self._get_obs_state_dict()
        self._cache_obs_state_dict = copy.deepcopy(state_dict)
        if self._obs_mode == "state":
            return flatten_state_dict(state_dict)
        elif self._obs_mode == "state_dict":
            return state_dict
        elif self._obs_mode == "rgbd":
            rgbd_dict = self._get_obs_rgbd()
            rgbd_dict.update(state_dict)
            return rgbd_dict
        else:
            raise NotImplementedError(self._obs_mode)

    def compute_other_flag_dict(self):
        ee_cords = self._agent.sample_ee_coords()  # [2, 10, 3]
        current_handle = apply_pose_to_points(
            self._handle_info["pcd"], self._target_link.get_pose()
        )  # [200, 3]
        ee_to_handle = ee_cords[..., None, :] - current_handle.reshape(1, 1, -1, 3)
        dist_ee_to_handle = np.linalg.norm(ee_to_handle, axis=-1).min((1, 2))  # [2]

        handle_mesh = trimesh.Trimesh(
            vertices=apply_pose_to_points(
                np.asarray(self._handle_info["mesh"].vertices),
                self._target_link.get_pose(),
            ),
            faces=np.asarray(np.asarray(self._handle_info["mesh"].triangles)),
        )

        dist_ee_mid_to_handle = (
            trimesh.proximity.ProximityQuery(handle_mesh)
            .signed_distance(ee_cords.mean(0))
            .max()
        )

        ee_close_to_handle = (
            dist_ee_to_handle.max() <= 0.01 and dist_ee_mid_to_handle > 0
        )
        other_info = OrderedDict()
        other_info["dist_ee_to_handle"] = dist_ee_to_handle
        other_info["dist_ee_mid_to_handle"] = np.array([dist_ee_mid_to_handle])
        other_info["ee_close_to_handle"] = convert_np_bool_to_float(ee_close_to_handle)
        return other_info

    def compute_eval_flag_dict(self):
        flag_dict = OrderedDict()
        flag_dict["cabinet_static"] = convert_np_bool_to_float(
            self.check_actor_static(
                self._target_link, max_v=self.max_v, max_ang_v=self.max_ang_v
            )
        )
        flag_dict["open_enough"] = convert_np_bool_to_float(
            self._articulation.get_qpos()[self._articulation_config.target_joint_idx]
            >= self.target_qpos
        )
        flag_dict["success"] = convert_np_bool_to_float(all(flag_dict.values()))
        return flag_dict

    def check_success(self):
        if self._cache_obs_state_dict:
            return self._cache_obs_state_dict["eval_flag_dict"]["success"]
        flag_dict = self.compute_eval_flag_dict()
        return flag_dict["success"]

    def compute_dense_reward(self):
        if not self._cache_obs_state_dict:
            self.get_obs()
        obs_state_dict = self._cache_obs_state_dict
        flag_dict = obs_state_dict["eval_flag_dict"]
        other_info = obs_state_dict["other_flag_dict"]
        dist_ee_to_handle = other_info["dist_ee_to_handle"]
        dist_ee_mid_to_handle = other_info["dist_ee_mid_to_handle"]

        grasp_site_pose = self._agent.grasp_site.get_pose()
        target_pose = self._target_link.get_pose() * self._handle_info["grasp"][0]
        target_pose_2 = self._target_link.get_pose() * self._handle_info["grasp"][1]

        angle1 = angle_distance(grasp_site_pose, target_pose)
        angle2 = angle_distance(grasp_site_pose, target_pose_2)
        gripper_angle_err = min(angle1, angle2) / np.pi

        cabinet_vel = self._articulation.get_qvel()[self._target_joint_idx]

        gripper_vel_norm = np.linalg.norm(self._agent.grasp_site.get_velocity())
        gripper_ang_vel_norm = np.linalg.norm(
            self._agent.grasp_site.get_angular_velocity()
        )
        gripper_vel_rew = -(gripper_vel_norm + gripper_ang_vel_norm * 0.5)

        scale = 1
        vel_coefficient = 1.5
        dist_coefficient = 0.5

        gripper_angle_rew = -gripper_angle_err * 3

        rew_ee_handle = -dist_ee_to_handle.mean() * 2
        rew_ee_mid_handle = (
            normalize_and_clip_in_interval(dist_ee_mid_to_handle, -0.01, 4e-3) - 1
        )

        reward = (
            gripper_angle_rew
            + rew_ee_handle
            + rew_ee_mid_handle
            - (dist_coefficient + vel_coefficient)
        )
        stage_reward = -(5 + vel_coefficient + dist_coefficient)

        vel_reward = 0
        dist_reward = 0

        if other_info["ee_close_to_handle"]:
            stage_reward += 0.5
            vel_reward = (
                normalize_and_clip_in_interval(cabinet_vel, -0.1, 0.5) * vel_coefficient
            )  # Push vel to positive
            dist_reward = (
                normalize_and_clip_in_interval(
                    self._articulation.get_qpos()[self._target_joint_idx],
                    0,
                    self.target_qpos,
                )
                * dist_coefficient
            )
            reward += dist_reward + vel_reward
            if flag_dict["open_enough"]:
                stage_reward += vel_coefficient + 2
                reward = reward - vel_reward + gripper_vel_rew
                if flag_dict["cabinet_static"]:
                    stage_reward += 1
        info_dict = {
            "dist_ee_to_handle": dist_ee_to_handle,
            "angle1": angle1,
            "angle2": angle2,
            "dist_ee_mid_to_handle": dist_ee_mid_to_handle,
            "rew_ee_handle": rew_ee_handle,
            "rew_ee_mid_handle": rew_ee_mid_handle,
            "qpos_rew": dist_reward,
            "qvel_rew": vel_reward,
            "gripper_angle_err": gripper_angle_err * 180,
            "gripper_angle_rew": gripper_angle_rew,
            "gripper_vel_norm": gripper_vel_norm,
            "gripper_ang_vel_norm": gripper_ang_vel_norm,
            "qpos": self._articulation.get_qpos()[self._target_joint_idx],
            "qvel": cabinet_vel,
            "target_qpos": self.target_qpos,
            "reward_raw": reward,
            "stage_reward": stage_reward,
        }
        reward = (reward + stage_reward) * scale
        self._cache_info = info_dict
        return reward

    def get_reward(self):
        if self._reward_mode == "sparse":
            return float(self.check_success())
        elif self._reward_mode == "dense":
            return self.compute_dense_reward()
        else:
            raise NotImplementedError(self._reward_mode)

    def get_info(self):
        info = super().get_info()
        info.update(self._cache_info)
        return info

    @property
    def handle_info(self):
        return self._handle_info

    @property
    def table(self):
        return self._table
