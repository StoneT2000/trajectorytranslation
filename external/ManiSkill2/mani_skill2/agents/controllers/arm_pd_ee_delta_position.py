import numpy as np
import sapien.core as sapien
from transforms3d.quaternions import axangle2quat, qmult

from mani_skill2.agents.controllers.base_controller import BaseController


class ArmPDEEDeltaPositionController(BaseController):
    def __init__(
        self, controller_config: dict, robot: sapien.Articulation, control_freq: int
    ):
        super(ArmPDEEDeltaPositionController, self).__init__(
            controller_config, robot, control_freq
        )
        self.action_dimension = (
            3  # only the translation, the rotation is kept unchanged
        )
        self.control_type = "pos"
        assert not self.interpolate, "Do NOT support interpolation"

        self.ee_delta_pos_min = self.nums2array(
            controller_config["ee_delta_pos_min"], 3
        )
        self.ee_delta_pos_max = self.nums2array(
            controller_config["ee_delta_pos_max"], 3
        )
        self.joint_stiffness = self.nums2array(
            controller_config["joint_stiffness"], self.num_control_joints
        )
        self.joint_damping = self.nums2array(
            controller_config["joint_damping"], self.num_control_joints
        )
        self.joint_friction = self.nums2array(
            controller_config["joint_friction"], self.num_control_joints
        )

        self.pmodel = self.robot.create_pinocchio_model()
        self.qmask = np.zeros(self.robot.dof)
        self.qmask[self.control_joint_index] = 1
        self.target_joint_pos = self._get_curr_joint_pos()

    def set_joint_drive_property(self):
        for j_idx, j in enumerate(self.control_joints):
            j.set_drive_property(self.joint_stiffness[j_idx], self.joint_damping[j_idx])
            j.set_friction(self.joint_friction[j_idx])

    def reset(self):
        self.target_joint_pos = self._get_curr_joint_pos()

    @property
    def action_range(self) -> np.ndarray:
        return np.stack([self.ee_delta_pos_min, self.ee_delta_pos_max], axis=1)

    def compute_ik(self, target_pos):
        target_pose = self.robot.pose.inv().transform(
            sapien.Pose(target_pos, self.end_link.pose.q)
        )
        result, success, error = self.pmodel.compute_inverse_kinematics(
            self.end_link_idx,  # is it always also link idx?
            target_pose,
            initial_qpos=self.robot.get_qpos(),
            active_qmask=self.qmask,
            max_iterations=100,
        )
        # print(len(result), success, error)
        # print(result, self.robot.get_qpos())
        if success:
            return result[self.control_joint_index]
        else:
            return None

    def set_action(self, action: np.ndarray):
        assert action.shape[0] == self.action_dimension
        target_ee_pos = self.end_link.pose.p + action
        # print(self.target_ee_pos, self.end_link.pose.p)
        target_joint_pos = self.compute_ik(target_ee_pos)
        if target_joint_pos is not None:
            self.target_joint_pos = target_joint_pos

    def simulation_step(self):
        for j_idx, j in enumerate(self.control_joints):
            j.set_drive_target(self.target_joint_pos[j_idx])


class ArmPDEEDeltaPoseController(BaseController):
    def __init__(
        self, controller_config: dict, robot: sapien.Articulation, control_freq: int
    ):
        super().__init__(controller_config, robot, control_freq)
        self.action_dimension = 6
        self.control_type = "pos"
        assert not self.interpolate, "Do NOT support interpolation"

        self.ee_delta_pos_min = self.nums2array(
            controller_config["ee_delta_pos_min"], 3
        )
        self.ee_delta_pos_max = self.nums2array(
            controller_config["ee_delta_pos_max"], 3
        )
        self.ee_delta_rot_min = self.nums2array(
            controller_config["ee_delta_rot_min"], 3
        )
        self.ee_delta_rot_max = self.nums2array(
            controller_config["ee_delta_rot_max"], 3
        )
        self.joint_stiffness = self.nums2array(
            controller_config["joint_stiffness"], self.num_control_joints
        )
        self.joint_damping = self.nums2array(
            controller_config["joint_damping"], self.num_control_joints
        )
        self.joint_friction = self.nums2array(
            controller_config["joint_friction"], self.num_control_joints
        )

        self.pmodel = self.robot.create_pinocchio_model()
        self.qmask = np.zeros(self.robot.dof)
        self.qmask[self.control_joint_index] = 1
        self.target_joint_pos = self._get_curr_joint_pos()

    def set_joint_drive_property(self):
        for j_idx, j in enumerate(self.control_joints):
            j.set_drive_property(self.joint_stiffness[j_idx], self.joint_damping[j_idx])
            j.set_friction(self.joint_friction[j_idx])

    def reset(self):
        self.target_joint_pos = self._get_curr_joint_pos()

    @property
    def action_range(self) -> np.ndarray:
        lower_bound = np.hstack([self.ee_delta_pos_min, self.ee_delta_rot_min])
        upper_bound = np.hstack([self.ee_delta_pos_max, self.ee_delta_rot_max])
        return np.stack([lower_bound, upper_bound], axis=1)

    def compute_ik(self, target_pose):
        # Convert to robot base frame
        target_pose = self.robot.pose.inv().transform(target_pose)
        result, success, error = self.pmodel.compute_inverse_kinematics(
            self.end_link_idx,  # is it always also link idx?
            target_pose,
            initial_qpos=self.robot.get_qpos(),
            active_qmask=self.qmask,
            max_iterations=100,
        )
        # print(len(result), success, error)
        # print(result, self.robot.get_qpos())
        if success:
            return result[self.control_joint_index]
        else:
            return None

    def set_action(self, action: np.ndarray):
        assert action.shape[0] == self.action_dimension
        target_ee_pos = self.end_link.pose.p + action[0:3]

        angle = np.linalg.norm(action[3:6])
        if angle < 1e-6:
            axis = (0, 0, 1)
            angle = 0
        else:
            axis = action[3:6] / angle
        delta_quat = axangle2quat(axis, angle)
        target_ee_quat = qmult(self.end_link.pose.q, delta_quat)
        target_ee_pose = sapien.Pose(target_ee_pos, target_ee_quat)
        # print(self.target_ee_pos, self.end_link.pose.p)
        target_joint_pos = self.compute_ik(target_ee_pose)
        if target_joint_pos is not None:
            self.target_joint_pos = target_joint_pos

    def simulation_step(self):
        for j_idx, j in enumerate(self.control_joints):
            j.set_drive_target(self.target_joint_pos[j_idx])

class ArmPDEEDeltaPosFixedQuatController(ArmPDEEDeltaPoseController):
    def __init__(
        self, controller_config: dict, robot: sapien.Articulation, control_freq: int
    ):
        super().__init__(controller_config, robot, control_freq)
        self.action_dimension = 3

        from mani_skill2.utils.sapien_utils import get_actor_by_name
        self.end_link = get_actor_by_name(
            robot.get_links(), "panda_hand"
        )
        self.end_link_idx = 0
        for link_idx, link in enumerate(robot.get_links()):
            if link == self.end_link:
                self.end_link_idx = link_idx

    @property
    def action_range(self) -> np.ndarray:
        return np.stack([self.ee_delta_pos_min, self.ee_delta_pos_max], axis=1)

    def set_action(self, action: np.ndarray):
        assert action.shape[0] == self.action_dimension
        target_ee_pos = self.end_link.pose.p + action
        target_ee_quat = np.array([0, 1, 0, 0])
        target_ee_pose = sapien.Pose(target_ee_pos, target_ee_quat)
        # print(self.target_ee_pos, self.end_link.pose.p)
        target_joint_pos = self.compute_ik(target_ee_pose)
        if target_joint_pos is not None:
            self.target_joint_pos = target_joint_pos