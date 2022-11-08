import numpy as np
import sapien.core as sapien
import transforms3d as t3d

from mani_skill2.agents.control_utils import (
    nullspace_torques,
    opspace_matrices,
    orientation_error,
)
from mani_skill2.agents.controllers.base_controller import BaseController


class ArmImpEEPosController(BaseController):
    def __init__(
        self, controller_config: dict, robot: sapien.Articulation, control_freq: int
    ):
        super(ArmImpEEPosController, self).__init__(
            controller_config, robot, control_freq
        )
        self.control_type = "torque"
        self.use_delta = controller_config[
            "use_delta"
        ]  # interpret action as delta or absolute pose

        if self.use_delta:
            self.ee_delta_pos_min = self.nums2array(
                controller_config["ee_delta_pos_min"], 6
            )
            self.ee_delta_pos_max = self.nums2array(
                controller_config["ee_delta_pos_max"], 6
            )
        else:
            self.ee_pos_min = self.nums2array(controller_config["ee_pos_min"], 6)
            self.ee_pos_max = self.nums2array(controller_config["ee_pos_max"], 6)
        self.ee_kp = None
        self.ee_kd = None

        curr_base_pose_RT = self.start_link.get_pose().to_transformation_matrix()
        curr_ee_pose_RT = self.end_link.get_pose().to_transformation_matrix()
        self.start_rel_pose_RT = np.linalg.inv(curr_base_pose_RT) @ curr_ee_pose_RT
        self.final_rel_pose_RT = self.start_rel_pose_RT.copy()
        self.delta_translation = np.zeros(3)
        self.delta_axis = np.array([1.0, 0.0, 0.0])
        self.delta_angle = 0.0
        self.init_joint_pos = self._get_curr_joint_pos()

    def set_action(self, action: np.ndarray):
        assert action.shape[0] == self.action_dimension
        self._sync_articulation()
        self.curr_step = 0
        curr_base_pose_RT = self.start_link.get_pose().to_transformation_matrix()
        curr_ee_pose_RT = self.end_link.get_pose().to_transformation_matrix()
        self.start_rel_pose_RT = np.linalg.inv(curr_base_pose_RT) @ curr_ee_pose_RT
        RT = np.eye(4)
        angle = np.linalg.norm(action[3:6])
        if angle < 1e-6:
            axis = (0, 0, 1)
        else:
            axis = action[3:6] / angle
        RT[:3, :3] = t3d.axangles.axangle2mat(axis, angle)
        RT[:3, 3] = action[:3]
        if self.use_delta:
            self.final_rel_pose_RT = self.start_rel_pose_RT @ RT
        else:
            self.final_rel_pose_RT = RT  #
        delta_RT = np.linalg.inv(self.start_rel_pose_RT) @ self.final_rel_pose_RT
        self.delta_translation = delta_RT[:3, 3]
        self.delta_axis, self.delta_angle = t3d.axangles.mat2axangle(delta_RT[:3, :3])

    def simulation_step(self):
        self._sync_articulation()
        self.curr_step += 1
        curr_base_pose_RT = self.start_link.get_pose().to_transformation_matrix()
        curr_ee_pose_RT = self.end_link.get_pose().to_transformation_matrix()
        curr_rel_pose_RT = np.linalg.inv(curr_base_pose_RT) @ curr_ee_pose_RT
        if self.interpolate:
            RT = np.eye(4)
            RT[:3, 3] = self.delta_translation / self.control_step * self.curr_step
            RT[:3, :3] = t3d.axangles.axangle2mat(
                self.delta_axis, self.delta_angle / self.control_step * self.curr_step
            )
            curr_target_pose_RT = self.start_rel_pose_RT @ RT
        else:
            curr_target_pose_RT = self.final_rel_pose_RT

        err_pos = curr_target_pose_RT[:3, 3] - curr_rel_pose_RT[:3, 3]
        err_ori = orientation_error(
            curr_target_pose_RT[:3, :3], curr_rel_pose_RT[:3, :3]
        )
        J_full = self.controller_articulation.compute_world_cartesian_jacobian()[-6:]

        J_pos, J_ori = J_full[:3, :], J_full[3:, :]
        vel = J_full @ self.controller_articulation.get_qvel()

        # Compute desired force and torque based on errors
        vel_pos_error = -vel[:3]

        # F_r = kp * pos_err + kd * vel_err
        desired_force = np.multiply(
            np.array(err_pos), np.array(self.ee_kp[0:3])
        ) + np.multiply(vel_pos_error, self.ee_kd[0:3])

        vel_ori_error = -vel[3:]

        # Tau_r = kp * ori_err + kd * vel_err
        desired_torque = np.multiply(
            np.array(err_ori), np.array(self.ee_kp[3:6])
        ) + np.multiply(vel_ori_error, self.ee_kd[3:6])

        mass_matrix = self._get_mass_matrix()
        # Compute nullspace matrix (I - Jbar * J) and lambda matrices ((J * M^-1 * J^T)^-1)
        lambda_full, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(
            mass_matrix, J_full, J_pos, J_ori
        )

        desired_wrench = np.concatenate([desired_force, desired_torque])
        decoupled_wrench = np.dot(lambda_full, desired_wrench)

        # Gamma (without null torques) = J^T * F. The passive force  compensation is handled by the combined controller.
        torques = np.dot(J_full.T, decoupled_wrench)

        # Calculate and add nullspace torques (nullspace_matrix^T * Gamma_null) to final torques
        # Note: Gamma_null = desired nullspace pose torques, assumed to be positional joint control relative
        #                     to the initial joint positions
        torques += nullspace_torques(
            mass_matrix,
            nullspace_matrix,
            self.init_joint_pos,
            self._get_curr_joint_pos(),
            self._get_curr_joint_vel(),
        )

        qf = np.zeros(len(self.robot.get_active_joints()))
        qf[self.control_joint_index] = torques
        return qf

    def _get_mass_matrix(self):
        return self.controller_articulation.compute_manipulator_inertia_matrix()

    def set_joint_drive_property(self):
        # clear joint drive property and use pure torque control
        for j in self.control_joints:
            j.set_drive_property(0.0, 0.0)
            j.set_friction(0.0)

    def reset(self):
        self.init_joint_pos = self._get_curr_joint_pos()
        curr_base_pose_RT = self.start_link.get_pose().to_transformation_matrix()
        curr_ee_pose_RT = self.end_link.get_pose().to_transformation_matrix()
        self.start_rel_pose_RT = np.linalg.inv(curr_base_pose_RT) @ curr_ee_pose_RT
        self.final_rel_pose_RT = self.start_rel_pose_RT.copy()


class ArmImpEEPosConstController(ArmImpEEPosController):
    action_dimension = 6  # 3 for translation, 3 for rotation in axis-angle

    def __init__(
        self, controller_config: dict, robot: sapien.Articulation, control_freq: int
    ):
        super(ArmImpEEPosConstController, self).__init__(
            controller_config, robot, control_freq
        )

        self.ee_kp = self.nums2array(controller_config["ee_kp"], 6)
        self.ee_kd = self.nums2array(controller_config["ee_kd"], 6)

    @property
    def action_range(self) -> np.ndarray:
        if self.use_delta:
            return np.stack([self.ee_delta_pos_min, self.ee_delta_pos_max], axis=1)
        else:
            return np.stack([self.ee_pos_min, self.ee_pos_max], axis=1)


class ArmImpEEPosKpController(ArmImpEEPosController):
    action_dimension = 12

    def __init__(
        self, controller_config: dict, robot: sapien.Articulation, control_freq: int
    ):
        super(ArmImpEEPosKpController, self).__init__(
            controller_config, robot, control_freq
        )
        self.ee_kp_min = self.nums2array(controller_config["ee_kp_min"], 6)
        self.ee_kp_max = self.nums2array(controller_config["ee_kp_max"], 6)
        self.ee_kd = self.nums2array(controller_config["ee_kd"], 6)

    @property
    def action_range(self) -> np.ndarray:
        if self.use_delta:
            return np.concatenate(
                [
                    np.stack([self.ee_delta_pos_min, self.ee_delta_pos_max], axis=1),
                    np.stack([self.ee_kp_min, self.ee_kp_max], axis=1),
                ]
            )
        else:
            return np.concatenate(
                [
                    np.stack([self.ee_pos_min, self.ee_pos_max], axis=1),
                    np.stack([self.ee_kp_min, self.ee_kp_max], axis=1),
                ]
            )

    def set_action(self, action: np.ndarray):
        super(ArmImpEEPosKpController, self).set_action(action)
        self.ee_kp = action[6:12]

    def reset(self):
        super(ArmImpEEPosKpController, self).reset()
        self.ee_kp = (self.ee_kp_min + self.ee_kp_max) / 2


class ArmImpEEPosKpKdController(ArmImpEEPosController):
    action_dimension = 18

    def __init__(
        self, controller_config: dict, robot: sapien.Articulation, control_freq: int
    ):
        super(ArmImpEEPosKpKdController, self).__init__(
            controller_config, robot, control_freq
        )
        self.ee_kp_min = self.nums2array(controller_config["ee_kp_min"], 6)
        self.ee_kp_max = self.nums2array(controller_config["ee_kp_max"], 6)
        self.ee_kd_min = self.nums2array(controller_config["ee_kd_min"], 6)
        self.ee_kd_max = self.nums2array(controller_config["ee_kd_max"], 6)

    @property
    def action_range(self) -> np.ndarray:
        if self.use_delta:
            return np.concatenate(
                [
                    np.stack([self.ee_delta_pos_min, self.ee_delta_pos_max], axis=1),
                    np.stack([self.ee_kp_min, self.ee_kp_max], axis=1),
                    np.stack([self.ee_kd_min, self.ee_kd_max], axis=1),
                ]
            )
        else:
            return np.concatenate(
                [
                    np.stack([self.ee_pos_min, self.ee_pos_max], axis=1),
                    np.stack([self.ee_kp_min, self.ee_kp_max], axis=1),
                    np.stack([self.ee_kd_min, self.ee_kd_max], axis=1),
                ]
            )

    def set_action(self, action: np.ndarray):
        super(ArmImpEEPosKpKdController, self).set_action(action)
        self.ee_kp = action[6:12]
        self.ee_kd = action[12:]

    def reset(self):
        super(ArmImpEEPosKpKdController, self).reset()
        self.ee_kp = (self.ee_kp_min + self.ee_kp_max) / 2
        self.ee_kd = (self.ee_kd_min + self.ee_kd_max) / 2
