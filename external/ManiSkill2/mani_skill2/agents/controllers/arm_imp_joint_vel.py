import numpy as np
import sapien.core as sapien

from mani_skill2.agents.controllers.base_controller import BaseController


class ArmImpJointVelController(BaseController):
    def __init__(
        self, controller_config: dict, robot: sapien.Articulation, control_freq: int
    ):
        super(ArmImpJointVelController, self).__init__(
            controller_config, robot, control_freq
        )
        self.control_type = "torque"

        self.joint_vel_min = self.nums2array(
            controller_config["joint_vel_min"], self.num_control_joints
        )
        self.joint_vel_max = self.nums2array(
            controller_config["joint_vel_max"], self.num_control_joints
        )
        self.joint_kd = None

        self.curr_joint_vel: np.ndarray = self._get_curr_joint_vel()
        self.start_joint_vel: np.ndarray = self.curr_joint_vel.copy()
        self.final_target_joint_vel: np.ndarray = self.curr_joint_vel.copy()

    def simulation_step(self):
        self._sync_articulation()
        self.curr_step += 1
        if self.interpolate:
            curr_target_joint_vel = (
                self.final_target_joint_vel - self.start_joint_vel
            ) / self.control_step * self.curr_step + self.start_joint_vel
        else:
            curr_target_joint_vel = self.final_target_joint_vel

        kd = self.joint_kd

        # torques = vel_err * kd
        vel_pos_error = curr_target_joint_vel - self._get_curr_joint_vel()
        desired_acc = np.multiply(vel_pos_error, kd)

        # Return desired torques. The passive force  compensation is handled by the combined controller.
        torques = np.dot(self._get_mass_matrix(), desired_acc)

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
        self.curr_joint_vel: np.ndarray = self._get_curr_joint_vel()
        self.start_joint_vel: np.ndarray = self.curr_joint_vel.copy()
        self.final_target_joint_vel: np.ndarray = self.curr_joint_vel.copy()


class ArmImpJointVelConstController(ArmImpJointVelController):
    def __init__(
        self, controller_config: dict, robot: sapien.Articulation, control_freq: int
    ):
        super(ArmImpJointVelConstController, self).__init__(
            controller_config, robot, control_freq
        )
        self.action_dimension = self.num_control_joints

        self.joint_kd = self.nums2array(
            controller_config["joint_kd"], self.num_control_joints
        )

    @property
    def action_range(self) -> np.ndarray:
        return np.stack([self.joint_vel_min, self.joint_vel_max], axis=1)

    def set_action(self, action: np.ndarray):
        assert action.shape[0] == self.action_dimension
        self.curr_step = 0
        self.start_joint_vel = self._get_curr_joint_vel()
        self.final_target_joint_vel = action


class ArmImpJointVelKdController(ArmImpJointVelController):
    def __init__(
        self, controller_config: dict, robot: sapien.Articulation, control_freq: int
    ):
        super(ArmImpJointVelKdController, self).__init__(
            controller_config, robot, control_freq
        )
        self.action_dimension = self.num_control_joints * 2

        self.joint_kd_min = self.nums2array(
            controller_config["joint_kd_min"], self.num_control_joints
        )
        self.joint_kd_max = self.nums2array(
            controller_config["joint_kd_max"], self.num_control_joints
        )

    @property
    def action_range(self) -> np.ndarray:
        return np.concatenate(
            [
                np.stack([self.joint_vel_min, self.joint_vel_max], axis=1),
                np.stack([self.joint_kd_min, self.joint_kd_max], axis=1),
            ]
        )

    def set_action(self, action: np.ndarray):
        assert action.shape[0] == self.action_dimension
        self.curr_step = 0
        self.start_joint_vel = self._get_curr_joint_vel()
        self.final_target_joint_vel = action[: self.num_control_joints]
        self.joint_kd = action[self.num_control_joints :]

    def reset(self):
        super(ArmImpJointVelKdController, self).reset()
        self.joint_kd = (self.joint_kd_min + self.joint_kd_max) / 2
