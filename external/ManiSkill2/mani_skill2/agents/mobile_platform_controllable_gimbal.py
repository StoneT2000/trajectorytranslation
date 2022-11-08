from collections import OrderedDict

import numpy as np
import sapien.core as sapien

from mani_skill2.agents.base_agent import BaseAgent
from mani_skill2.utils.sapien_utils import get_actor_by_name


class MobilePlatformControllableGimbal(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mobile_link: sapien.Actor = get_actor_by_name(
            self._robot.get_links(), "base_link"
        )

    def get_proprioception(self):
        state_dict = OrderedDict()
        qpos = self._robot.get_qpos()
        qvel = self._robot.get_qvel()

        mobile_pos_p, mobile_pos_q = (
            self._mobile_link.get_pose().p,
            self._mobile_link.get_pose().q,
        )
        cam_pos_p, cam_pos_q = (
            self._cameras["platform_d415"].get_pose().p,
            self._cameras["platform_d415"].get_pose().q,
        )

        state_dict["qpos"] = qpos
        state_dict["qvel"] = qvel
        state_dict["mobile_pos_p"] = mobile_pos_p
        state_dict["mobile_pos_q"] = mobile_pos_q
        state_dict["mobile_tra_vel"] = np.sqrt(qvel[0] ** 2 + qvel[1] ** 2)
        state_dict["mobile_rot_vel"] = qvel[2]
        state_dict["cam_pos_p"] = cam_pos_p
        state_dict["cam_pos_q"] = cam_pos_q

        return state_dict
