import numpy as np
from sapien.core import Pose

from mani_skill2.envs.fixed_single_articulation.open_cabinet_door import OpenCabinetDoor
from mani_skill2.utils.common import register_gym_env


@register_gym_env("FixedOpenCabinetDrawer-v0", max_episode_steps=500)
class OpenCabinetDrawer(OpenCabinetDoor):
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

    def __init__(self, *args, **kwargs):
        super(OpenCabinetDrawer, self).__init__(*args, **kwargs)

    def _initialize_agent(self):
        qpos = np.zeros(9)
        qpos[1] = 0.2
        qpos[:-2] += self._episode_rng.normal(0, 0.02, len(qpos) - 2)
        self._robot_init_qpos = qpos
        self._agent.reset(qpos)
        self._agent._robot.set_pose(Pose([-0.6, 0.4, 0]))
