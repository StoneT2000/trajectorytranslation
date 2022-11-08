from pathlib import Path

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.euler import euler2quat

this_file = Path(__file__).parent.resolve()

AGENT_CONFIG_DIR = this_file / '../../assets/blockstack/agent_config'
from mani_skill2.utils.sapien_utils import get_entity_by_name, look_at

from .agents.float_panda import FloatPanda
from .agents.panda import Panda
from .sapien_env import BaseEnv


class PandaEnv(BaseEnv):
    SUPPORTED_OBS_MODES = ("state", "state_dict", "rgbd", "pointcloud")
    SUPPORTED_REWARD_MODES = ("dense", "sparse")
    _agent: Panda

    def _build_goal_site(self, radius=0.02, color=(0, 1, 0), name="goal_site"):
        """Build a sphere site (visual only) to indicate the goal position.
        Unhide the visual before rendering.
        """
        builder = self._scene.create_actor_builder()
        builder.add_sphere_visual(radius=radius, color=color)
        goal_site = builder.build_static(name)
        goal_site.hide_visual()
        return goal_site

    def _load_agent(self):
        self._agent = Panda.from_config_file(
            AGENT_CONFIG_DIR / "panda_4d_from_blockstack.yml", self._scene, self._control_freq
        )
        self.grasp_site: sapien.Link = get_entity_by_name(
            self._agent._robot.get_links(), "grasp_site"
        )

    def _initialize_agent(self):
        qpos = np.array(
            [0, np.pi / 16, 0, -np.pi * 5 / 6, 0, np.pi - 0.2, np.pi / 4, 0, 0]
        )
        qpos[:-2] += self._episode_rng.normal(0, 0.02, len(qpos) - 2)
        self._agent.reset(qpos)
        self._agent._robot.set_pose(Pose([-0.56, 0, 0]))

    def _setup_lighting(self):
        self._scene.set_ambient_light([0.3, 0.3, 0.3])
        self._scene.add_point_light([2, 2, 2], [1, 1, 1])
        self._scene.add_point_light([2, -2, 2], [1, 1, 1])
        self._scene.add_point_light([-2, 0, 2], [1, 1, 1])
        self._scene.add_directional_light([1, -1, -1], [0.3, 0.3, 0.3])
        self._scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _setup_cameras(self):
        # Camera only for rendering, not included in `_cameras`
        self.render_camera = self._scene.add_camera(
            "render_camera", 512, 512, 1, 0.001, 10
        )
        self.render_camera.set_local_pose(
            sapien.Pose([1.2, 0, 1.2], euler2quat(0, 0.5, 3.14))
        )

        # third_view_camera = self._scene.add_camera(
        #     "third_view_camera", 128, 128, np.pi / 2, 0.001, 10
        # )
        # third_view_camera.set_local_pose(look_at([0.2, 0, 0.4], [0, 0, 0]))
        # self._cameras["third_view_camera"] = third_view_camera

        ## fovy, near, far
        # test1 = self._scene.add_camera(
        #     'test_1', 128, 128, np.pi / 2, 0.001, 10
        # )
        # test1.set_local_pose(look_at([0.2,0,0.4], [0,0,0]))
        # self._cameras['test_1'] = test1

        # from left side
        state_visual = self._scene.add_camera(
            'state_visual', 512, 512, np.pi / 2.4, 0.001, 10
        )
        state_visual.set_local_pose(look_at([0.25,-0.4,0.5], [0.0,0.15,0.05]))
        self.state_visual = state_visual
        self._cameras['state_visual'] = state_visual

        # # from right side
        # test3 = self._scene.add_camera(
        #     'test_3', 128, 128, np.pi / 2, 0.001, 10
        # )
        # test3.set_local_pose(look_at([0,0.2,0.3], [0,0,0]))
        # self._cameras['test_3'] = test3

        # test4 = self._scene.add_camera(
        #     'test_4', 128, 128, np.pi / 3, 0.001, 10
        # )
        # test4.set_local_pose(look_at([0.2,0,0.4], [0,0,1]))
        # self._cameras['test_4'] = test4




    def _setup_viewer(self):
        super()._setup_viewer()
        self._viewer.set_camera_xyz(1.0, 0.0, 1.2)
        self._viewer.set_camera_rpy(0, -0.5, 3.14)

class FloatPandaEnv(PandaEnv):
    def _load_agent(self):
        self._agent = FloatPanda.from_config_file(
            AGENT_CONFIG_DIR / "float_panda.yml",
            self._scene,
            self._control_freq,
        )
        self.grasp_site: sapien.Link = get_entity_by_name(
            self._agent._robot.get_links(), "grasp_site"
        )

    def _initialize_agent(self):
        raise NotImplementedError()
