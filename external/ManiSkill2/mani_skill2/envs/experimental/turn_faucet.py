from collections import OrderedDict
from typing import Dict, List, Sequence

import numpy as np
import sapien.core as sapien
import trimesh
from sapien.core import Pose
from scipy.spatial.distance import cdist
from transforms3d.euler import euler2quat

from mani_skill2 import AGENT_CONFIG_DIR, ASSET_DIR
from mani_skill2.agents.panda import Panda
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.common import np_random, random_choice, register_gym_env
from mani_skill2.utils.geometry import transform_points
from mani_skill2.utils.io import load_json
from mani_skill2.utils.sapien_utils import get_entity_by_name, look_at, vectorize_pose
from mani_skill2.utils.trimesh_utils import get_actor_meshes, merge_meshes


class PandaEnv(BaseEnv):
    SUPPORTED_OBS_MODES = ("state", "state_dict", "rgbd", "pointcloud")
    SUPPORTED_REWARD_MODES = ("dense", "sparse")

    _agent: Panda

    def _load_actors(self):
        self._add_ground()

    def _load_agent(self):
        self._agent = Panda.from_config_file(
            AGENT_CONFIG_DIR / "panda.yml", self._scene, self._control_freq
        )
        self.grasp_site: sapien.Link = get_entity_by_name(
            self._agent._robot.get_links(), "grasp_site"
        )

    def _initialize_agent(self):
        # qpos = np.array([0, -0.785, 0, -2.356, 0, 1.57, 0.785, 0, 0])
        qpos = np.array([0.027, 0.048, -0.027, -1.873, -0.006, 1.86, 0.785, 0, 0])
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
        self.render_camera.set_local_pose(look_at([0, -1.0, 0.5], [0, 0, 0]))

        third_view_camera = self._scene.add_camera(
            "third_view_camera", 128, 128, np.pi / 2, 0.001, 10
        )
        third_view_camera.set_local_pose(look_at([0.2, 0, 0.4], [0, 0, 0]))
        self._cameras["third_view_camera"] = third_view_camera

    def _setup_viewer(self):
        super()._setup_viewer()
        self._viewer.set_camera_xyz(1.0, 0.0, 1.2)
        self._viewer.set_camera_rpy(0, -0.5, 3.14)


@register_gym_env(name="TurnFaucetPanda-v0", max_episode_steps=200)
class TurnFaucetPandaEnv(PandaEnv):
    DEFAULT_MODEL_JSON = ASSET_DIR / "partnet_mobility/meta/info_faucet_v0.json"

    _joint_friction = 1.0
    _joint_damping = 1.0
    _faucet_density = 5e4

    def __init__(self, model_ids: List[str] = (), model_json: str = None, **kwargs):
        if model_json is None:
            model_json = self.DEFAULT_MODEL_JSON
        self.model_db: Dict[str, Dict] = load_json(model_json)
        if len(model_ids) == 0:
            model_ids = sorted(self.model_db.keys())
        assert len(model_ids) > 0, model_json
        # model_ids = list(map(str, model_ids))
        self.model_ids = model_ids
        self.model_id = model_ids[0]
        self.model_scale = None
        self.target_link_name = None

        self._should_reconfigure = False

        super().__init__(**kwargs)

    def seed(self, seed=None):
        # Since reward is based on sampled points, reconfigure when seed changes.
        self._should_reconfigure = True
        return super().seed(seed)

    def reset(self, seed=None, reconfigure=False):
        self.set_episode_rng(seed)

        if self._should_reconfigure:
            reconfigure = True
            self._should_reconfigure = False

        # -------------------------------------------------------------------------- #
        # Model selection
        # -------------------------------------------------------------------------- #
        next_model_id = random_choice(self.model_ids, self._episode_rng)
        is_same_model = next_model_id == self.model_id
        if not is_same_model:
            self.model_id = next_model_id
            reconfigure = True

        model_info = self.model_db[self.model_id]

        bbox = model_info["bbox"]
        bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])
        # TODO(jigu): random scale selection
        self.model_scale = 0.3 / max(bbox_size)
        self.offset = -np.array(bbox["min"]) * self.model_scale

        switch_link_names = []
        for semantic in model_info["semantics"]:
            if semantic[2] == "switch":
                switch_link_names.append(semantic[0])
        if len(switch_link_names) == 0:
            raise RuntimeError(self.model_id)

        next_target_link_name = random_choice(switch_link_names, self._episode_rng)
        is_same_link = next_target_link_name == self.target_link_name
        self.target_link_name = next_target_link_name

        # -------------------------------------------------------------------------- #

        if reconfigure:  # Reconfigure the scene if assets change
            self.reconfigure()
        else:
            self.set_sim_state(self._initial_sim_state)

        if not (is_same_model and is_same_link):
            self._set_target_link()
        self._set_init_and_target_angle()
        self.initialize_episode()

        return self.get_obs()

    def _load_articulations(self):
        self.faucet = self._load_faucet()
        # Set friction and damping for all joints here
        for joint in self.faucet.get_active_joints():
            joint.set_friction(self._joint_friction)
            joint.set_drive_property(0.0, self._joint_damping)

    def _load_faucet(self):
        loader = self._scene.create_urdf_loader()
        loader.scale = self.model_scale
        loader.fix_root_link = True

        model_dir = ASSET_DIR / f"partnet_mobility/dataset/{self.model_id}"
        # urdf_path = model_dir / "mobility_vhacd.urdf"
        urdf_path = model_dir / "mobility_cvx.urdf"
        loader.load_multiple_collisions_from_file = True
        if not urdf_path.exists():
            print(f"{urdf_path} does not exist!")
            urdf_path = model_dir / "mobility.urdf"
            loader.load_multiple_collisions_from_file = False

        articulation = loader.load(
            str(urdf_path), config={"density": self._faucet_density}
        )
        articulation.set_name("faucet")
        return articulation

    def _load_agent(self):
        super()._load_agent()

        links = self._agent._robot.get_links()
        self.lfinger = get_entity_by_name(links, "panda_leftfinger")
        self.rfinger = get_entity_by_name(links, "panda_rightfinger")
        self.lfinger_mesh = merge_meshes(get_actor_meshes(self.lfinger))
        self.rfinger_mesh = merge_meshes(get_actor_meshes(self.rfinger))

        # NOTE(jigu): trimesh uses np.random to sample
        with np_random(self._main_seed):
            self.lfinger_pcd = self.lfinger_mesh.sample(256)
            self.rfinger_pcd = self.rfinger_mesh.sample(256)
        # trimesh.PointCloud(self.lfinger_pcd).show()
        # trimesh.PointCloud(self.rfinger_pcd).show()

    def _set_target_link(self):
        self.target_link = get_entity_by_name(
            self.faucet.get_links(), self.target_link_name
        )
        self.target_joint = self.faucet.get_joints()[self.target_link.get_index()]
        # self.target_joint.set_friction(self._joint_friction)
        # self.target_joint.set_drive_property(0.0, self._joint_damping)
        self.target_joint_idx = self.faucet.get_active_joints().index(self.target_joint)
        self.target_link_mesh = merge_meshes(get_actor_meshes(self.target_link))
        # NOTE(jigu): trimesh uses np.random to sample
        with np_random(self._main_seed):
            self.target_link_pcd = self.target_link_mesh.sample(256)

    def _set_init_and_target_angle(self):
        qmin, qmax = self.target_joint.get_limits()[0]
        if np.isinf(qmin):
            self.init_angle = 0
        else:
            self.init_angle = qmin
        if np.isinf(qmax):
            # maybe we can count statics
            self.target_angle = np.pi / 3
        else:
            self.target_angle = qmin + (qmax - qmin) * 0.9

    def _initialize_articulations(self):
        qpos = self.faucet.get_qpos()
        qpos[self.target_joint_idx] = self.init_angle
        self.faucet.set_qpos(qpos)

        p = np.zeros(3)
        p[:2] = self._episode_rng.uniform(-0.05, 0.05, [2])
        p[2] = self.offset[1]  # 1-dim seems to be z offset
        # q = euler2quat(0, 0, np.pi + self._episode_rng.uniform(-np.pi / 12, np.pi / 12))
        q = euler2quat(0, 0, self._episode_rng.uniform(-np.pi / 12, np.pi / 12))
        self.faucet.set_pose(Pose(p, q))

    @property
    def current_angle(self):
        return self.faucet.get_qpos()[self.target_joint_idx]

    def check_success(self):
        return self.current_angle >= self.target_angle

    def _compute_distance(self):
        """Compute the distance between the tap and robot fingers."""
        T = self.target_link.pose.to_transformation_matrix()
        pcd = transform_points(T, self.target_link_pcd)
        T1 = self.lfinger.pose.to_transformation_matrix()
        T2 = self.rfinger.pose.to_transformation_matrix()
        pcd1 = transform_points(T1, self.lfinger_pcd)
        pcd2 = transform_points(T2, self.rfinger_pcd)
        # trimesh.PointCloud(np.vstack([pcd, pcd1, pcd2])).show()
        distance1 = cdist(pcd, pcd1)
        distance2 = cdist(pcd, pcd2)

        return min(distance1.min(), distance2.min())

    def compute_dense_reward(self):
        reward = 0.0

        if self.check_success():
            reward = 5.0
        else:
            distance = self._compute_distance()
            reward += 1 - np.tanh(distance * 10.0)

            is_contacted = any(self.agent.check_contact_fingers(self.target_link))
            if is_contacted:
                reward += 0.25

            angle_diff = self.target_angle - self.current_angle
            reward += 1 - np.tanh(max(angle_diff, 0) * 2.0)

        return reward

    def get_articulations(self):
        return [self._agent._robot, self.faucet]

    def get_state(self) -> np.ndarray:
        state = super().get_state()
        return np.hstack([state, self.target_angle])

    def set_state(self, state):
        self.target_angle = state[-1]
        super().set_state(state[:-1])

    def _get_obs_extra(self) -> OrderedDict:
        extra_dict = OrderedDict()
        extra_dict["model_id"] = np.array([int(self.model_id)])
        T = self.target_link.pose.to_transformation_matrix()
        pcd = transform_points(T, self.target_link_pcd)
        extra_dict["curr_link_pcd"] = pcd

        qpos_back = self.faucet.get_qpos()
        qpos_target = qpos_back.copy()
        qpos_target[self.target_joint_idx] = self.target_angle
        self.faucet.set_qpos(qpos_target)
        T = self.target_link.pose.to_transformation_matrix()
        pcd = transform_points(T, self.target_link_pcd)
        extra_dict["target_link_pcd"] = pcd
        self.faucet.set_qpos(qpos_back)

        extra_dict["curr_angle"] = np.array([self.current_angle])
        extra_dict["target_angle"] = np.array([self.target_angle])
        assert self.target_joint.type == "revolute"
        joint_pose = (
            self.target_joint.get_parent_link().get_pose().to_transformation_matrix()
            @ self.target_joint.get_pose_in_parent().to_transformation_matrix()
        )
        extra_dict["joint_position"] = joint_pose[:3, 3]
        extra_dict["joint_axis"] = joint_pose[:3, 0]

        return extra_dict


def build_faucet(scene: sapien.Scene, density=1000, friction=0.1, damping=0.5):
    builder = scene.create_articulation_builder()
    color = [1.0, 0.8, 0]

    root = builder.create_link_builder()
    root.set_name("root")
    # base
    half_size = np.array([50, 50, 100]) / 1000 / 2
    pose = Pose([0, 0, 100 / 1000 / 2])
    root.add_box_collision(pose, half_size)
    root.add_box_visual(pose, half_size, color=color)
    # vertical spout
    half_size = np.array([32, 32, 180]) / 1000 / 2
    pose = Pose([0, 0, (100 + 180 / 2) / 1000])
    root.add_box_collision(pose, half_size)
    root.add_box_visual(pose, half_size, color=color)
    # horizontal spout
    half_size = np.array([200, 32, 32]) / 1000 / 2
    pose = Pose([(200 / 2 - 32 / 2) / 1000, 0, (100 + 180 + 32 / 2) / 1000])
    root.add_box_collision(pose, half_size)
    root.add_box_visual(pose, half_size, color=color)

    # If the tap is too light, the simulation might fail.
    color = [0.8, 1, 0]
    switch = builder.create_link_builder(root)
    switch.set_name("switch")
    half_size = np.array([50, 30, 50]) / 1000 / 2
    switch.add_box_collision(half_size=half_size, density=density)
    switch.add_box_visual(half_size=half_size, color=color)
    half_size = np.array([10, 10, 50]) / 1000 / 2
    pose = Pose([0, 0, (50 / 2 + 50 / 2) / 1000])
    switch.add_box_collision(pose=pose, half_size=half_size, density=density)
    switch.add_box_visual(pose=pose, half_size=half_size, color=color)
    switch.set_joint_name("tap_joint")
    switch.set_joint_properties(
        "revolute",
        limits=[[0, np.pi / 2]],
        pose_in_parent=Pose(
            p=np.array([0, 30, 50]) / 1000, q=euler2quat(0, 0, np.pi / 2)
        ),
        pose_in_child=Pose(
            p=np.array([0, -30 / 2, 0]) / 1000, q=euler2quat(0, 0, np.pi / 2)
        ),
        friction=friction,
        damping=damping,
    )

    return builder.build(fix_root_link=True)


@register_gym_env(name="TurnCustomFaucetPanda-v0", max_episode_steps=200)
class TurnCustomFaucetPandaEnv(TurnFaucetPandaEnv):
    _joint_friction = 0.1
    _joint_damping = 0.5

    def __init__(self, *args, **kwargs):
        self._should_reconfigure = False
        super(TurnFaucetPandaEnv, self).__init__(*args, **kwargs)

    def reset(self, seed=None, reconfigure=False):
        self.set_episode_rng(seed)

        if self._should_reconfigure:
            reconfigure = True
            self._should_reconfigure = False

        self.target_link_name = "switch"
        self.offset = [0, 0, 0]

        if reconfigure:  # Reconfigure the scene if assets change
            self.reconfigure()
            self._set_target_link()
            self._set_init_and_target_angle()
        else:
            self.set_sim_state(self._initial_sim_state)

        self.initialize_episode()

        return self.get_obs()

    def _load_faucet(self):
        return build_faucet(self._scene)

    def _initialize_articulations(self):
        qpos = self.faucet.get_qpos()
        qpos[self.target_joint_idx] = self.init_angle
        self.faucet.set_qpos(qpos)

        p = np.zeros(3)
        p[:2] = self._episode_rng.uniform(-0.05, 0.05, [2])
        q = euler2quat(0, 0, np.pi + self._episode_rng.uniform(-np.pi / 12, np.pi / 12))
        self.faucet.set_pose(Pose(p, q))

    def _set_init_and_target_angle(self):
        self.init_angle = 0
        self.target_angle = np.pi / 3

    def _get_obs_extra(self):
        return OrderedDict()
