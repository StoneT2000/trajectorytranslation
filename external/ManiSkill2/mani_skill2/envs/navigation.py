from pathlib import Path

import gym
import numpy as np
import sapien.core as sapien
import yaml
from gym import spaces

from mani_skill2 import AGENT_CONFIG_DIR
from mani_skill2.agents.builder import create_agent_from_config
from mani_skill2.envs.sapien_env import SapienEnv


class Navigation(SapienEnv):
    def __init__(
        self,
        agent_config_path="mobile_platform_controllable_gimbal.yml",
    ):
        super(Navigation, self).__init__(sim_freq=500, control_freq=20)
        self._agent_config_path = AGENT_CONFIG_DIR / agent_config_path

    def _setup_physical_materials(self):
        self.add_physical_material("default", 1.0, 1.0, 0.0)

    def _setup_render_materials(self):
        self.add_render_material(
            "ground", color=np.array([202, 164, 114, 256]) / 256, specular=0.5
        )
        self.add_render_material("default", metallic=0, roughness=0.9, specular=0.0)

    def get_obs(self):
        obs_dict = {}
        agent_state_dict = self._agent.get_proprioception()
        obs_dict.update(agent_state_dict)
        obs_dict["views"] = self._agent.get_images(depth=True)

        return obs_dict

    def _load_static_actors(self):
        self._scene.add_ground(0.0, render_material=self._render_materials["ground"])
        # cube
        builder = self._scene.create_actor_builder()
        half_size = [0.2, 0.2, 1.2]
        builder.add_box_collision(
            half_size=half_size, material=self._physical_materials["default"]
        )
        builder.add_box_visual(
            half_size=half_size,
            color=[1, 0, 0],
            # material=self._render_materials["default"],
        )
        self.cube = builder.build_kinematic("cube")
        self.cube.set_pose(sapien.Pose([2, 2, 1.2]))

        # wall
        wall_size = [3.0, 0.05, 2.4]
        for i in range(4):
            builder = self._scene.create_actor_builder()
            builder.add_box_collision(
                half_size=wall_size, material=self._physical_materials["default"]
            )
            builder.add_box_visual(half_size=wall_size, color=np.random.rand(3))
            wall = builder.build_kinematic(f"wall{i}")
            wall.set_pose(
                sapien.Pose(
                    [3 * np.sin(np.pi / 2 * i), 3 * np.cos(np.pi / 2 * i), 2.4],
                    [np.sin(np.pi / 4 * i), 0, 0, np.cos(np.pi / 4 * i)],
                )
            )

    def _load_agent(self):
        self._agent = create_agent_from_config(
            self._agent_config_path, self._scene, self._control_freq
        )
        self._agent.reset()
        self._action_range = self._agent.action_range

    def reset(self):
        self._clear()
        # ---------------------------------------------------------------------------- #
        # Create new scene
        # ---------------------------------------------------------------------------- #
        self._setup_scene()
        self._setup_lighting()
        self._setup_physical_materials()
        self._setup_render_materials()
        self._load_static_actors()
        self._load_agent()

        # ---------------------------------------------------------------------------- #
        # Initialize start states
        # ---------------------------------------------------------------------------- #
        self._initialize_agent()

        return self.get_obs()

    def get_reward(self):
        return 0.0

    def _clip_and_scale_action(self, action):  # from [-1, 1] to real action range
        action = np.clip(action, -1, 1)
        action = 0.5 * (
            self._action_range.high - self._action_range.low
        ) * action + 0.5 * (self._action_range.high + self._action_range.low)
        return action

    def step(self, action: np.ndarray):
        processed_action = self._clip_and_scale_action(action)
        self._agent.set_action(processed_action)
        for _ in range(self._sim_step_per_control):
            self._agent.simulation_step()
            self._scene.step()
        obs = self.get_obs()
        reward = self.get_reward()
        done = False
        info = {}
        return obs, reward, done, info

    # ---------------------------------------------------------------------------- #
    # Visualization
    # ---------------------------------------------------------------------------- #
    def _setup_lighting(self):
        self._scene.set_ambient_light([0.3, 0.3, 0.3])
        self._scene.add_point_light([-0.3, -0.3, 2.5], [3, 3, 3])

    def _setup_viewer(self):
        self._viewer.set_scene(self._scene)
        self._viewer.set_camera_xyz(1.0, 0.0, 1.2)
        self._viewer.set_camera_rpy(0, -0.5, 3.14)

    def _initialize_agent(self):
        self._agent.reset()


def main():
    import cv2

    np.set_printoptions(suppress=True, precision=3)
    env = Navigation()
    env.reset()
    viewer = env.render()
    while not viewer.closed:
        if viewer.window.key_down("w"):
            env.reset()
        env.render()
        action = np.zeros(4)
        action[:2] = 0.1
        obs, rew, done, info = env.step(action)

        print(
            f'mobile vel: tra: {obs["mobile_tra_vel"]:.3f}, rot: {obs["mobile_rot_vel"]:.3f}, '
            f'cam_pos_p: {obs["cam_pos_p"]}, cam_pos_q: {obs["cam_pos_q"]}'
        )
        depth = obs["views"]["platform_d415"]["depth"]
        print(f"depth: min: {depth.min():.3f}, max: {depth.max():.3f}")
        rgb = obs["views"]["platform_d415"]["rgb"]
        rgb = (rgb * 255).astype(np.uint8)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("view_color", rgb)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
