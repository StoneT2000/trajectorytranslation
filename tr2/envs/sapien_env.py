from collections import OrderedDict
from typing import Optional

import gym
import numpy as np
import numpy.random as np_random
import sapien.core as sapien
import transforms3d
from gym.utils import seeding

engine = None


class SapienEnv(gym.Env):
    """Superclass for Sapien environments."""

    def __init__(self, control_freq, timestep, offscreen_only=False, gravity=None, ccd=None, contact_offset=None):
        global engine
        self.control_freq = control_freq  # alias: frame_skip in mujoco_py
        self.timestep = timestep

        if engine is None:
            engine = sapien.Engine()
            engine.set_log_level("error")
        self._engine = engine
        self._renderer = sapien.VulkanRenderer(offscreen_only=offscreen_only)
        self.offscreen_only = offscreen_only
        self._engine.set_renderer(self._renderer)
        scene_config = sapien.SceneConfig()
        if ccd is not None:
            scene_config.enable_ccd = ccd
        if gravity is not None:
            scene_config.gravity = gravity
        if contact_offset is not None:
            scene_config.contact_offset = contact_offset
        self.scene_config = scene_config

        self._scene = self._engine.create_scene(self.scene_config)
        self.timestep = timestep
        self._scene.set_timestep(timestep)

        # self._build_world()
        self.viewer = None
        self.camera = None
        self.seed_val = self.seed()[0]
        # self.np_random: np_random.Generator = None

    def _set_state(self, state):
        raise NotImplementedError()
    def _get_obs(self):
        raise NotImplementedError()

    def _build_world(self):
        raise NotImplementedError()

    def _setup_viewer(self):
        raise NotImplementedError()

    # ---------------------------------------------------------------------------- #
    # Override gym functions
    # ---------------------------------------------------------------------------- #
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.seed_val = seed
        return [seed]
    def reconfigure(self):
        self._clear()
        self._setup_scene(self.scene_config)
        self._build_world()
        if not self.offscreen_only:
            self._setup_viewer()

    def _setup_scene(self, scene_config: Optional[sapien.SceneConfig] = None):
        """Setup the simulation scene instance.
        The function should be called in reset(), and might be overrided.
        """
        if scene_config is None:
            scene_config = sapien.SceneConfig()
        self._scene = self._engine.create_scene(scene_config)
        self._scene.set_timestep(self.timestep)
    def _clear(self):
        # self.viewer = None
        self._scene = None
        self.camera = None
    def close(self):
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.close()  # release viewer
        self.viewer = None
        self._scene = None
    def __del__(self):
        self.close()
    def setup_camera(self):
        near, far = 0.1, 100
        width, height = 1024, 1024
        camera_mount_actor = self._scene.create_actor_builder().build_kinematic()
        self.camera = self._scene.add_mounted_camera(
            name="camera",
            actor=camera_mount_actor,
            pose=sapien.Pose(),  # relative to the mounted actor
            width=width,
            height=height,
            fovx=0,
            fovy=np.pi / 3,
            near=near,
            far=far,
        )
        q = [0.7073883, 0, 0.7068252, 0]
        camera_mount_actor.set_pose(sapien.Pose(p=[0, 0, 4.0], q=q))

    def _setup_lighting(self):
        from sys import platform

        if platform == "darwin":
            # 1.x code
            rscene = self._scene.get_renderer_scene()
            rscene.set_ambient_light([0.5, 0.5, 0.5])
            rscene.add_directional_light([0, 0, -1], [1, 1, 1], shadow=True)
        else:
            # 2.x code
            self._scene.set_ambient_light([0.5, 0.5, 0.5])
            self._scene.add_directional_light([0, 0, -1], [1, 1, 1], shadow=True)

    def render(self, mode="human"):
        
        if mode == "human":
            if self.viewer is None:
                self._setup_viewer()
            self._scene.update_render()
            self.viewer.render()
        elif mode == "rgb_array":
            if self.camera is None:
                self.setup_camera()
                self._setup_lighting()
            self._scene.update_render()
            self.camera.take_picture()
            rgb = self.camera.get_float_texture("Color")[:, :, :3]
            rgb_img = (rgb * 255).clip(0, 255).astype("uint8")
            return rgb_img
        else:
            raise NotImplementedError("Unsupported render mode {}.".format(mode))

    # ---------------------------------------------------------------------------- #
    # Utilities
    # ---------------------------------------------------------------------------- #
    def get_actor(self, name) -> sapien.ArticulationBase:
        all_actors = self._scene.get_all_actors()
        actor = [x for x in all_actors if x.name == name]
        if len(actor) > 1:
            raise RuntimeError(f"Not a unique name for actor: {name}")
        elif len(actor) == 0:
            raise RuntimeError(f"Actor not found: {name}")
        return actor[0]

    def get_articulation(self, name) -> sapien.ArticulationBase:
        all_articulations = self._scene.get_all_articulations()
        articulation = [x for x in all_articulations if x.name == name]
        if len(articulation) > 1:
            raise RuntimeError(f"Not a unique name for articulation: {name}")
        elif len(articulation) == 0:
            raise RuntimeError(f"Articulation not found: {name}")
        return articulation[0]

    @property
    def dt(self):
        return self.timestep * self.control_freq
