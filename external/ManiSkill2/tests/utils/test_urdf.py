import numpy as np
import sapien.core as sapien
from sapien.utils import Viewer

import mani_skill2.utils.urdf.urdfParser as urdfParser
from mani_skill2 import ASSET_DIR
from mani_skill2.utils.urdf.urdf import URDF


def main():
    engine = sapien.Engine()  # Create a physical simulation engine
    renderer = sapien.VulkanRenderer()  # Create a Vulkan renderer
    engine.set_renderer(renderer)  # Bind the renderer and the engine

    scene = engine.create_scene()  # Create an instance of simulation world (aka scene)
    scene.set_timestep(1 / 100.0)  # Set the simulation frequency

    # NOTE: How to build actors (rigid bodies) is elaborated in create_actors.py
    scene.add_ground(altitude=0)  # Add a ground

    urdf_file = str(ASSET_DIR / "partnet_mobility/1000/mobility.urdf")
    # using_old_parser = True
    using_old_parser = False

    if using_old_parser:
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        robot = loader.load(urdf_file)
        robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
    else:
        urdfParser.load_urdf_into_scene(
            urdf_file, scene
        )  # Add entities to the environment

    # Add some lights so that you can observe the scene
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = Viewer(renderer)  # Create a viewer (window)
    viewer.set_scene(scene)  # Bind the viewer and the scene

    # The coordinate frame in Sapien is: x(forward), y(left), z(upward)
    # The principle axis of the camera is the x-axis
    viewer.set_camera_xyz(x=-4, y=0, z=2)
    # The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
    # The camera now looks at the origin
    viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 4), y=0)
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

    robots = scene.get_all_articulations()
    for robot in robots:
        active_joints = robot.get_active_joints()
        for active_joint in active_joints:
            active_joint.set_drive_property(stiffness=20, damping=5)
            active_joint.set_drive_target(0.0)
        qpos = robot.get_qpos()
        robot.set_qpos(qpos)

    while not viewer.closed:  # Press key q to quit
        for _ in range(4):  # render every 4 steps
            robots = scene.get_all_articulations()
            for robot in robots:
                qf = robot.compute_passive_force(
                    gravity=True,
                    coriolis_and_centrifugal=True,
                )
                robot.set_qf(qf)
            scene.step()  # Simulate the world
        scene.update_render()  # Update the world to the renderer
        viewer.render()


if __name__ == "__main__":
    main()
