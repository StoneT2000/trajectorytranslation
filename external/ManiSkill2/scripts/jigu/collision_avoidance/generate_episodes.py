import argparse
from pathlib import Path

import numpy as np
import sapien.core as sapien
import tqdm
from sapien.utils import Viewer

from mani_skill2 import ASSET_DIR
from mani_skill2.utils.io import dump_json
from mani_skill2.utils.sapien_utils import (
    get_articulation_contacts,
    get_articulation_max_impulse_norm,
    vectorize_pose,
)

# NOTE(jigu): Use a global engine and renderer here
# to avoid unnecessary recreation which might get stuck
engine = sapien.Engine()
renderer = sapien.VulkanRenderer()
engine.set_renderer(renderer)


def get_qlimits(robot: sapien.Articulation):
    qlimits = robot.get_qlimits()  # [n, 2]
    for i in range(len(qlimits)):
        qmin, qmax = qlimits[i]
        if np.isinf(qmin):
            qmin = 0
        if np.isinf(qmax):
            qmax = 2 * np.pi
        qlimits[i] = qmin, qmax
    return qlimits


def generate_episode(
    workspace, n_obstacles, scale_factor=1.0, seed=None, lock=False, onscreen=False
):
    # -------------------------------------------------------------------------- #
    # Initialization
    # -------------------------------------------------------------------------- #
    np.random.seed(seed)

    scene_config = sapien.SceneConfig()
    scene_config.gravity = np.zeros(3)
    scene_config.solver_iterations = 25
    scene_config.solver_velocity_iterations = 2
    scene_config.enable_pcm = False
    scene_config.default_restitution = 0
    scene_config.default_dynamic_friction = 0.5
    scene_config.default_static_friction = 0.5
    scene = engine.create_scene(scene_config)
    scene.set_timestep(1 / 250)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    urdf_path = str(ASSET_DIR / "descriptions/panda_v1.urdf")
    robot = loader.load(urdf_path)
    for joint in robot.get_active_joints():
        joint.set_drive_property(1000, 100)
    init_qpos = [0, 0, 0, 0, 0, 3.14, 1.57, 0, 0]
    robot.set_qpos(init_qpos)
    robot.set_drive_target(init_qpos)

    if onscreen:
        scene.set_ambient_light([0.3, 0.3, 0.3])
        scene.add_point_light([2, 2, 2], [1, 1, 1])
        scene.add_point_light([2, -2, 2], [1, 1, 1])
        scene.add_point_light([-2, 0, 2], [1, 1, 1])
        scene.add_directional_light([1, -1, -1], [0.3, 0.3, 0.3])
        scene.add_directional_light([0, 0, -1], [1, 1, 1])

        viewer = Viewer(renderer)
        viewer.set_scene(scene)
        viewer.set_camera_xyz(1.0, 0.0, 1.2)
        viewer.set_camera_rpy(0, -0.5, 3.14)

    def simulate(n_steps, render_freq=4):
        for i in range(n_steps):
            robot.set_qf(robot.compute_passive_force(True, True, False))
            scene.step()
            if onscreen and (i % render_freq == 0):
                scene.update_render()
                viewer.render()

    def is_collision_free(qpos):
        robot.set_qpos(qpos)
        robot.set_drive_target(qpos)
        simulate(1)
        impulse = get_articulation_max_impulse_norm(scene.get_contacts(), robot)
        return impulse < 1e-6

    def sample_qpos(max_trials=100):
        for i in range(max_trials):
            qpos = []
            for j in range(len(qlimits)):
                qmin, qmax = qlimits[j]
                qpos_j = np.random.uniform(qmin, qmax)
                qpos.append(qpos_j)
        robot.set_drive_target(qpos)
        simulate(100)
        actual_qpos = robot.get_qpos()
        return actual_qpos

    qlimits = get_qlimits(robot)
    start_qpos = sample_qpos()
    end_qpos = sample_qpos()

    # -------------------------------------------------------------------------- #
    # Generate obstacles
    # -------------------------------------------------------------------------- #
    workspace = np.array(workspace) * scale_factor
    actors = []
    for i in range(n_obstacles):
        for j in range(100):
            builder = scene.create_actor_builder()
            # half_size = np.random.uniform(0.005, 0.05, size=3)
            half_size = np.random.uniform(0.02, 0.1, size=3)
            builder.add_box_collision(half_size=half_size)
            color = np.random.uniform(0, 1, size=3)
            builder.add_box_visual(half_size=half_size, color=color)
            actor = builder.build_static(f"obstacle_{i}")

            p = np.random.uniform(workspace[0], workspace[1])
            q = np.random.random(4)
            q /= np.linalg.norm(q)
            actor.set_pose(sapien.Pose(p, q))

            if is_collision_free(start_qpos) and is_collision_free(end_qpos):
                actors.append(
                    dict(
                        pose=vectorize_pose(actor.pose),
                        half_size=half_size,
                        color=color,
                    )
                )
                break
            else:
                print("Not collision-free", i, j)
                scene.remove_actor(actor)
                simulate(1)

    simulate(100)
    if onscreen:
        viewer.close()

    episode = dict(actors=actors, start_qpos=start_qpos, end_qpos=end_qpos)
    return episode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str, required=True, help="output path")
    parser.add_argument("--start-id", type=int, default=0, help="initial episode id")
    parser.add_argument(
        "-n", "--num-episodes", type=int, default=1, help="number of episodes"
    )
    parser.add_argument(
        "--onscreen", action="store_true", help="whether to visualize on screen"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.set_printoptions(precision=3)

    output_path = Path(args.output_path)
    if output_path.exists():
        answer = input("Overwrite the existing file?[y/n]")
        if answer != "y":
            exit(0)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    episodes = []
    start_id = args.start_id
    num_episodes = args.num_episodes

    # Panda workspace
    workspace = [-0.8769944, -0.88843143, -0.1972883], [0.905851, 0.90889686, 1.2705939]

    for episode_id in tqdm.tqdm(range(start_id, start_id + num_episodes)):
        episode = generate_episode(
            workspace,
            n_obstacles=20,
            scale_factor=0.6,
            seed=episode_id,
            onscreen=args.onscreen,
        )

        episode.update(episode_id=episode_id)
        episodes.append(episode)

    dump_json(output_path, episodes, indent=0, sort_keys=True)


if __name__ == "__main__":
    main()
