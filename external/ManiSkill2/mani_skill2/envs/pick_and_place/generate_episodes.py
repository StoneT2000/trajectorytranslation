import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import sapien.core as sapien
import tqdm
import transforms3d
from sapien.core import Pose
from sapien.utils import Viewer

from mani_skill2.envs.pick_and_place.utils import build_actor_orctoc, sample_scale
from mani_skill2.utils.io import dump_json
from mani_skill2.utils.sapien_utils import vectorize_pose

# NOTE(jigu): Use a global engine and renderer here
# to avoid unnecessary recreation which might get stuck
engine = sapien.Engine()
renderer = sapien.VulkanRenderer()
engine.set_renderer(renderer)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str, required=True, help="output path")
    parser.add_argument(
        "--model-json",
        type=str,
        required=True,
        help="path of json including model info",
    )
    parser.add_argument("--start-id", type=int, default=0, help="initial episode id")
    parser.add_argument(
        "-n", "--num-episodes", type=int, required=True, help="number of episodes"
    )
    parser.add_argument(
        "--lock", action="store_true", help="whether to lock x-axis and y-axis rotation"
    )
    parser.add_argument(
        "--onscreen", action="store_true", help="whether to visualize on screen"
    )

    args = parser.parse_args()
    return args


def build_wall(scene: sapien.Scene, xhs, yhs, thickness, height):
    builder = scene.create_actor_builder()
    poses = [
        Pose([-xhs - thickness, 0, 0], [1, 0, 0, 0]),
        Pose([xhs + thickness, 0, 0], [1, 0, 0, 0]),
        Pose([0, -yhs - thickness, 0], [0.7071068, 0, 0, 0.7071068]),
        Pose([0, yhs + thickness, 0], [0.7071068, 0, 0, 0.7071068]),
    ]
    half_sizes = [
        [thickness, yhs, height],
        [thickness, yhs, height],
        [thickness, xhs, height],
        [thickness, xhs, height],
    ]
    for pose, half_size in zip(poses, half_sizes):
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size)
    return builder.build_static()


def generate_episode(model_ids, model_scales, seed=None, lock=False, onscreen=False):
    # -------------------------------------------------------------------------- #
    # Initialization
    # -------------------------------------------------------------------------- #
    np.random.seed(seed)

    scene_config = sapien.SceneConfig()
    scene_config.solver_iterations = 25
    scene_config.solver_velocity_iterations = 2
    scene_config.enable_pcm = False
    scene_config.default_restitution = 0
    scene_config.default_dynamic_friction = 0.5
    scene_config.default_static_friction = 0.5
    scene = engine.create_scene(scene_config)
    scene.set_timestep(1 / 250)

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
            scene.step()
            if onscreen and (i % render_freq == 0):
                scene.update_render()
                viewer.render()

    # -------------------------------------------------------------------------- #
    # Load assets
    # -------------------------------------------------------------------------- #
    scene.add_ground(0)

    actors: List[sapien.Actor] = []
    assert len(model_ids) == len(model_scales)
    for model_id, model_scale in zip(model_ids, model_scales):
        actor = build_actor_orctoc(model_id, scene, scale=model_scale)
        actor.name = model_id
        actors.append(actor)

    # -------------------------------------------------------------------------- #
    # Randomize initial states
    # -------------------------------------------------------------------------- #
    # The workspace is a [-xhs, xhs] X [-yhs, yhs] region on the ground.
    xhs = 0.1
    yhs = 0.1
    clearance = 0.05
    wall_height = 3

    # add walls to prevent objects from moving outside the workspace
    wall = build_wall(scene, xhs + clearance, yhs + clearance, 1, wall_height)

    # add a floor
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[xhs + clearance, yhs + clearance, 1])
    builder.add_box_visual(half_size=[xhs + clearance, yhs + clearance, 1])
    floor = builder.build_static()
    floor.set_pose(Pose([0, 0, wall_height + 1]))

    # place all actors far away
    for i, actor in enumerate(actors):
        actor.set_pose(Pose([100 * (i + 1), 0, 1.0]))

    # sequentially drop objects into the workspace
    for actor in actors:
        x = np.random.uniform(-xhs, xhs)
        y = np.random.uniform(-yhs, yhs)
        z = np.random.uniform(1, 2)

        if lock:
            q = transforms3d.euler.euler2quat(0, 0, np.random.uniform(0, 2 * np.pi))
        else:
            q = np.random.random(4)
            q /= np.linalg.norm(q)

        actor.set_pose(Pose([x, y, z], q))

        actor.set_velocity(np.zeros(3))
        # v = np.random.uniform(-0.5, 0.5, size=3)
        # actor.set_velocity(v)

        if lock:
            actor.lock_motion(0, 0, 0, 1, 1, 0)

        # The object has fallen down to the ground.
        simulate(200)

        if lock:
            actor.lock_motion(0, 0, 0, 0, 0, 0)

    simulate(1000)

    scene.remove_actor(wall)
    scene.remove_actor(floor)

    simulate(1000)

    if onscreen:
        viewer.close()

    episode = dict(actors=[])
    for i, actor in enumerate(actors):
        actor_config = dict(name=actor.name)
        actor_config.update(pose=vectorize_pose(actor.pose).tolist())
        actor_config.update(scale=model_scales[i])
        episode["actors"].append(actor_config)

    return episode


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

    # load model db
    with open(args.model_json, "rt") as f:
        model_db = json.loads(f.read())
    model_ids = model_db.keys()

    episodes = []
    start_id = args.start_id
    num_episodes = args.num_episodes

    for episode_id in tqdm.tqdm(range(start_id, start_id + num_episodes)):
        # sample model scales
        np.random.seed(episode_id)
        model_scales = []
        for model_id in model_ids:
            model_scale = sample_scale(model_db[model_id]["scales"])
            model_scales.append(model_scale)
        assert len(model_scales) == len(model_ids)

        # generate one episode
        episode = generate_episode(
            model_ids,
            model_scales,
            seed=episode_id,
            lock=args.lock,
            onscreen=args.onscreen,
        )

        # generate episode id
        episode.update(episode_id=episode_id)
        episodes.append(episode)

    # output json
    dump_json(output_path, episodes, indent=0)


if __name__ == "__main__":
    main()
