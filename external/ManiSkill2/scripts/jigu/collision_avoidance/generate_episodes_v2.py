import numpy as np
import trimesh
from shapely import affinity, geometry


class UniformSampler2D:
    def __init__(self, ranges, rng: np.random.RandomState) -> None:
        self._ranges = np.array(ranges)
        self._rng = rng
        self._fixtures = []

    def sample(self, bbox_xyxy, ori, max_trials, append=True):
        for _ in range(max_trials):
            pos = self._rng.uniform(*self._ranges)

            abb = geometry.box(*bbox_xyxy)
            obb = affinity.rotate(abb, ori, use_radians=True)
            obb = affinity.translate(obb, *pos)

            if not self._check_collision(obb):
                if append:
                    self._fixtures.append(obb)
                return pos

    def _check_collision(self, obb: geometry.Polygon):
        for other in self._fixtures:
            if obb.intersects(other):
                return True
        return False

    def plot(self):
        from matplotlib import patches
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        for obb in self._fixtures:
            xy = np.array(obb.exterior.xy).T
            ax.add_patch(patches.Polygon(xy))
        ax.set_xlim(self._ranges[0][0], self._ranges[1][0])
        ax.set_ylim(self._ranges[0][1], self._ranges[1][1])
        ax.set_aspect("equal")
        plt.show()


def test_UniformSampler2D():
    # sampler = UniformSampler2D([[-1.0, -1.0], [1.0, 1.0]], rng=np.random)
    sampler = UniformSampler2D([[-0.5, -0.5], [0.5, 0.5]], rng=np.random)
    for _ in range(20):
        sampler.sample(
            np.random.uniform(0.05, 0.2, size=2), np.random.uniform(0, np.pi * 2), 100
        )
    sampler.plot()


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


def generate_episode(model_db, num_objects, use_scale=False, seed=None, onscreen=False):
    np.random.seed(seed)
    model_ids = sorted(model_db.keys())

    if num_objects is not None:
        model_ids = [
            model_ids[i]
            for i in np.random.choice(len(model_ids), num_objects, replace=True)
        ]

    # sample model scales
    model_scales = []
    for model_id in model_ids:
        if use_scale:
            model_scale = sample_scale(model_db[model_id]["scales"])
        else:
            model_scale = 0.6
        model_scales.append(model_scale)
    assert len(model_scales) == len(model_ids)

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
    # Randomize initial states
    # -------------------------------------------------------------------------- #
    # The workspace is a [-xhs, xhs] X [-yhs, yhs] region on the ground.
    xhs, yhs = 0.1, 0.1
    scene.add_ground(0)

    sampler = UniformSampler2D([[-xhs, -yhs], [xhs, yhs]], rng=np.random)
    actors: List[sapien.Actor] = []
    inds = []

    for i in range(len(model_ids)):
        model_id = model_ids[i]
        model_scale = model_scales[i]
        
        bbox = model_db[model_id]["bbox"]
        bbox_min = np.array(bbox["min"]) * model_scale
        bbox_max = np.array(bbox["max"]) * model_scale

        # additional rotation
        # if model_id in ["cracker_box"]:
        # if "box" in model_id or "cube" in model_id:
        is_box = model_id in [
            "cracker_box",
            "gelatin_box",
            "pudding_box",
            "sugar_box",
            "wood_block",
        ]
        if is_box and np.min(bbox_max[0:2] - bbox_min[0:2]) > 0.08:
            bbox_min2, bbox_max2 = bbox_min, bbox_max
            if np.random.rand() > 0.5:
                rel_q = transforms3d.euler.euler2quat(np.pi / 2, 0, 0)
                # swap
                bbox_min = np.array([bbox_min2[0], -bbox_max2[2], bbox_min2[1]])
                bbox_max = np.array([bbox_max2[0], -bbox_min2[2], bbox_max2[1]])
            else:
                rel_q = transforms3d.euler.euler2quat(0, np.pi / 2, 0)
                bbox_min = np.array([bbox_min2[2], bbox_min2[1], -bbox_max2[0]])
                bbox_max = np.array([bbox_max2[2], bbox_max2[1], -bbox_min2[0]])
            # print(model_id, rel_q, bbox_min, bbox_max)
        else:
            rel_q = transforms3d.quaternions.qeye()

        bbox_xyxy = np.hstack([bbox_min[0:2], bbox_max[0:2]])
        z = -bbox_min[2]

        for _ in range(10):
            ori = np.random.uniform(0, 2 * np.pi)
            xy = sampler.sample(bbox_xyxy, ori, max_trials=10)
            if xy is not None:
                actor = build_actor_orctoc(model_id, scene, scale=model_scale)
                actor.name = model_id
                q = transforms3d.euler.euler2quat(0, 0, ori)
                q = transforms3d.quaternions.qmult(q, rel_q)
                pose = Pose([xy[0], xy[1], z + 0.001], q)
                actor.set_pose(pose)
                actors.append(actor)
                inds.append(i)
                # simulate(10)
                break

    # -------------------------------------------------------------------------- #
    # Get stable states
    # -------------------------------------------------------------------------- #
    simulate(1000)
    # sampler.plot()
    if onscreen:
        viewer.close()

    episode = dict(actors=[])
    assert len(actors) == len(inds)
    for i, ind in enumerate(inds):
        actor = actors[i]
        actor_config = dict(name=actor.name)
        actor_config.update(pose=vectorize_pose(actor.pose))
        actor_config.update(scale=model_scales[ind])
        episode["actors"].append(actor_config)

    return episode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output-path", type=str, required=True, help="output path"
    )
    parser.add_argument(
        "--model-json",
        type=str,
        required=True,
        help="path of json including model info",
    )
    parser.add_argument("--num-objects", type=int, help="number of objects")
    parser.add_argument("--start-id", type=int, default=0, help="initial episode id")
    parser.add_argument(
        "-n", "--num-episodes", type=int, default=1, help="number of episodes"
    )
    parser.add_argument(
        "--onscreen", action="store_true", help="whether to visualize on screen"
    )

    args = parser.parse_args()
    return args


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

    episodes = []
    start_id = args.start_id
    num_episodes = args.num_episodes

    for episode_id in tqdm.tqdm(range(start_id, start_id + num_episodes)):
        episode = generate_episode(
            model_db,
            args.num_objects,
            seed=episode_id,
            onscreen=args.onscreen,
        )

        episode.update(episode_id=episode_id)
        episodes.append(episode)

    # output json
    dump_json(output_path, episodes, indent=0)


if __name__ == "__main__":
    main()
