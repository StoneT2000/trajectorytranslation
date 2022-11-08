import json
from pathlib import Path
from typing import List

import numpy as np
import sapien.core as sapien

from mani_skill2 import ASSET_DIR
from mani_skill2.utils.io import load_json

OCRTOC_DIR = ASSET_DIR / "ocrtoc"
EGAD_DIR = ASSET_DIR / "egad"
GRASPNET_DIR = ASSET_DIR / "graspnet"
HAB_YCB_DIR = ASSET_DIR / "hab_ycb_v1.1"


def build_actor_orctoc(
    model_id: str,
    scene: sapien.Scene,
    scale: float = 1.0,
    physical_material: sapien.PhysicalMaterial = None,
    density=1000,
):
    builder = scene.create_actor_builder()

    collision_file = str(
        OCRTOC_DIR / "models" / model_id / "collision_meshes" / "collision.obj"
    )
    builder.add_multiple_collisions_from_file(
        filename=collision_file,
        scale=[scale] * 3,
        material=physical_material,
        density=density,
    )

    visual_file = str(OCRTOC_DIR / "models" / model_id / "visual_meshes" / "visual.dae")
    builder.add_visual_from_file(
        filename=visual_file,
        scale=[scale] * 3,
    )

    actor = builder.build()
    return actor


def build_actor_egad(
    model_id: str,
    scene: sapien.Scene,
    scale: float = 1.0,
    physical_material: sapien.PhysicalMaterial = None,
    density=1000,
):
    builder = scene.create_actor_builder()
    # A heuristic way to infer split
    split = "train" if "_" in model_id else "eval"

    collision_file = str(
        EGAD_DIR / "egad_{split}_set_vhacd" / f"{model_id}.obj"
    ).format(split=split)
    builder.add_multiple_collisions_from_file(
        filename=collision_file,
        scale=[scale] * 3,
        material=physical_material,
        density=density,
    )

    visual_file = str(EGAD_DIR / "egad_{split}_set" / f"{model_id}.obj").format(
        split=split
    )
    builder.add_visual_from_file(
        filename=visual_file,
        scale=[scale] * 3,
    )

    actor = builder.build()
    return actor


def build_actor_graspnet(
    model_id: str,
    scene: sapien.Scene,
    scale: float = 1.0,
    physical_material: sapien.PhysicalMaterial = None,
    density=1000,
):
    builder = scene.create_actor_builder()

    collision_file = str(GRASPNET_DIR / "models_vhacd" / f"{model_id}.obj")
    builder.add_multiple_collisions_from_file(
        filename=collision_file,
        scale=[scale] * 3,
        material=physical_material,
        density=density,
    )

    # visual_file = str(GRASPNET_DIR / "models" / model_id / "nontextured.ply")
    visual_file = str(GRASPNET_DIR / "models" / model_id / "textured.obj")
    builder.add_visual_from_file(
        filename=visual_file,
        scale=[scale] * 3,
    )

    actor = builder.build()
    return actor


def build_actor_hab_ycb(
    model_id: str,
    scene: sapien.Scene,
    scale: float = 1.0,
    physical_material: sapien.PhysicalMaterial = None,
    density=1000,
):
    builder = scene.create_actor_builder()

    config = load_json(HAB_YCB_DIR / f"{model_id}.object_config.json")

    collision_file = str(HAB_YCB_DIR / config["collision_asset"])
    builder.add_multiple_collisions_from_file(
        filename=collision_file,
        pose=sapien.Pose(q=[0.707, 0.707, 0, 0]),
        scale=[scale] * 3,
        material=physical_material,
        density=density,
    )

    visual_file = str(HAB_YCB_DIR / config["render_asset"])
    builder.add_visual_from_file(
        filename=visual_file,
        scale=[scale] * 3,
    )

    actor = builder.build()
    return actor


def sample_scale(scales: List[float], rng=np.random):
    if len(scales) == 1:
        return scales[0]
    else:
        return rng.choice(scales)
