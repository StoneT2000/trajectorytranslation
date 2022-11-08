import io
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from mani_skill2.envs.pick_and_place.utils import GRASPNET_DIR


def generate_square_layout(n):
    import math

    n_cols = math.ceil(math.sqrt(n))
    n_rows = math.ceil(n / n_cols)
    return n_rows, n_cols


def main():
    model_dir = GRASPNET_DIR / "models_vhacd"
    collision_shapes = []
    for model_path in sorted(Path(model_dir).glob("*.obj")):
        mesh = trimesh.load_mesh(model_path)
        # mesh.show()
        collision_shapes.append(mesh)

    visual_shapes = []
    model_dir = GRASPNET_DIR / "models"
    for model_path in sorted(Path(model_dir).glob("*/nontextured_simplified.ply")):
        mesh = trimesh.load_mesh(model_path)
        # mesh.show()
        visual_shapes.append(mesh)

    camera_transform = np.eye(4)
    camera_transform[:3, 3] = [0, 0, 1]
    scene = trimesh.Scene(camera_transform=camera_transform)
    nr, nc = generate_square_layout(len(collision_shapes))
    ys = np.arange(nr) * 0.4
    xs = np.arange(nc) * 0.4

    for i, mesh in enumerate(collision_shapes):
        T = np.eye(4)
        r, c = i // nc, i % nc
        T[0, 3] = xs[c]
        T[1, 3] = ys[r]
        scene.add_geometry(mesh, transform=T)
        # T[2, 3] = 1.0
        T[0, 3] += 0.2
        scene.add_geometry(visual_shapes[i], transform=T)
    scene.show()
    # data = scene.save_image(resolution=(1080, 720))
    # Image.open(io.BytesIO(data)).save("graspnet.png")


if __name__ == "__main__":
    main()
