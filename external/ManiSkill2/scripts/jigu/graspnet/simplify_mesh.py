from pathlib import Path

import open3d as o3d

from mani_skill2.envs.pick_and_place.utils import GRASPNET_DIR


def main():
    model_dir = GRASPNET_DIR / "models"
    # for model_path in sorted(Path(model_dir).glob("046/textured.obj"))[:1]:
    for model_path in sorted(Path(model_dir).glob("046/nontextured_simplified.ply"))[
        :1
    ]:
        mesh = o3d.io.read_triangle_mesh(str(model_path), True)
        print(model_path)
        print("Before simplified", mesh)
        o3d.visualization.draw_geometries([mesh])
        mesh_smp = mesh.simplify_quadric_decimation(target_number_of_triangles=10000)
        print("After simplified", mesh_smp)
        o3d.visualization.draw_geometries([mesh_smp])


if __name__ == "__main__":
    main()
