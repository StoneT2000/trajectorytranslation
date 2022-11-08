"""Repair texture (*.mtl) for Habitat YCB assets.
There is an extra whitespace for .mtl files, which leads to assimp loading failure for SAPIEN.
"""
from pathlib import Path

YCB_DIR = Path("mani_skill2/assets/hab_ycb_v1.1/ycb")

for model_dir in YCB_DIR.iterdir():
    print(model_dir)
    if model_dir.name.startswith("."):
        continue
    mtl_file = model_dir / "google_16k" / "textured.mtl"
    with open(mtl_file, "r") as f:
        lines = [x.strip() for x in f.readlines()]
        lines = list(filter(None, lines))
    with open(mtl_file, "w") as f:
        f.writelines("\n".join(lines))
