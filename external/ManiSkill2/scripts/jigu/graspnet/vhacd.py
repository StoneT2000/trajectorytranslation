import os
from pathlib import Path

import pybullet as p


def do_vhacd(input_path, output_path, log_path=None, **kwargs):
    p.connect(p.DIRECT)
    if log_path is None:
        root, _ = os.path.splitext(output_path)
        log_path = root + ".log.txt"
        print(log_path)
    p.vhacd(input_path, output_path, log_path, **kwargs)
    p.disconnect()


def do_vhacd_on_graspnet():
    root_dir = Path("mani_skill2/assets/graspnet/models")
    output_dir = Path("mani_skill2/assets/graspnet/models_vhacd")
    output_dir.mkdir(exist_ok=True)
    for model_path in Path(root_dir).glob("*/textured.obj"):
        # print(model_path)
        output_path = Path(output_dir) / (model_path.parent.name + ".obj")
        # print(output_path)
        do_vhacd(str(model_path), str(output_path))


do_vhacd_on_graspnet()
