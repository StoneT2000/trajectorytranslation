import json
import os
from pathlib import Path

import pybullet as p
from bs4 import BeautifulSoup

from mani_skill2 import ASSET_DIR


def do_vhacd(input_path, output_path, log_path=None, **kwargs):
    p.connect(p.DIRECT)
    if log_path is None:
        root, _ = os.path.splitext(output_path)
        log_path = root + ".log.txt"
        print(log_path)
    p.vhacd(input_path, output_path, log_path, **kwargs)
    p.disconnect()


def load_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def load_model_ids(json_path):
    infos = load_json(json_path)
    model_ids = [x["id"] for x in infos]
    return model_ids


def load_semantics(txt_path):
    with open(txt_path, "r") as f:
        annos = [x.strip().split() for x in f.readlines()]
    return annos


def main():
    root_dir = ASSET_DIR / "partnet_mobility/dataset"
    model_ids = load_model_ids("meta/faucet.json")
    for model_id in model_ids:
        print("model_id", model_id)
        model_dir = root_dir / str(model_id)

        vhacd_dir = model_dir / "vhacd_objs"
        vhacd_dir.mkdir(exist_ok=True)

        urdf_path = model_dir / "mobility.urdf"
        new_urdf_path = model_dir / "mobility_vhacd.urdf"

        # if new_urdf_path.exists():
        #     continue

        with open(urdf_path, "r") as f:
            urdf = BeautifulSoup(f, features="xml")

        links = list(urdf.find_all("link"))
        for link in links:
            collisions = list(link.find_all("collision"))
            # print(link.attrs["name"], collisions)
            for collision in collisions:
                mesh = collision.find("mesh")
                if mesh is None:
                    continue

                col_filename = mesh.attrs["filename"]
                basename: str = os.path.basename(col_filename)
                vhacd_path = vhacd_dir / basename
                if not vhacd_path.exists():
                    do_vhacd(str(model_dir / col_filename), str(vhacd_path))
                mesh.attrs["filename"] = col_filename.replace(
                    "textured_objs", "vhacd_objs"
                )

        with open(new_urdf_path, "w") as f:
            f.write(urdf.prettify())


if __name__ == "__main__":
    main()
