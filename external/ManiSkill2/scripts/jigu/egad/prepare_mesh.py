"""Modified from https://github.com/dougsm/egad/blob/master/scripts/prepare_meshes.py"""

import argparse
import json
import sys
from pathlib import Path

import trimesh

parser = argparse.ArgumentParser(description="Process meshes for specific gripper.")
parser.add_argument(
    "width",
    type=float,
    help="Gripper width maximum. Either meters or millimeters depending on desired output.",
)
parser.add_argument("path", type=str, help="Path to directory containing .obj files.")
parser.add_argument("output", type=str, help="Path to output json.")
parser.add_argument(
    "--stl",
    action="store_true",
    help="Output .stl rather than .obj files (useful for printing)",
)
args = parser.parse_args()

GRIPPER_WIDTH = args.width
GRIPPER_FRAC = 0.8
gripper_target = GRIPPER_WIDTH * GRIPPER_FRAC

ip = Path(args.path)

# op = ip / "processed_meshes"
# op.mkdir(exist_ok=True)

input_meshes = ip.glob("*.obj")
model_scales = {}

for im in input_meshes:
    # oname = op / im.name

    m = trimesh.load(im)
    exts = m.bounding_box_oriented.primitive.extents

    max_dim = max(list(exts))
    scale = GRIPPER_WIDTH / max_dim
    # scale_mesh(m, scale)
    print(im.name, scale)

    exts = m.bounding_box_oriented.primitive.extents

    min_dim = min(list(exts))
    if min_dim > gripper_target:
        scale = gripper_target / min_dim
        # scale_mesh(m, scale)
        print(im.name, scale)

    model_scales[im.stem] = scale

    # if args.stl:
    #     oname = oname.with_suffix(".stl")

    # print(oname)
    # m.export(oname)


def export_scales_to_json(model_scales):
    json_obj = {}
    for k, v in model_scales.items():
        json_obj[k] = {"scales": [v]}
    return json_obj


def generate_json(model_scales, json_path):
    json_obj = export_scales_to_json(model_scales)
    with open(json_path, "w") as f:
        json.dump(json_obj, f, sort_keys=True, indent=4)


generate_json(model_scales, args.output)
