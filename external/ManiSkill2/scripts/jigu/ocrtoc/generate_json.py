import csv
import json
from collections import OrderedDict, defaultdict
from pathlib import Path


def parse_objects_csv(filename):
    """Parse the CSV file to acquire the information of objects used in TOC.

    Args:
        filename (str): a CSV file containing object information.

    Returns:
        Dict[str, dict]: {object_name: object_info}
    """
    object_db = OrderedDict()
    with open(filename, "r") as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            row["bbox"] = {
                "min": [float(row[f"min_{c}"]) for c in ["x", "y", "z"]],
                "max": [float(row[f"max_{c}"]) for c in ["x", "y", "z"]],
            }
            object_db[row["object"]] = row
    return object_db


MODEL_DB = parse_objects_csv("meta/objects.csv")
MODEL_IDS = tuple(MODEL_DB.keys())
NUM_MODELS = len(MODEL_IDS)
# print(MODEL_DB)


def parse_scenes_csv(filename):
    scene_db = []
    with open(filename, "r") as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            row["object_names"] = eval(row["object_names"])
            row["scales"] = eval(row["scales"])
            scene_db.append(row)
    return scene_db


SCENE_DB = parse_scenes_csv("meta/scenes.csv")


def find_scales(
    scene_db, allow_levels=None, allowed_sources=None, allowed_classes=None
):
    model_scales = defaultdict(set)

    for row in scene_db:
        if allow_levels is not None:
            level_id, variant_id = row["scene_id"].split("-")
            if int(level_id) not in allow_levels:
                continue

        for name, scale in zip(row["object_names"], row["scales"]):
            model_scales[name].add(scale)

    if allowed_sources is not None:
        model_scales = {
            k: v
            for k, v in model_scales.items()
            if MODEL_DB[k]["source"] in allowed_sources
        }

    if allowed_classes is not None:
        model_scales = {
            k: v
            for k, v in model_scales.items()
            if MODEL_DB[k]["class"] in allowed_classes
        }

    return dict(model_scales)


def export_scales_to_json(model_scales):
    json_obj = {}
    for k, v in model_scales.items():
        json_obj[k] = {"scales": list(v)}
    return json_obj


def generate_json(model_scales, json_path):
    json_obj = export_scales_to_json(model_scales)

    # add bounding box info as well
    for model_id in json_obj:
        json_obj[model_id]["bbox"] = MODEL_DB[model_id]["bbox"]

    with open(json_path, "w") as f:
        json.dump(json_obj, f, sort_keys=True, indent=4)


# generate_json(
#     find_scales(
#         SCENE_DB, allow_levels=[1], allowed_sources=["YCB"], allowed_classes=["box"]
#     ),
#     "info_ycb_box_v0.json",
# )
# generate_json(
#     find_scales(
#         SCENE_DB,
#         allowed_sources=["YCB"],
#         allowed_classes=["can"],
#     ),
#     "info_ycb_can_v0.json",
# )
# generate_json(
#     find_scales(
#         SCENE_DB,
#         allow_levels=[1, 2, 3],
#         allowed_sources=["YCB"],
#         allowed_classes=["cup"],
#     ),
#     "info_ycb_cup_v0.json",
# )
# generate_json(
#     find_scales(
#         SCENE_DB,
#         allowed_sources=["YCB"],
#         allowed_classes=["bottle", "bowl", "plate"],
#     ),
#     "info_ycb_others_v0.json",
# )
