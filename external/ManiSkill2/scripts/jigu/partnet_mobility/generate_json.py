import json
from pathlib import Path

from mani_skill2 import ASSET_DIR


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


def parse_info(root_dir, model_ids):
    root_dir = Path(root_dir)
    model_infos = dict()
    for model_id in model_ids:
        model_dir = root_dir / str(model_id)

        # {"min": (x, y, z), "max": (x, y, z)}
        bbox = load_json(model_dir / "bounding_box.json")

        # (name, type, semantic)
        semantics = load_semantics(model_dir / "semantics.txt")

        if not any(x[2] == "switch" for x in semantics):
            print(f"{model_id}: no switch")
            continue

        model_infos[model_id] = dict(id=model_id, bbox=bbox, semantics=semantics)
    return model_infos


def export_to_json():
    root_dir = ASSET_DIR / "partnet_mobility/dataset"
    cur_dir = Path(__file__).parent
    model_ids = load_model_ids(cur_dir / "meta/faucet.json")
    model_infos = parse_info(root_dir, model_ids)
    output_path = ASSET_DIR / "partnet_mobility/meta/info_faucet_v0.json"
    with open(output_path, "w") as f:
        json.dump(model_infos, f, sort_keys=True, indent=4)


export_to_json()
