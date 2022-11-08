import argparse
import json
from pathlib import Path


def export_scales_to_json(model_scales):
    json_obj = {}
    for k, v in model_scales.items():
        json_obj[k] = {"scales": list(v)}
    return json_obj


def generate_json(model_scales, json_path):
    json_obj = export_scales_to_json(model_scales)
    with open(json_path, "w") as f:
        json.dump(json_obj, f, sort_keys=True, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str)
    parser.add_argument("output", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    model_scales = {}
    for path in Path(args.model_dir).glob("*/nontextured_simplified.ply"):
        model_id = path.parent.name
        model_scales[model_id] = [1.0]
    generate_json(model_scales, args.output)


if __name__ == "__main__":
    main()
