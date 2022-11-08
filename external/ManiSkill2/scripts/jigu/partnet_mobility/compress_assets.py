import json
import os
import shutil
from pathlib import Path

import tqdm

from mani_skill2 import ASSET_DIR

THIS_DIR = Path(__file__).parent


def load_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def load_model_ids(json_path):
    infos = load_json(json_path)
    model_ids = [x["id"] for x in infos]
    return model_ids


def main():
    root_dir = ASSET_DIR / "partnet_mobility/dataset"
    model_ids = load_model_ids(THIS_DIR / "meta/faucet.json")
    for model_id in tqdm.tqdm(model_ids):
        print("model_id", model_id)
        model_dir = root_dir / str(model_id)
        print(model_dir)
        # https://docs.python.org/3/library/shutil.html#shutil-archiving-example-with-basedir
        shutil.make_archive(
            str(model_dir), "zip", root_dir=root_dir, base_dir=str(model_id)
        )


if __name__ == "__main__":
    main()
