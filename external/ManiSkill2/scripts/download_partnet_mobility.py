import io
import os
import zipfile

import requests
from tqdm import tqdm

from mani_skill2 import ASSET_DIR, ROOT_DIR


def main():
    data_dir = ASSET_DIR / "partnet_mobility_dataset"
    data_dir.mkdir(exist_ok=True)

    config_dir = ASSET_DIR / "partnet_mobility_configs"
    config_dir.mkdir(exist_ok=True)

    model_ids = []
    with open(ROOT_DIR / "scripts/partnet_mobility_ids.txt", "r") as f:
        for l in f.readlines():
            model_ids.append(l.strip())
    print(f"{len(model_ids)} models. Downloading...")
    for model_id in tqdm(model_ids):
        url = f"https://storage1.ucsd.edu/datasets/PartNetMobilityScrambled/{model_id}.zip"
        # download file
        r = requests.get(url, stream=True)
        if not r.ok:
            raise Exception(
                f"Download {model_id} failed. Please check your token and IP address."
            )
        z = zipfile.ZipFile(io.BytesIO(r.content))

        # model_dir = data_dir / model_id
        # model_dir.mkdir(exist_ok=True)
        z.extractall(str(data_dir))

    # TODO (ruic): Add config download


if __name__ == "__main__":
    main()
