# GraspNet

https://graspnet.net/datasets.html

## Download models

gdown --id 1Gxwu2C5wRQ0QwjdA8CbMXx-bYf_wwPT5 -O mani_skill2/assets/graspnet/models.zip

## Generate scales

```bash
python generate_json.py ${MODEL_DIR} info_v0.json  # scale=1.0
```