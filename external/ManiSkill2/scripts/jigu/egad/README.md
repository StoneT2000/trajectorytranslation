# EGAD

https://github.com/dougsm/egad

## Download models

https://data.researchdatafinder.qut.edu.au/dataset/c5a0ccba-fa28-4cb7-a9f8-4a7f93670344/resource/f01c0b75-aa6d-4af9-b2a9-5edfee823e03/download/egadevalset.zip

## Generate scales

```bash
# The unit of the original model is mm.
# 0.08 is for Panda
python prepare_mesh.py 0.08 ${EGAD_EVAL_SET} info_eval_v0.json
```