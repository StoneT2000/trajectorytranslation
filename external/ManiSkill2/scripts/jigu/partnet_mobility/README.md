# PartNet-Mobility

<https://sapien.ucsd.edu/downloads>

## HowTo

**Find ids for a certain category**

Save the results from <https://sapien.ucsd.edu/api/models?category=Faucet&limit=10000&offset=0>,
or collect information from `meta.json` in each model directory.

**Upload processed files to storage1**

```bash
rsync -avh mani_skill2/assets/partnet_mobility/dataset/*.zip storage1:/data/datasets/ManiSkill2022-assets/partnet_mobility/dataset/
```