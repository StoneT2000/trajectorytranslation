# Environments

- [Environments](#environments)
  - [PickAndPlace](#pickandplace)
    - [LiftCube](#liftcube)
    - [PickCube](#pickcube)
    - [StackCube](#stackcube)
    - [PickSingle](#picksingle)
    - [PickClutter](#pickclutter)
    - [ServeFood](#servefood)
  - [Assembly](#assembly)
    - [PegInsertionSide](#peginsertionside)
  - [Fixed Single Articulation](#fixed-single-articulation)
    - [Open Cabinet Door](#open-cabinet-door)
    - [Open Cabinet Drawer](#open-cabinet-drawer)
    - [TurnFaucet](#turnfaucet)

## PickAndPlace

### LiftCube

```python
env = gym.make("LiftCubePanda-v0")
```

### PickCube

```python
env = gym.make("PickCubePanda-v0")
```

2022.01.13:
- Update the success metric: require the robot arm to be nearly static
- Increase the success reward: adapted since the success metric is stricter

### StackCube

```python
env = gym.make("StackCubePanda-v0")
```

---

Download data:
```bash
cd mani_skill2/assets
wget https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/ocrtoc_models.zip && unzip ocrtoc_models.zip -d ocrtoc/ && rm -rf ocrtoc_models.zip
```

### PickSingle

```python
env_name = "PickSinglePanda-v0"
env = gym.make(env_name)
# Certain objects
env = gym.make(env_name, model_ids=["cracker_box", "wood_block"])
# Use other set of models
env = gym.make(env_name, model_json="path/to/json")
```

2022.01.13:
- Initialize the object with random z-axis rotation by default
- Update the success metric: require the robot arm to be nearly static
- Increase the success reward: adapted since the success metric is stricter

### PickClutter

```python
env = gym.make("PickClutterPanda-v0", json_path="mani_skill2/assets/ocrtoc/episodes/train_ycb_box_v0_1000.json.gz")
env = gym.make("PickClutterPanda-v0", json_path="mani_skill2/assets/ocrtoc/episodes/test_ycb_box_v0_100.json.gz")
```

The `json_path` points to a (gzip compressed) json file containing episode specification, named as `{[train/val/test]}_{assets}_{version}_{num_episodes}.json.gz`.

### ServeFood

```bash
cd mani_skill2/assets
# wget https://dl.fbaipublicfiles.com/habitat/ycb/hab_ycb_v1.1.zip
# unzip hab_ycb_v1.1.zip -d hab_ycb_v1.1
wget https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/hab_ycb_v1.1_fixed.zip
unzip hab_ycb_v1.1_fixed.zip
```

## Assembly

### PegInsertionSide

```python
env = gym.make("PegInsertionSidePanda-v0")
```

## Fixed Single Articulation
Download PartNet Mobility

```bash
python mani_skill2/scripts/download_partnet_mobility.py
```
### Open Cabinet Door

### Open Cabinet Drawer

### TurnFaucet

Note that it is currently experimental.
Please download data from `https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/partnet_mobility/dataset/{model_id}.zip` to `mani_skill2/assets/partnet_mobility/dataset`.
Or run `python mani_skill2/utils/data_utils.py --name faucet_v0`.

```python
import mani_skill2.envs.experimental
# PartNet assets
env = gym.make("TurnFaucetPanda-v0")
# Certain model
env = gym.make("TurnFaucetPanda-v0", model_ids=["148", "149"])
# Custom asset (primitive-based)
env = gym.make("TurnCustomFaucetPanda-v0")
```
