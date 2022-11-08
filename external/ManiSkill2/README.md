# ManiSkill 2022

- [ManiSkill 2022](#maniskill-2022)
  - [Installation](#installation)
  - [Code Style](#code-style)
  - [HowTo](#howto)
    - [Interactive play](#interactive-play)
  - [Changelog](#changelog)
    - [2022.02.04](#20220204)
    - [2022.01.22](#20220122)
    - [2022.01.20](#20220120)

## Installation

```bash
conda env create -n mani_skill -f environment.yml
conda activate mani_skill
pip install https://storage1.ucsd.edu/wheels/sapien-dev/sapien-2.0.0.dev20220118-cp38-cp38-manylinux2014_x86_64.whl
pip install -e .
```

## Code Style

```bash
pip install black isort
```

- black: python formatter
- isort: python utility to sort imports


## HowTo

### Interactive play

```bash
python examples/demo_manual_control.py --env LiftCubePanda-v0
# If there are extra arguments
python examples/demo_manual_control.py --env TurnFaucetPanda-v0 model_ids "['148']"
```

## Changelog

### 2022.02.04

**Changed**
- The logic of `seed` is changed
  - `seed` is not called in `SapienEnv.__init__`.
  - `BaseEnv.seed` will set a main random state. The random state for an episode can be set explictly or randomly by the main random state.
- The observation space is categorized by following keys:
  - image: RGB and depth images
  - pointcloud: fused point cloud in the world frame
  - agent: proprioceptive states
  - extra: additional task informations

### 2022.01.22

**Added**
- `BaseEnv` is added to unify interface for all ManiSkill Environments

**Changed**
- Remove basic and pick, and reorganize into pick_and_place, assembly and experimental
- Rename Lift (similar for Pick and Stack) to LiftCube
- Improvements to save typing
  - update default physical material
  - add `_add_ground`
  - use multiple inheritance to support other robots easily

### 2022.01.20

**Changed**
- `BaseEnv` is renamed to `SapienEnv`
- The interface of `PandaEnv` in basic envs is refactored
  - `step` accepts a dictionary to switch control mode.
  - Wrappers are provided in `mani_skill2.utils.wrappers` to use a single control mode and normalize action. See examples in `examples/demo_manual_control.py`.
- SAPIEN version is updated to 2.0.0.dev20220118

**Removed**
- `set_env_mode` is removed. The observation mode is fixed after initialization. The reward mode can be initialized or changed. The control mode is removed.