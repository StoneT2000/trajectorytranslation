# Abstract-to-Executable Trajectory Translation for One Shot Task Generalization

This is the official codebase for the paper

[Abstract-to-Executable Trajectory Translation for One Shot Task Generalization](https://arxiv.org/abs/2210.07658) by


[Stone Tao](https://stoneztao.com/), [Xiaochen Li](https://sites.google.com/view/xiaochen-li), [Tongzhou Mu](https://cseweb.ucsd.edu//~t3mu/), [Zhiao Huang](https://sites.google.com/view/zhiao-huang), [Yuzhe Qin](https://yzqin.github.io/), [Hao Su](https://cseweb.ucsd.edu/~haosu/)

For visualizations and videos see our project page: https://trajectorytranslation.github.io/
[TODO, add graphic?]

## Installation

To get started, install the repo with conda as so

```
conda create -f environment.yml
conda activate tr2
```

And then run
```
pip install -e ./paper_rl/
pip install -e . 
pip install -e external/ManiSkill2 
pip install -e external/ManiSkill # used only for the open drawer environment
```

## Usage

Our approach relies on following abstract trajectories. Abstract trajectories are easily generated via heuristics that just move 3D points in space, describing a general plan of what should be achieved by a low-level agent (e.g. the robot arm). During RL training, these abstract trajectories are loaded up and given as part of the environment observation. 

Follow the subsequent sections for instructions on obtaining abstract trajectories, training with them, and evaluating with them.

### Abstract Trajectory Generation / Dataset download links

The dataset files can all be found at this google drive link:

Download and unzip to a folder called `datasets` for the rest of the code to work.

To generate the abstract trajectories for each environment, follow TODO

### Training

To train with online RL, specify a base configuration yml file, specify the experiment name

```
python scripts/train_translation_online.py \
    cfg=train_cfg.yml restart_training=True logging_cfg.exp_name=test_exp \
    env_cfg.trajectories="datasets/blockstacking/dataset_train_ids.npy" env_cfg.trajectories_dataset="datasets/blockstacking/dataset.pkl"
```

Results including saved model checkpoints and evalution vidoes are stored in a `results` folder. Note that `results/<exp_name>/models/best_train_EpRet.pt` will be the model with the best training return.

In order to achieve greater precision and success rate, you can run the "finetuning" step by turning on gradient accumulation to stabilize RL training. This was used in the paper for training agents for the Blockstacking task. This can be done by running the following and specifying the initial weights (from the initial online training)

```
python scripts/train_translation_online.py \
    cfg=train_cfg.yml restart_training=True logging_cfg.exp_name=test_exp_finetune \
    env_cfg.trajectories="datasets/blockstacking/dataset_train_ids.npy" env_cfg.trajectories_dataset="datasets/blockstacking/dataset.pkl" \
    pretrained_ac_weights=results/test_exp/models/best_train_EpRet.pt exp_cfg.accumulate_grads=True
```

For each environment, there is an associated `train_cfg.yml` file that specifies the base hyperparameters for online RL training and environment configs. These are stored at `cfgs/<env_name>/train.yml`

### Evaluation

To batch evalute trained models, specify the configurataion file and the model weights.

```
python scripts/eval_translation.py \
    cfg=eval_cfg.yml model=results/test_exp/models/best_train_EpRet.pt
```

To simply watch the trained model, specify the configuration file, the model weights, and the ID of the trajectory

```
python scripts/watch_translation.py \
    cfg=watch_cfg.yml model=results/test_exp/models/best_train_EpRet.pt
```

For each environment, there is an associated config file for evalultaion and watching. These are stored at `cfgs/<env_name>/<eval_cfg|watch_cfg>.yml`

### Reproducing Results

For specific configurations for experiments to reproduce table 1 in our paper, see `experiments/cfgs/<exp_name>` [TODO]

## Code Organization

The general codebase is organized as follows

skilltranslation/envs - all envs

skilltranslation/models - all models / policies

skilltranslation/scripts - various scripts to use and run


## Citation

To cite our work, you can use the following bibtex [TODO]
