# Abstract-to-Executable Trajectory Translation for One Shot Task Generalization

This is the official codebase for the paper

[Abstract-to-Executable Trajectory Translation for One Shot Task Generalization](https://arxiv.org/abs/2210.07658) by

[Stone Tao](https://stoneztao.com/), [Xiaochen Li](https://sites.google.com/view/xiaochen-li), [Tongzhou Mu](https://cseweb.ucsd.edu//~t3mu/), [Zhiao Huang](https://sites.google.com/view/zhiao-huang), [Yuzhe Qin](https://yzqin.github.io/), [Hao Su](https://cseweb.ucsd.edu/~haosu/)

For visualizations and videos see our project page: https://trajectorytranslation.github.io/

## Installation

To get started, install the repo with conda as so

```
conda env create -f environment.yml
conda activate tr2
```

And then run
```
pip install -e ./paper_rl/
pip install -e . 
pip install -e external/ManiSkill2 
```

Due to some compatability issues, we are still cleaning up the code and setup details to benchmark on opendrawer (which uses ManiSkill 1). Check back for updates or watch this repo.

## Getting Started

Our approach relies on following abstract trajectories. Abstract trajectories are easily generated via heuristics that just move 3D points in space, describing a general plan of what should be achieved by a low-level agent (e.g. the robot arm) without incorporating low-level details like physical manipulation. During RL training, these abstract trajectories are loaded up and given as part of the environment observation. 

Follow the subsequent sections for instructions on obtaining abstract trajectories, training with them, and evaluating with them.

### Abstract Trajectory Generation / Dataset download links

The dataset files can all be found at this google drive link: https://drive.google.com/file/d/1z38DTgzmTc2mfePYnP9qNDUfGgN80FYH/view?usp=sharing

Download and unzip to a folder called `datasets` for the rest of the code to work.

To generate the abstract trajectories for each environment, see the scripts in [scripts/abstract_trajectories/<env_name>](https://github.com/StoneT2000/trajectorytranslation/tree/main/scripts/abstract_trajectories)

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
    cfg=watch_cfg.yml model=results/test_exp/models/best_train_EpRet.pt traj_id=2
```

For each environment, there is an associated config file for evaluation and watching. These are stored at `cfgs/<env_name>/<eval|watch>.yml`

### Reproducing Results

For specific scripts to run experiments to reproduce table 1 in our paper, see `scripts/exps/<env_name>/*.sh`. These contain copy+pastable bash scripts to reproduce the individual results of each trial used to produce the mean values shown in table 1. 


Already trained models and weights can be downloaded here: https://drive.google.com/file/d/1m3GwDAsPypxXQdGppVNJxsr19qWfdKLS/view?usp=sharing
They are organized by `results/<env_name>/<model>`

We are still busy cleaning and organizing results for other non-core environments that were tested on.

#### Reproducing Real World Experiments

Open sourced code for real world experiments is a work in progress, but here is a high level overview: We first predict the pose of a block in the real world, placed it in simulation and ran our trained blockstacking TR2-GPT2 agent to generate a simulated trajectory. Using position control, we execute the simulated trajectory step by step. Then we place a new block into view and repeat the steps until done.

<!-- To setup real world experiments, you need a depth camera (our code is configured for intel-real sense), and some calibration of the camera so that you get a transformation matrix from camera frame to robot base frame. -->

### Creating Your Own Environments

This part is still WIP. However in general, you can subclass of the TrajectoryEnv class which lets you load abstract trajectories, stack observations, skip sampling, and more. See existing environments, (BoxPusher is a simple generally cleaner example) of how to do this.


## Citation

To cite our work, you can use the following bibtex

```
@article{tao2022tr2,
  title     = {Abstract-to-Executable Trajectory Translation for One-Shot Task Generalization}, 
  author    = {Tao, Stone and Li, Xiaochen and Mu, Tongzhou and Huang, Zhiao and Qin, Yuzhe and Su, Hao},
  journal   = {arXiv},
  year      = {2022},
}
```
