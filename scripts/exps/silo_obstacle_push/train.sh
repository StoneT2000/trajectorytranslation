# Script used to train models for the Obstacle Push environment used in SILO
# Seeds used are 376, 723, 986
seed=376
python scripts/train_translation_online.py exp_cfg.seed=$seed \
    cfg=cfgs/silo_obstacle_push/train.yml restart_training=False \
    logging_cfg.wandb=True logging_cfg.workspace=results logging_cfg.exp_name=silo_obstacle_push/transformer/$seed logging_cfg.wandb_cfg.group=silo_obstacle_push \
    eval_cfg=/stao-fast-vol/skilltranslation/skilltranslation/cfgs/silo_obstacle_push/eval.yml \
    exp_cfg.n_envs=20 exp_cfg.steps_per_epoch=10000 exp_cfg.epochs=1500