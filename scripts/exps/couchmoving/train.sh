# All Training Scripts used to train models for the Couchmoving Environment

# seeds used are 5330, 7263, 9577
seed=5330

## TR2-GP2
python scripts/train_translation_online.py exp_cfg.seed=$seed \
    cfg=cfgs/couchmoving/train.yml restart_training=False \
    logging_cfg.wandb=True logging_cfg.workspace=results logging_cfg.exp_name=couchmoving/transformer/$seed logging_cfg.wandb_cfg.group=couchmoving \
    exp_cfg.n_envs=20 exp_cfg.steps_per_epoch=20000 exp_cfg.epochs=1500

## TR2-LSTM
python scripts/train_translation_online.py exp_cfg.seed=$seed \
    cfg=cfgs/couchmoving/train.yml restart_training=False \
    logging_cfg.wandb=True logging_cfg.workspace=results logging_cfg.exp_name=couchmoving/lstm/$seed logging_cfg.wandb_cfg.group=couchmoving \
    model_cfg.lstm_config.num_layers=4 model_cfg.lstm_config.dropout=0.1 model_cfg.type=LSTM model_cfg.state_dims=15 \
    exp_cfg.n_envs=20 exp_cfg.steps_per_epoch=20000 exp_cfg.epochs=1500

## SGC
python scripts/train_translation_online.py exp_cfg.seed=$seed \
    cfg=cfgs/couchmoving/train.yml restart_training=False \
    logging_cfg.wandb=True logging_cfg.workspace=results logging_cfg.exp_name=couchmoving/mlp_subgoal/$seed logging_cfg.wandb_cfg.group=couchmoving \
    model_cfg.type=MLPID model_cfg.stack_size=1 model_cfg.mlp_config.dropout=0.1 model_cfg.state_dims=17 env_cfg.sub_goals=True env_cfg.sub_goal_nstep=5 \
    exp_cfg.n_envs=20 exp_cfg.steps_per_epoch=20000 exp_cfg.epochs=1500