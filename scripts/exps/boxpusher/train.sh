# All scripts used to train models for the boxpushing environment

# seeds used are 9252, 5844, 2586
seed=9252

## TR2-GPT2
python scripts/train_translation_online.py exp_cfg.seed=$seed \
    cfg=cfgs/boxpusher/train.yml restart_training=False \
    logging_cfg.wandb=True logging_cfg.workspace=results logging_cfg.exp_name=boxpusher/transformer/$seed logging_cfg.wandb_cfg.group=boxpusher \
    exp_cfg.n_envs=20 exp_cfg.steps_per_epoch=20000 exp_cfg.epochs=2000

## TR2-LSTM
python scripts/train_translation_online.py exp_cfg.seed=$seed \
    cfg=cfgs/boxpusher/train.yml restart_training=False \
    logging_cfg.wandb=True logging_cfg.workspace=results logging_cfg.exp_name=boxpusher/lstm/$seed logging_cfg.wandb_cfg.group=boxpusher \
    model_cfg.lstm_config.num_layers=4 model_cfg.lstm_config.dropout=0.1 model_cfg.type=LSTM \
    exp_cfg.n_envs=20 exp_cfg.steps_per_epoch=20000 exp_cfg.epochs=2000

## SGC
python scripts/train_translation_online.py exp_cfg.seed=$seed \
    cfg=cfgs/boxpusher/train.yml restart_training=False \
    logging_cfg.wandb=True logging_cfg.workspace=results logging_cfg.exp_name=boxpusher/mlp_subgoal/$seed logging_cfg.wandb_cfg.group=boxpusher \
    model_cfg.type=MLPID model_cfg.stack_size=1 model_cfg.mlp_config.dropout=0.1 env_cfg.exclude_target_state=False model_cfg.state_dims=10 env_cfg.sub_goals=True \
    exp_cfg.n_envs=20 exp_cfg.steps_per_epoch=20000 exp_cfg.epochs=2000

## GC
python scripts/train_translation_online.py exp_cfg.seed=$seed \
    cfg=cfgs/boxpusher/train.yml restart_training=False \
    logging_cfg.wandb=True logging_cfg.workspace=results logging_cfg.exp_name=boxpusher/mlp/$seed logging_cfg.wandb_cfg.group=boxpusher \
    model_cfg.type=MLPID model_cfg.stack_size=1 model_cfg.mlp_config.dropout=0.1 env_cfg.exclude_target_state=False model_cfg.state_dims=6 \
    exp_cfg.n_envs=20 exp_cfg.steps_per_epoch=20000 exp_cfg.epochs=2000