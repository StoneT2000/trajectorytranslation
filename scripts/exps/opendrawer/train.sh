# All scripts used to train models for the boxpushing environment

# seeds used are 5670, 5791, 6983
seed=5670

## TR2-GPT2
python scripts/train_translation_online.py exp_cfg.seed=$seed \
    cfg=cfgs/opendrawer/train.yml restart_training=False \
    logging_cfg.wandb=True logging_cfg.workspace=results logging_cfg.exp_name=opendrawer/transformer/$seed logging_cfg.wandb_cfg.group=opendrawer \
    exp_cfg.n_envs=20 exp_cfg.steps_per_epoch=20000 exp_cfg.epochs=2000

## TR2-LSTM
python scripts/train_translation_online.py exp_cfg.seed=$seed \
    cfg=cfgs/opendrawer/train.yml restart_training=False \
    logging_cfg.wandb=True logging_cfg.workspace=results logging_cfg.exp_name=opendrawer/lstm/$seed logging_cfg.wandb_cfg.group=opendrawer \
    model_cfg.lstm_config.num_layers=4 model_cfg.lstm_config.dropout=0.1 model_cfg.type=LSTM \
    exp_cfg.n_envs=20 exp_cfg.steps_per_epoch=20000 exp_cfg.epochs=2000

## SGC
python scripts/train_translation_online.py exp_cfg.seed=$seed \
    cfg=cfgs/opendrawer/train.yml restart_training=False \
    logging_cfg.wandb=True logging_cfg.workspace=results logging_cfg.exp_name=opendrawer/mlp_subgoal/$seed logging_cfg.wandb_cfg.group=opendrawer \
    model_cfg.type=MLPID model_cfg.stack_size=1 model_cfg.mlp_config.dropout=0.1 env_cfg.sub_goals=True \
    exp_cfg.n_envs=20 exp_cfg.steps_per_epoch=20000 exp_cfg.epochs=2000

## GC
python scripts/train_translation_online.py exp_cfg.seed=$seed \
    cfg=cfgs/opendrawer/train.yml restart_training=False \
    logging_cfg.wandb=True logging_cfg.workspace=results logging_cfg.exp_name=opendrawer/mlp/$seed logging_cfg.wandb_cfg.group=opendrawer \
    model_cfg.type=MLPID model_cfg.stack_size=1 model_cfg.mlp_config.dropout=0.1 \
    exp_cfg.n_envs=20 exp_cfg.steps_per_epoch=20000 exp_cfg.epochs=2000