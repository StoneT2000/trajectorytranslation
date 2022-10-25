# All scripts used to train models for the blockstacking environment
# logging to Weights and Biases is automatically turned on here, to turn it off set logging_cfg.wandb=False.

### TR2-GPT2

# Stage 1 Training
# seeds used are 224, 3103, 5280
seed=224
python scripts/train_translation_online.py exp_cfg.seed=$seed \
    cfg=cfgs/blockstacking/train.yml restart_training=False \
    logging_cfg.wandb=True logging_cfg.workspace=results logging_cfg.exp_name=blockstacking/transformer/$seed logging_cfg.wandb_cfg.group=blockstacking \
    exp_cfg.n_envs=20 exp_cfg.accumulate_grads=False exp_cfg.epochs=2000

# Stage 2 Training (finetuning)
# note that this is only applied for tr2-gpt2 since during the first stage tr2-gpt2 is alreaady able to achieve good success rates unlike other bselines
python scripts/train_translation_online.py exp_cfg.seed=$seed \
    cfg=cfgs/blockstacking/train.yml restart_training=False \
    logging_cfg.wandb=True logging_cfg.workspace=results logging_cfg.exp_name=blockstacking/transformer/"$seed"_finetune logging_cfg.wandb_cfg.group=blockstacking_finetune \
    exp_cfg.n_envs=20 exp_cfg.accumulate_grads=True exp_cfg.epochs=1600 \
    pretrained_ac_weights=results/blockstacking/transformer/"$seed"/models/best_train_EpRet.pt

### TR2-LSTM

seed=224
python scripts/train_translation_online.py exp_cfg.seed=$seed \
    cfg=cfgs/blockstacking/train.yml restart_training=False \
    logging_cfg.wandb=True logging_cfg.workspace=results logging_cfg.exp_name=blockstacking/lstm/$seed logging_cfg.wandb_cfg.group=blockstacking \
    model_cfg.lstm_config.num_layers=4 model_cfg.lstm_config.dropout=0.1 model_cfg.type=LSTM \
    exp_cfg.n_envs=20 exp_cfg.accumulate_grads=False exp_cfg.epochs=2000


### SGC

seed=224
python scripts/train_translation_online.py exp_cfg.seed=$seed \
    cfg=cfgs/blockstacking/train.yml restart_training=False \
    logging_cfg.wandb=True logging_cfg.workspace=results logging_cfg.exp_name=blockstacking/mlp_subgoal/$seed logging_cfg.wandb_cfg.group=blockstacking \
    model_cfg.type=MLPID model_cfg.stack_size=1 model_cfg.mlp_config.dropout=0.1 env_cfg.sub_goals=True model_cfg.state_dims=42 model_cfg.act_dims=4 \
    exp_cfg.n_envs=20 exp_cfg.accumulate_grads=False exp_cfg.epochs=2000

### GC

seed=224
python scripts/train_translation_online.py exp_cfg.seed=$seed \
    cfg=cfgs/blockstacking/train.yml restart_training=False \
    logging_cfg.wandb=True logging_cfg.workspace=results logging_cfg.exp_name=blockstacking/mlp/$seed logging_cfg.wandb_cfg.group=blockstacking \
    model_cfg.type=MLPID model_cfg.stack_size=1 model_cfg.mlp_config.dropout=0.1 \
    exp_cfg.n_envs=20 exp_cfg.accumulate_grads=False exp_cfg.epochs=2000