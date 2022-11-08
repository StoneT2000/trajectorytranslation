# Table 1 results

## Box Pusher (Train) - Row 1
# seeds used are 9252, 5844, 2586
seed=9252

## TR2-GPT2
python scripts/eval_translation.py cfg=cfgs/boxpusher/eval.yml \
    model=results/boxpusher/transformer/$seed/models/best_train_EpRet.pt

## TR2-LSTM
python scripts/eval_translation.py cfg=cfgs/boxpusher/eval.yml \
    model=results/boxpusher/lstm/$seed/models/best_train_EpRet.pt

## SGC
python scripts/eval_translation.py cfg=cfgs/boxpusher/eval.yml \
    model=results/boxpusher/mlp_subgoal/$seed/models/best_train_EpRet.pt env_cfg.exclude_target_state=False env_cfg.sub_goals=True

## GC
python scripts/eval_translation.py cfg=cfgs/boxpusher/eval.yml \
    model=results/boxpusher/mlp/$seed/models/best_train_EpRet.pt env_cfg.exclude_target_state=False