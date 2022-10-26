# Table 1 results

## Couch Moving Short 3 (Train) - Row 3
# seeds used are 5330, 7263, 9577
seed=9252

## TR2-GPT2
python scripts/eval_translation.py cfg=cfgs/couchmoving/eval.yml \
    model=results/couchmoving/transformer/$seed/models/best_train_EpRet.pt

## TR2-LSTM
python scripts/eval_translation.py cfg=cfgs/couchmoving/eval.yml \
    model=results/couchmoving/lstm/$seed/models/best_train_EpRet.pt

## SGC
python scripts/eval_translation.py cfg=cfgs/couchmoving/eval.yml \
    model=results/couchmoving/mlp_subgoal/$seed/models/best_train_EpRet.pt env_cfg.sub_goal_nstep=5 env_cfg.sub_goals=True

## GC
python scripts/eval_translation.py cfg=cfgs/couchmoving/eval.yml \
    model=results/couchmoving/mlp/$seed/models/best_train_EpRet.pt env_cfg.exclude_target_state=False