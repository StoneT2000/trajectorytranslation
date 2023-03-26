# Table 1 results

## Couch Moving Short 3 (Train) - Row 3
seeds=( 1256 5330 7263 7299 9577 )

## TR2-GPT2
for seed in ${seeds[@]}
do
    python scripts/eval_translation.py cfg=cfgs/couchmoving/eval.yml \
        model=results/couchmoving/transformer/$seed/models/best_train_EpRet.pt
done
## TR2-LSTM
for seed in ${seeds[@]}
do
    python scripts/eval_translation.py cfg=cfgs/couchmoving/eval.yml \
        model=results/couchmoving/lstm/$seed/models/best_train_EpRet.pt
done
## SGC
for seed in ${seeds[@]}
do
    python scripts/eval_translation.py cfg=cfgs/couchmoving/eval.yml \
        model=results/couchmoving/mlp_subgoal/$seed/models/best_train_EpRet.pt env_cfg.sub_goal_nstep=5 env_cfg.sub_goals=True
done
## GC
for seed in ${seeds[@]}
do
    python scripts/eval_translation.py cfg=cfgs/couchmoving/eval.yml \
        model=results/couchmoving/mlp/$seed/models/best_train_EpRet.pt env_cfg.exclude_target_state=False
done