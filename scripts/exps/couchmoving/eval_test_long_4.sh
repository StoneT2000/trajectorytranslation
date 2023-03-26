# Table 1 results

## Couch Moving Long 4 - Row 5
seeds=( 1256 5330 7263 7299 9577 )

## TR2-GPT2
for seed in ${seeds[@]}
do
    python scripts/eval_translation.py cfg=cfgs/couchmoving/eval.yml \
        env_cfg.fixed_max_ep_len=300 env_cfg.trajectories_dataset="datasets/couchmoving/couch_5_corridorrange_12_20/dataset_teacher.pkl" \
        model=results/couchmoving/transformer/$seed/models/best_train_EpRet.pt
done
## TR2-LSTM
for seed in ${seeds[@]}
do
    python scripts/eval_translation.py cfg=cfgs/couchmoving/eval.yml \
        env_cfg.fixed_max_ep_len=300 env_cfg.trajectories_dataset="datasets/couchmoving/couch_5_corridorrange_12_20/dataset_teacher.pkl" \
        model=results/couchmoving/lstm/$seed/models/best_train_EpRet.pt
done
## SGC
for seed in ${seeds[@]}
do
    python scripts/eval_translation.py cfg=cfgs/couchmoving/eval.yml \
        env_cfg.fixed_max_ep_len=300 env_cfg.trajectories_dataset="datasets/couchmoving/couch_5_corridorrange_12_20/dataset_teacher.pkl" \
        model=results/couchmoving/mlp_subgoal/$seed/models/best_train_EpRet.pt env_cfg.sub_goal_nstep=5 env_cfg.sub_goals=True
done
## GC
for seed in ${seeds[@]}
do
    python scripts/eval_translation.py cfg=cfgs/couchmoving/eval.yml \
        env_cfg.fixed_max_ep_len=300 env_cfg.trajectories_dataset="datasets/couchmoving/couch_5_corridorrange_12_20/dataset_teacher.pkl" \
        model=results/couchmoving/mlp/$seed/models/best_train_EpRet.pt env_cfg.exclude_target_state=False
done