# Table 1 results

## Couch Moving Long 5 - Row 6
# seeds used are 5330, 7263, 9577
seed=5330

## TR2-GPT2
python scripts/eval_translation.py cfg=cfgs/couchmoving/eval.yml \
    env_cfg.fixed_max_ep_len=400 env_cfg.trajectories_dataset="datasets/couchmoving/couch_6_corridorrange_20_24/dataset_teacher.pkl" \
    model=results/couchmoving/transformer/$seed/models/best_train_EpRet.pt

## TR2-LSTM
python scripts/eval_translation.py cfg=cfgs/couchmoving/eval.yml \
    env_cfg.fixed_max_ep_len=400 env_cfg.trajectories_dataset="datasets/couchmoving/couch_6_corridorrange_20_24/dataset_teacher.pkl" \
    model=results/couchmoving/lstm/$seed/models/best_train_EpRet.pt

## SGC
python scripts/eval_translation.py cfg=cfgs/couchmoving/eval.yml \
    env_cfg.fixed_max_ep_len=400 env_cfg.trajectories_dataset="datasets/couchmoving/couch_6_corridorrange_20_24/dataset_teacher.pkl" \
    model=results/couchmoving/mlp_subgoal/$seed/models/best_train_EpRet.pt env_cfg.sub_goal_nstep=5 env_cfg.sub_goals=True

## GC
python scripts/eval_translation.py cfg=cfgs/couchmoving/eval.yml \
    env_cfg.fixed_max_ep_len=400 env_cfg.trajectories_dataset="datasets/couchmoving/couch_6_corridorrange_20_24/dataset_teacher.pkl" \
    model=results/couchmoving/mlp/$seed/models/best_train_EpRet.pt env_cfg.exclude_target_state=False