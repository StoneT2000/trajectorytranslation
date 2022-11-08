# Table 1 results

## Open 1 unseen drawer - Row 14
# seeds used are 5670, 5791, 6983
seed=5670

## TR2-GPT2
python scripts/eval_translation.py cfg=cfgs/opendrawer/eval.yml \
    env_cfg.trajectories="datasets/opendrawer/dataset_test.npy" env_cfg.trajectories_dataset="datasets/opendrawer/dataset_test.pkl" \
    model=results/couchmoving/transformer/$seed/models/best_train_EpRet.pt

## TR2-LSTM
python scripts/eval_translation.py cfg=cfgs/opendrawer/eval.yml \
    env_cfg.trajectories="datasets/opendrawer/dataset_test.npy" env_cfg.trajectories_dataset="datasets/opendrawer/dataset_test.pkl" \
    model=results/couchmoving/lstm/$seed/models/best_train_EpRet.pt

## SGC
python scripts/eval_translation.py cfg=cfgs/opendrawer/eval.yml \
    env_cfg.trajectories="datasets/opendrawer/dataset_test.npy" env_cfg.trajectories_dataset="datasets/opendrawer/dataset_test.pkl" \
    model=results/couchmoving/mlp_subgoal/$seed/models/best_train_EpRet.pt env_cfg.sub_goals=True

## GC
python scripts/eval_translation.py cfg=cfgs/opendrawer/eval.yml \
    env_cfg.trajectories="datasets/opendrawer/dataset_test.npy" env_cfg.trajectories_dataset="datasets/opendrawer/dataset_test.pkl" \
    model=results/couchmoving/mlp/$seed/models/best_train_EpRet.pt