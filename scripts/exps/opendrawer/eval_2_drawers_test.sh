# Table 1 results

## Open 2 drawers - Row 15
seeds=( 3773 5670 5791 6983 9009 )

## TR2-GPT2
for seed in ${seeds[@]}
do
    python scripts/eval_translation.py cfg=cfgs/opendrawer/eval_two_drawers.yml \
        model=results/opendrawer/transformer/$seed/models/best_train_EpRet.pt
done
## TR2-LSTM
for seed in ${seeds[@]}
do
    python scripts/eval_translation.py cfg=cfgs/opendrawer/eval_two_drawers.yml \
        model=results/opendrawer/lstm/$seed/models/best_train_EpRet.pt
done
## SGC
for seed in ${seeds[@]}
do
    python scripts/eval_translation.py cfg=cfgs/opendrawer/eval_two_drawers.yml \
        model=results/opendrawer/mlp_subgoal/$seed/models/best_train_EpRet.pt env_cfg.sub_goals=True
done
## GC
for seed in ${seeds[@]}
do
    python scripts/eval_translation.py cfg=cfgs/opendrawer/eval_two_drawers.yml \
        model=results/opendrawer/mlp/$seed/models/best_train_EpRet.pt
done