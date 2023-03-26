# Table 1 results

## Box Pusher (Train) - Row 1
seeds=( 1332  2586  5844  7968  9252 )

## TR2-GPT2
# for seed in ${seeds[@]}
# do
#     python scripts/eval_translation.py cfg=cfgs/boxpusher/eval.yml \
#         model=results/boxpusher/transformer/$seed/models/best_train_EpRet.pt
# done

# ## TR2-LSTM
# for seed in ${seeds[@]}
# do
#     python scripts/eval_translation.py cfg=cfgs/boxpusher/eval.yml \
#         model=results/boxpusher/lstm/$seed/models/best_train_EpRet.pt
# done

## SGC
# for seed in ${seeds[@]}
# do
#     python scripts/eval_translation.py cfg=cfgs/boxpusher/eval.yml \
#         model=results/boxpusher/mlp_subgoal/$seed/models/best_train_EpRet.pt env_cfg.exclude_target_state=False env_cfg.sub_goals=True
# done

## GC
for seed in ${seeds[@]}
do
    python scripts/eval_translation.py cfg=cfgs/boxpusher/eval.yml \
        model=results/boxpusher/mlp/$seed/models/best_train_EpRet.pt env_cfg.exclude_target_state=False
done
