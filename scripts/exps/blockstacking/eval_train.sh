# Table 1 Results
# To run and get a clean output do
# bash scripts/exps/eval_train.sh 2> /dev/null | grep results

## Pick and Place (Train) - Row 7
### TR2-GPT2
seeds=( 224 3103 5280 7429 8820 )
for seed in ${seeds[@]}
do
    python scripts/eval_translation.py \
        cfg=cfgs/blockstacking/eval.yml model=results/blockstacking/transformer/${seed}_finetune/models/best_train_EpRet.pt
done

### TR2-LSTM
for seed in ${seeds[@]}
do
    python scripts/eval_translation.py \
        cfg=cfgs/blockstacking/eval.yml model=results/blockstacking/lstm/${seed}/models/best_train_EpRet.pt
done

# python scripts/eval_translation.py \
#     cfg=cfgs/blockstacking/eval.yml model=results/blockstacking/lstm/224/models/best_train_EpRet.pt
# python scripts/eval_translation.py \
#     cfg=cfgs/blockstacking/eval.yml model=results/blockstacking/lstm/3103/models/best_train_EpRet.pt
# python scripts/eval_translation.py \
#     cfg=cfgs/blockstacking/eval.yml model=results/blockstacking/lstm/5280/models/best_train_EpRet.pt

### SGC
for seed in ${seeds[@]}
do
    python scripts/eval_translation.py \
        cfg=cfgs/blockstacking/eval.yml model=results/blockstacking/mlp_subgoal/${seed}/models/best_train_EpRet.pt env_cfg.sub_goals=True
done
# python scripts/eval_translation.py \
#     cfg=cfgs/blockstacking/eval.yml model=results/blockstacking/mlp_subgoal/224/models/best_train_EpRet.pt env_cfg.sub_goals=True
# python scripts/eval_translation.py \
#     cfg=cfgs/blockstacking/eval.yml model=results/blockstacking/mlp_subgoal/3103/models/best_train_EpRet.pt env_cfg.sub_goals=True
# python scripts/eval_translation.py \
#     cfg=cfgs/blockstacking/eval.yml model=results/blockstacking/mlp_subgoal/5280/models/best_train_EpRet.pt env_cfg.sub_goals=True

### GC
for seed in ${seeds[@]}
do
    python scripts/eval_translation.py \
        cfg=cfgs/blockstacking/eval.yml model=results/blockstacking/mlp/${seed}/models/best_train_EpRet.pt
done
# python scripts/eval_translation.py \
#     cfg=cfgs/blockstacking/eval.yml model=results/blockstacking/mlp/224/models/best_train_EpRet.pt
# python scripts/eval_translation.py \
#     cfg=cfgs/blockstacking/eval.yml model=results/blockstacking/mlp/3103/models/best_train_EpRet.pt
# python scripts/eval_translation.py \
#     cfg=cfgs/blockstacking/eval.yml model=results/blockstacking/mlp/5280/models/best_train_EpRet.pt
