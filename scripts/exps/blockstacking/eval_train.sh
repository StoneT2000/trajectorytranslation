# Table 1 Results

## Pick and Place (Train) - Row 7
### TR2-GPT2
python scripts/eval_translation.py \
    cfg=cfgs/blockstacking/eval.yml model=results/blockstacking/transformer/224_finetune/models/best_train_EpRet.pt
python scripts/eval_translation.py \
    cfg=cfgs/blockstacking/eval.yml model=results/blockstacking/transformer/3103_finetune/models/best_train_EpRet.pt
python scripts/eval_translation.py \
    cfg=cfgs/blockstacking/eval.yml model=results/blockstacking/transformer/5280_finetune/models/best_train_EpRet.pt

### TR2-LSTM
python scripts/eval_translation.py \
    cfg=cfgs/blockstacking/eval.yml model=results/blockstacking/lstm/224/models/best_train_EpRet.pt
python scripts/eval_translation.py \
    cfg=cfgs/blockstacking/eval.yml model=results/blockstacking/lstm/3103/models/best_train_EpRet.pt
python scripts/eval_translation.py \
    cfg=cfgs/blockstacking/eval.yml model=results/blockstacking/lstm/5280/models/best_train_EpRet.pt

### SGC
python scripts/eval_translation.py \
    cfg=cfgs/blockstacking/eval.yml model=results/blockstacking/mlp_subgoal/224/models/best_train_EpRet.pt env_cfg.sub_goals=True
python scripts/eval_translation.py \
    cfg=cfgs/blockstacking/eval.yml model=results/blockstacking/mlp_subgoal/3103/models/best_train_EpRet.pt env_cfg.sub_goals=True
python scripts/eval_translation.py \
    cfg=cfgs/blockstacking/eval.yml model=results/blockstacking/mlp_subgoal/5280/models/best_train_EpRet.pt env_cfg.sub_goals=True

### GC
python scripts/eval_translation.py \
    cfg=cfgs/blockstacking/eval.yml model=results/blockstacking/mlp/224/models/best_train_EpRet.pt
python scripts/eval_translation.py \
    cfg=cfgs/blockstacking/eval.yml model=results/blockstacking/mlp/3103/models/best_train_EpRet.pt
python scripts/eval_translation.py \
    cfg=cfgs/blockstacking/eval.yml model=results/blockstacking/mlp/5280/models/best_train_EpRet.pt
