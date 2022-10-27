# Script used to evaluate models for the Obstacle Push environment used in SILO
# Seeds used are 376, 723, 986
seed=376
python scripts/eval_translation.py cfg=cfgs/silo_obstacle_push/eval.yml \
   model=results/silo_obstacle_push/transformer/$seed/models/best_train_EpRet.pt 