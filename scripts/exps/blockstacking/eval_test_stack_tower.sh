# Table 1 Results

## Stack k Blocks - Row 8-10

### TR2-GPT2
declare -a models=("results/blockstacking/transformer/224_finetune/models/best_train_EpRet.pt" "results/blockstacking/transformer/3103_finetune/models/best_train_EpRet.pt" "results/blockstacking/transformer/5280_finetune/models/best_train_EpRet.pt")
for model in "${models[@]}"
do
    base="test_results/blockstacking/transformer"
    mkdir -p "test_results/blockstacking/transformer"
    for i in $(seq 4 6)
    do
        echo "Testing on tower " $i
        max_ep_len=$((i*60))
        echo ${max_ep_len}
        python scripts/eval_translation.py cfg=cfgs/blockstacking/eval.yml \
            env_cfg.trajectories_dataset=datasets/blockstacking/dataset_tower-${i}.pkl \
            env_cfg.trajectories=datasets/blockstacking/dataset_tower-${i}_train_ids.npy n_envs=16 test_n=128 env_cfg.fixed_max_ep_len=${max_ep_len} \
            env_cfg.partial_trajectories=True \
            env_cfg.goal=tower-${i} \
            model=${model} \
            save_results_path=${base}.tower-${i}.csv \
            # env_cfg.show_goal_visuals=False env_cfg.obs_mode=state_visual \ # uncomment to bench models using visual pipeline that is used in real world experiments
            device=cuda
    done
done

### TR2-LSTM
declare -a models=("results/blockstacking/lstm/224/models/best_train_EpRet.pt" "results/blockstacking/lstm/3103/models/best_train_EpRet.pt" "results/blockstacking/lstm/5280/models/best_train_EpRet.pt")
for model in "${models[@]}"
do
    base="test_results/blockstacking/lstm"
    mkdir -p "test_results/blockstacking/lstm"
    for i in $(seq 4 6)
    do
        echo "Testing on tower " $i
        max_ep_len=$((i*60))
        echo ${max_ep_len}
        python scripts/eval_translation.py cfg=cfgs/blockstacking/eval.yml \
            env_cfg.trajectories_dataset=datasets/blockstacking/dataset_tower-${i}.pkl \
            env_cfg.trajectories=datasets/blockstacking/dataset_tower-${i}_train_ids.npy n_envs=16 test_n=128 env_cfg.fixed_max_ep_len=${max_ep_len} \
            env_cfg.partial_trajectories=True \
            env_cfg.goal=tower-${i} \
            model=${model} \
            save_results_path=${base}.tower-${i}.csv \
            # env_cfg.show_goal_visuals=False env_cfg.obs_mode=state_visual \ # uncomment to bench models using visual pipeline that is used in real world experiments
            device=cuda
    done
done

### SGC
declare -a models=("results/blockstacking/mlp_subgoal/224/models/best_train_EpRet.pt" "results/blockstacking/mlp_subgoal/3103/models/best_train_EpRet.pt" "results/blockstacking/mlp_subgoal/5280/models/best_train_EpRet.pt")
for model in "${models[@]}"
do
    base="test_results/blockstacking/mlp_subgoal"
    mkdir -p "test_results/blockstacking/mlp_subgoal"
    for i in $(seq 4 6)
    do
        echo "Testing on tower " $i
        max_ep_len=$((i*60))
        echo ${max_ep_len}
        python scripts/eval_translation.py cfg=cfgs/blockstacking/eval.yml \
            env_cfg.trajectories_dataset=datasets/blockstacking/dataset_tower-${i}.pkl \
            env_cfg.trajectories=datasets/blockstacking/dataset_tower-${i}_train_ids.npy n_envs=16 test_n=128 env_cfg.fixed_max_ep_len=${max_ep_len} \
            env_cfg.partial_trajectories=True \
            env_cfg.goal=tower-${i} \
            model=${model} \
            save_results_path=${base}.tower-${i}.csv \
            # env_cfg.show_goal_visuals=False env_cfg.obs_mode=state_visual \ # uncomment to bench models using visual pipeline that is used in real world experiments
            device=cuda env_cfg.sub_goals=True
    done
done

### GC
declare -a models=("results/blockstacking/mlp/224/models/best_train_EpRet.pt" "results/blockstacking/mlp/3103/models/best_train_EpRet.pt" "results/blockstacking/mlp/5280/models/best_train_EpRet.pt")
for model in "${models[@]}"
do
    base="test_results/blockstacking/mlp"
    mkdir -p "test_results/blockstacking/mlp"
    for i in $(seq 4 6)
    do
        echo "Testing on tower " $i
        max_ep_len=$((i*60))
        echo ${max_ep_len}
        python scripts/eval_translation.py cfg=cfgs/blockstacking/eval.yml \
            env_cfg.trajectories_dataset=datasets/blockstacking/dataset_tower-${i}.pkl \
            env_cfg.trajectories=datasets/blockstacking/dataset_tower-${i}_train_ids.npy n_envs=16 test_n=128 env_cfg.fixed_max_ep_len=${max_ep_len} \
            env_cfg.partial_trajectories=True \
            env_cfg.goal=tower-${i} \
            model=${model} \
            save_results_path=${base}.tower-${i}.csv \
            # env_cfg.show_goal_visuals=False env_cfg.obs_mode=state_visual \ # uncomment to bench models using visual pipeline that is used in real world experiments
            device=cuda env_cfg.sub_goals=True
    done
done