# Training set
python scripts/abstract_trajectories/blockstacking/collect_dataset_train.py save_path=datasets/blockstacking/dataset.pkl n=4000 cpu=16

# tower-9
python scripts/abstract_trajectories/blockstacking/collect_dataset_test.py n=32 goal=tower-9 save_path=datasets/blockstacking/dataset_tower-9.pkl &

# tower-8
python scripts/abstract_trajectories/blockstacking/collect_dataset_test.py n=32 goal=tower-8 save_path=datasets/blockstacking/dataset_tower-8.pkl &

# tower-7
python scripts/abstract_trajectories/blockstacking/collect_dataset_test.py n=32 goal=tower-7 save_path=datasets/blockstacking/dataset_tower-7.pkl &

# tower-6
python scripts/abstract_trajectories/blockstacking/collect_dataset_test.py n=32 goal=tower-6 save_path=datasets/blockstacking/dataset_tower-6.pkl &

# tower-5
python scripts/abstract_trajectories/blockstacking/collect_dataset_test.py n=32 goal=tower-5 save_path=datasets/blockstacking/dataset_tower-5.pkl &

# tower-4
python scripts/abstract_trajectories/blockstacking/collect_dataset_test.py n=32 goal=tower-4 save_path=datasets/blockstacking/dataset_tower-4.pkl &

# tower-3
python scripts/abstract_trajectories/blockstacking/collect_dataset_test.py n=32 goal=tower-3 save_path=datasets/blockstacking/dataset_tower-3.pkl &

# pyramid-5
python scripts/abstract_trajectories/blockstacking/collect_dataset_test.py goal=pyramid-5 save_path=datasets/blockstacking/dataset_pyramid-5.pkl n=32 &
# pyramid-4
python scripts/abstract_trajectories/blockstacking/collect_dataset_test.py goal=pyramid-4 save_path=datasets/blockstacking/dataset_pyramid-4.pkl n=32 &
# pyramid-3
python scripts/abstract_trajectories/blockstacking/collect_dataset_test.py goal=pyramid-3 save_path=datasets/blockstacking/dataset_pyramid-3.pkl n=32 &
# pyrmaid-2
python scripts/abstract_trajectories/blockstacking/collect_dataset_test.py goal=pyramid-2 save_path=datasets/blockstacking/dataset_pyramid-2.pkl n=32 &