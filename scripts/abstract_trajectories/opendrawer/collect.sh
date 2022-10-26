# generate abstract trajectories
python scripts/abstract_trajectories/opendrawer/collect_dataset.py n=640 mode=train cpu=16 save_path=datasets/opendrawer/dataset_open.pkl
python scripts/abstract_trajectories/opendrawer/collect_dataset.py n=160 mode=test cpu=16 save_path=datasets/opendrawer/dataset_test.pkl

# test abstract trajectories for opening two drawers are not pre-generated. They are generated on the fly by the planner in this environment