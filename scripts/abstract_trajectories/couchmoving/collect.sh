### Generate abstract trajectory datasets for Couchmoving

# Couchmoving Short 3 (Train)
python abstract_trajectories/couchmoving/collect_dataset.py max_walks=3 walk_dist_range="(12, 20)" N=2400 sparsity=2.5e-2

# Couchmoving Long 3
python abstract_trajectories/couchmoving/collect_dataset.py max_walks=3 walk_dist_range="(20, 24)" N=2400 sparsity=2.5e-2

# Couchmoving Long 4
python abstract_trajectories/couchmoving/collect_dataset.py max_walks=4 walk_dist_range="(20, 24)" N=2400 sparsity=2.5e-2

# Couchmoving Long 5
python abstract_trajectories/couchmoving/collect_dataset.py max_walks=5 walk_dist_range="(20, 24)" N=2400 sparsity=2.5e-2
