# import box_pusher.models.translation.decisiontransformer as DecisionTransformer
import importlib
import os
import os.path as osp
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.notebook import tqdm


def get_trajectory_pairs(
    path, 
    verbose=True, 
    max_length=150,
    train_ids=None,
    val_ids=None,
    train_test_split=True,
    split_seed=0,
    split_ratio=0.8,
    trajectory_sample_skip_steps=0,
):
    ext = osp.splitext(path)[1]
    if ext == ".npy":
        dataset = np.load(path, allow_pickle=True).reshape(1)[0]
    elif ext == ".pkl":
        with open(path, "rb") as f:
            dataset = pickle.load(f)
    if verbose: print(f"Loaded {len(dataset['student'])} trajectories")
    

    # load data
    trajectory_pairs = []
    trajectory_ids = []
    MAX_LENGTH=max_length
    teacher_lengths = []
    orig_teacher_lengths = []
    ignored=0
    all_traj_ids = sorted(list(dataset['student'].keys()))
    for traj_id in tqdm(all_traj_ids, total=len(all_traj_ids)):
        data = dataset['student'][traj_id]

        S_GT = data['observations'][:-1]
        S_GT_A = data['actions']
        teacher_data = dataset['teacher'][traj_id]
        if "box_pusher" in path:
            # Fixes a bug for the box pusher dataset where the last frame is faulty
            S_T_orig = teacher_data['observations'][:-1]
        else:
            S_T_orig = teacher_data['observations']
        if trajectory_sample_skip_steps > 0:
            # include first frame and exclude last
            S_T = S_T_orig[0:-1:trajectory_sample_skip_steps+1].copy()
            # add on last frame
            S_T = np.vstack([S_T, S_T_orig[-1]])
        else:
            S_T = S_T_orig
        teacher_lengths.append(len(S_T))
        orig_teacher_lengths.append(len(S_T_orig))
        if len(S_T) > MAX_LENGTH: 
            ignored+=1
            continue
        student_traj =  {"teacher": S_T, "student_obs": S_GT, "student_acts": S_GT_A}
        if "returns_to_go" in data: student_traj["student_rtg"] = data["returns_to_go"]
        trajectory_pairs.append(
           student_traj
        )
        trajectory_ids.append(traj_id)
    if verbose: print(f"{len(trajectory_pairs)} trajectories left after removing trajectories of length > {max_length}. Ignored {ignored} trajectories. Original avg teacher length: {np.mean(orig_teacher_lengths)}, Sampled avg teacher length: {np.mean(teacher_lengths)}")

    if train_ids is not None and val_ids is not None:
        train_trajectory_ids = []
        val_trajectory_ids = []
        
        train_trajectory_pairs, val_trajectory_pairs = [], []
        train_trajectory_ids_set = set(train_ids)
        val_trajectory_ids_set = set(val_ids)
        for i, t_id in enumerate(trajectory_ids):
            if t_id in train_trajectory_ids_set:
                train_trajectory_pairs.append(trajectory_pairs[i])
                train_trajectory_ids.append(t_id)
            if t_id in val_trajectory_ids_set:
                val_trajectory_pairs.append(trajectory_pairs[i])
                val_trajectory_ids.append(t_id)

        print("Train size", len(train_trajectory_ids), "Val size", len(val_trajectory_ids))
        return train_trajectory_pairs, train_trajectory_ids, val_trajectory_pairs, val_trajectory_ids
    elif train_test_split:
        rng = np.random.default_rng(split_seed)
        train_size = int(len(trajectory_pairs) * split_ratio)
        val_size = len(trajectory_pairs) - train_size
        print("Train size", train_size, "Val size", val_size)

        choices = rng.choice(np.arange(len(trajectory_pairs)), train_size, replace=False)
        idx = np.zeros(len(trajectory_pairs), dtype=bool)
        idx[choices] = True
        train_trajectory_pairs, train_trajectory_ids, val_trajectory_pairs, val_trajectory_ids = [], [], [], []
        for i, truthy in enumerate(idx):
            if truthy:
                train_trajectory_pairs.append(trajectory_pairs[i])
                train_trajectory_ids.append(trajectory_ids[i])
            else:
                val_trajectory_pairs.append(trajectory_pairs[i])
                val_trajectory_ids.append(trajectory_ids[i])
        return train_trajectory_pairs, train_trajectory_ids, val_trajectory_pairs, val_trajectory_ids

    return trajectory_pairs, trajectory_ids

class MinMaxScaler():
    def __init__(self, rng=[-1,1]):
        self.min = -1
        self.max = 1
        # min,max=-1,1 and rng=[-1,1] means no transformation actually occurs
        self.rng = rng
    def fit(self, X):
        self.max = X.max(axis=(0,1)) 
        self.min = X.min(axis=(0,1))
    def transform(self, X):
        X_std = (X - self.min) / (self.max - self.min)
        return X_std * (self.rng[1] - self.rng[0]) + self.rng[0]
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    def untransform(self, X, device=None):
        if device is not None:
            X = (X - self.rng[0]) / (self.rng[1] - self.rng[0])
            X = X * (self.max.to(device) - self.min.to(device)) + self.min.to(device)
            return X
        X = (X - self.rng[0]) / (self.rng[1] - self.rng[0])
        X = X * (self.max - self.min) + self.min
        return X