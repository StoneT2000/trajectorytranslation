import numpy as np


def resample_teacher_trajectory(obs, max_dist=4e-2):
    lengths = []
    lengths_before = []

        # resample the points
    sampled_os = [obs[0]]
    prev_o = obs[0]
    for i in range(1, len(obs)):
        prev_o
        o2=obs[i]
        dist = np.linalg.norm(o2-prev_o)
        if dist <= max_dist: # then we skip to next i
            continue
        while dist >= max_dist:
            prev_o = prev_o + ((o2-prev_o)/np.linalg.norm((o2-prev_o))) * max_dist
            dist = np.linalg.norm(o2-prev_o)
            sampled_os.append(prev_o)
    sampled_os.append(obs[-1])
    new_obs = np.vstack(sampled_os)
    return new_obs