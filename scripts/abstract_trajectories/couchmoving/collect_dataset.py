import json
import pickle
from pathlib import Path

import numpy as np

from tr2.envs.couchmoving.env import CouchmovingEnv
from tr2.utils.sampling import resample_teacher_trajectory


def generate_teacher(env: CouchmovingEnv, seed, teacher_seed, render=False, fragment=False,augmented=False, sparsity=1e-1, env_cfg=dict(

)):
    def extract_teacher_obs(obs):
        return obs["agent"]
    np.random.seed(teacher_seed)
    done = False
    env.seed(seed)
    obs = env.reset()
    agent_xy = obs["agent"][:2]
    target_xy = obs["target"]
    curr_j = agent_xy
    path_locs = np.array([env._scale_world(x) for x in env.path_locs])
    teacher_obs = []
    for i in range(1, len(path_locs)):
        target = path_locs[i]
        path_locs[i-1]
        curr = path_locs[i - 1]
        teacher_obs.append(curr)
        while np.linalg.norm(curr - target) > 1e-2:
            dist = np.linalg.norm(curr - target)
            curr = curr + ((target - curr) / dist) * min(1e-2, dist)
            teacher_obs.append(curr)

    init_state = env._get_state()
    teacher_obs = np.array(teacher_obs)
    if fragment:
        # cut off earlier part of teacher obs
        diffs_s = np.linalg.norm(init_state["agent_info"]["p"][:2] - teacher_obs, axis=1)
        diffs_t = np.linalg.norm(init_state["target"] - teacher_obs, axis=1)
        start_idx = diffs_s.argmin()
        if augmented:
            teacher_obs[:start_idx] = np.random.randn(start_idx, 2)
        else:
            teacher_obs = teacher_obs[start_idx:]
        # print("###", len(teacher_obs))
    # traj["observations"] 
    teacher_obs = resample_teacher_trajectory(teacher_obs, max_dist=sparsity)#traj["observations"])
    if render:
        for o in teacher_obs[::1]:
            env._add_visual_target(o[0], o[1])
    if render:
        viewer = env.render()
        while True:
            viewer=env.render("human")
            if env.viewer.window.key_down("c"):
                break
        # env.viewer.paused=True
    # print(len(init_state["walls"]))
    return dict(observations=teacher_obs, env_init_state=init_state, seed=seed, env_cfg=env_cfg)
from omegaconf import OmegaConf, ValidationError

if __name__ == "__main__":
    cfg = OmegaConf.from_cli()
    short = True
    medium = False
    render = False
    long = False
    N=cfg.N
    max_walks = 5
    walk_dist_range = (12,30)
    tiny=False
    fragment = False
    augmented = False
    sparsity = 1e-1
    ds_path = "couchmoving"
    if "path" in cfg: ds_path = cfg.path
    if "sparsity" in cfg: sparsity = cfg.sparsity
    if "short" in cfg: short = cfg.short
    if "max_walks" in cfg: max_walks = cfg.max_walks
    if "render" in cfg: render = cfg.render
    if "walk_dist_range" in cfg: walk_dist_range = eval(cfg.walk_dist_range)
    if short:
        env_cfg = dict(
            max_walks=5,
            walk_dist_range=(12,30),
            world_size=200,
        )
    env_cfg["max_walks"] = max_walks
    env_cfg["walk_dist_range"] = walk_dist_range
    env = CouchmovingEnv(agent_type="couch", offscreen_only=not render, random_init=fragment, target_next_chamber=True, **env_cfg)
    dataset = dict(teacher=dict())
    teacher_seed = 0
    from tqdm import tqdm
    env_seed = 0
    pbar = tqdm(total=N)
    traj_ids = []
    traj_lengths = []
    
    while len(dataset["teacher"]) < N:
        import time
        
        traj = None
        retries = 0
        while traj is None and retries < 20:
            traj = generate_teacher(env, env_seed, teacher_seed=teacher_seed, render=render, fragment=fragment, augmented=augmented, sparsity=sparsity, env_cfg=env_cfg)
            # import ipdb;ipdb.set_trace()
            traj_lengths.append(len(traj["observations"]))
            teacher_seed += 1
            retries += 1
        if traj is not None:
            dataset["teacher"][env_seed] = traj
            pbar.update()
            traj_ids.append(env_seed)
        env_seed += 1
    add_info = ""
    # if max_walks != 5:
    add_info = f"_{max_walks}"
    add_info += f"_corridorrange_{'_'.join([str(x) for x in walk_dist_range])}"

    augmented_info = ""
    if augmented:
        augmented_info = "_aug"
    Path(f"datasets/{ds_path}/couch{add_info}").mkdir(parents=True, exist_ok=True)
    with open(f"datasets/{ds_path}/couch{add_info}/dataset_teacher{augmented_info}.pkl", "wb") as f:
        pickle.dump(dataset, f)

    with open(f"datasets/{ds_path}/couch{add_info}/meta.json", "w") as f:
        json.dump(dict(
            avg_traj_length=np.mean(traj_lengths),
            sparsity=sparsity
        ), f)
    print("Avg traj length", np.mean(traj_lengths))
    np.random.seed(0)
    np.random.shuffle(traj_ids)
    for split in [int(0.2*N), int(0.5*N), int(0.8*N), N]:
        np.save(f"datasets/{ds_path}/couch{add_info}/dataset_train_ids_{split}.npy", traj_ids[:split])
        np.save(f"datasets/{ds_path}/couch{add_info}/dataset_val_ids_{split}.npy", traj_ids[split:])