import os.path as osp
import pickle

import numpy as np
from omegaconf import OmegaConf

from tr2.envs.maze.traj_env import MazeTrajectory


def solve(x, render=False):
    traj_id, teacher_dataset, fragment, env_seed, max_noise = x
    np.random.seed(0)
    env = MazeTrajectory(
        trajectories=[traj_id],
        trajectories_dataset=teacher_dataset,
        reward_type="lcs_dp",
        max_trajectory_length=500,
        stack_size=1,
        patch_size=3,
        trajectory_sample_skip_steps=3,
        fixed_max_ep_len=500,
        task_agnostic=False,
        offscreen_only=not render,
    )
    env.seed(env_seed)
    obs = env.reset()
    student_obs = [obs["observation"]]
    student_acts = []
    if render:
        env.render()
        env.viewer.paused = True
        env.draw_teacher_trajectory(skip=env.trajectory_sample_skip_steps)
    orient_left = True
    mode = "move"
    target_angle = obs["observation"][2:4]
    ep_len = 0
    last_turn_idx = -10
    rewards = []
    rotations = 0
    while True:
        ep_len += 1
        # if ep_len < 20:
        #     action = np.zeros(3)
        #     obs, r, d, info = env.step(action)
        #     rewards.append(r)
        #     student_obs.append(obs["observation"])
        #     if render:
        #         env.render()
        #     continue
        action = np.zeros(3)
        teacher_obs = obs["teacher_frames"][obs["teacher_attn_mask"]]
        agent_xy = obs["observation"][:2]
        agent_angle = obs["observation"][2:4]

        diffs = np.linalg.norm(teacher_obs - agent_xy, axis=1)
        image_patch = obs["observation"][4:]
        traj_idx = diffs.argmin()

        allow_turns = True
        if traj_idx > len(teacher_obs) - 4:
            allow_turns = False
        if allow_turns:
            direction = teacher_obs[traj_idx + 1] - teacher_obs[traj_idx]
            next_dir = teacher_obs[traj_idx + 2] - teacher_obs[traj_idx + 1]
            turning_angle = np.arccos(
                np.dot(direction, next_dir) / (np.linalg.norm(direction) * np.linalg.norm(next_dir) + 1e-10)
            )
        angle_diff = np.linalg.norm(agent_angle - target_angle)
        if mode == "turn":
            # check if turn is complete
            # print(agent_angle, target_angle, angle_diff)
            if orient_left and angle_diff < 5e-2:
                mode = "move"
                rotations+=1
            elif not orient_left and angle_diff < 5e-2:
                mode = "move"
                rotations+=1
        if mode == "move":
            look_ahead_idx = min(traj_idx + 2, len(teacher_obs) - 1)
            action[:2] = teacher_obs[look_ahead_idx] - agent_xy + np.random.randn(2) * max_noise
            # print(image_patch)
            if turning_angle > 1:
                
                # print("Try and turn")
                if orient_left:
                    action[2] = 0.8
                    # print("Turn corner counterc")
                else:
                    # print("Turn corner clock")
                    action[2] = -0.8
            if image_patch.sum() < 4 and ep_len - last_turn_idx > 10 and traj_idx <= len(teacher_obs) - 3:
                # print("TURNING")
                direction = teacher_obs[traj_idx + 1] - teacher_obs[traj_idx]
                prev_march_obs = teacher_obs[traj_idx + 1]
                for i in range(traj_idx + 2, len(teacher_obs)):
                    # check where teacher turns. teacher_obs[i]
                    new_dir = teacher_obs[i] - prev_march_obs
                    angle = np.arccos(
                        np.dot(direction, new_dir) / (np.linalg.norm(direction) * np.linalg.norm(new_dir) + 1e-10)
                    )
                    if angle > 1:
                        new_dir_sign = np.sign(teacher_obs[i + 1] - teacher_obs[i])
                        # print("found turn up ahead!", "curr angle", agent_angle, "turn angle" ,angle, new_dir_sign)
                        mode = "turn"
                        if new_dir_sign[0] == -1 and new_dir_sign[1] == 0:
                            target_angle = np.array([0, -1])
                            orient_left = True
                            if np.sign(direction)[1] == -1:
                                orient_left = False
                        elif new_dir_sign[0] == 1 and new_dir_sign[1] == 0:
                            target_angle = np.array([0, 1])
                            orient_left = True
                            if np.sign(direction)[1] == 1:
                                orient_left = False
                        elif new_dir_sign[0] == 0 and new_dir_sign[1] == 1:
                            target_angle = np.array([1, 0])
                            orient_left = True
                            if np.sign(direction)[0] == -1:
                                orient_left = False
                        elif new_dir_sign[0] == 0 and new_dir_sign[1] == -1:
                            target_angle = np.array([-1, 0])
                            orient_left = True
                            if np.sign(direction)[0] == 1:
                                orient_left = False
                        break
                    else:
                        prev_march_obs = teacher_obs[i]
        elif mode == "turn":
            # print("agulardiff", angle_diff)
            last_turn_idx = ep_len
            if orient_left:
                action = np.zeros(3)
                action[2] = 1
            else:
                action = np.zeros(3)
                action[2] = -1
            if angle_diff < 4e-1:
                action[2] *= angle_diff + 1e-1

        # print(mode, agent_xy, agent_angle, action)
        student_acts.append(action)
        obs, r, d, info = env.step(action)
        rewards.append(r)
        student_obs.append(obs["observation"])
        if render:
            env.render()
        if info["task_complete"] or d:
            break
        # uncomment code below to force stop early for shorter trajectories
        if fragment:
            if ep_len > 80:
                break
    if info["task_complete"] or fragment:
        # print("success")
        # if actual success or generating fragmented trajectories, keep them
        pass
    else:
        # print(f"{traj_id} - past timelimit")
        return None
    rtg = np.cumsum(rewards[::-1])[::-1] 
    env.close()
    return dict(observations=np.vstack(student_obs), actions=np.vstack(student_acts), returns_to_go=rtg, traj_id=traj_id)


if __name__ == "__main__":
    from tqdm import tqdm

    cfg = OmegaConf.from_cli()
    max_walks = 5
    walk_dist_range = (12, 30)
    random_init = False
    fragment = False
    add_info = ""
    if "max_walks" in cfg:
        max_walks = cfg.max_walks
    if "walk_dist_range" in cfg:
        walk_dist_range = eval(cfg.walk_dist_range)
    if "fragment" in cfg: fragment=cfg.fragment
    if fragment:
        add_info = "_fragment"
    teacher_dataset = f"datasets/maze/couch_{max_walks}_corridorrange_{'_'.join([str(x) for x in walk_dist_range])}{add_info}/dataset_teacher.pkl"
    if fragment:
        with open(f"datasets/maze/couch_{max_walks}_corridorrange_{'_'.join([str(x) for x in walk_dist_range])}{add_info}/dataset_teacher_aug.pkl", "rb") as f:
            teacher_dataset_aug = pickle.load(f)
    with open(teacher_dataset, "rb") as f:
        dataset = pickle.load(f)
    student_trajectories = {}
    N = len(dataset["teacher"])
    max_noise = 1e-1
    student_traj = solve((14, teacher_dataset,fragment,0, max_noise), render=True)
    # student_traj = solve((1,teacher_dataset,True), render=True)
    i = 0
    from multiprocessing import Pool

    ids = list(dataset["teacher"].keys())
    args = []
    env_seed = 0
    for traj_id in ids:
        args.append(
            (traj_id, teacher_dataset, fragment, env_seed, max_noise)
        )
    with Pool(10) as p:
        trajs = list(tqdm(p.imap(solve, args), total=len(ids)))
    for traj in trajs:
        if traj is not None:
            student_trajectories[traj["traj_id"]] = traj
    dataset["student"] = student_trajectories
    if fragment:
        dataset["teacher"] = teacher_dataset_aug["teacher"]
    with open(f"{osp.join(osp.dirname(teacher_dataset), 'dataset.pkl')}", "wb") as f:
        pickle.dump(dataset, f)
