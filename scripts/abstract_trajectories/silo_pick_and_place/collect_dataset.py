import json
import pickle
from omegaconf import OmegaConf
from skilltranslation.envs.blockstacking import BlockStackFloatPandaEnv
import gym
import numpy as np
from mani_skill2.utils.wrappers import ManiSkillActionWrapper, NormalizeActionWrapper
import os.path as osp
from skilltranslation.envs.blockstacking.env import BlockStackMagicPandaEnv
from skilltranslation.utils.sampling import resample_teacher_trajectory
from pathlib import Path
import sapien.core as sapien
def animate(imgs, filename="animation.mp4", _return=True, fps=10):
    if isinstance(imgs, dict):
        imgs = imgs["image"]
    print(f"animating {filename}")
    from moviepy.editor import ImageSequenceClip

    imgs = ImageSequenceClip(imgs, fps=fps)
    imgs.write_videofile(filename, fps=fps)
    if _return:
        from IPython.display import Video

        return Video(filename, embed=True)


def collect_magic_teacher_traj(env: BlockStackMagicPandaEnv, render=False, mode='human', change_in_height=0, change_in_goal_height=0, release_dist=0.096, dreamed=True, init_offset=np.zeros(3)):
    assert env.connected == False

    assert env.goal_coords is not None
    init_xyz = env.get_articulations()[0].get_qpos()[:3]
    magic_traj=dict(
        observations=[],
        attns=[],
    )
    vids=[]
    if render and mode == 'human':
        env.render()
        env.get_viewer().paused=True
    robot = env.get_articulations()[0]
    curr_xyz=env.get_articulations()[0].get_qpos()[:3]
    grasping = False
    def goto(xyz, gripper_action, attn=0):
        assert gripper_action is None
        nonlocal grasping, curr_xyz, robot
        i_step = 0
        target=np.array(xyz)
        observations=[]  # during this goto()
        attns = []

        if dreamed:
            dist_to_goal = np.linalg.norm(curr_xyz - xyz)
            block: sapien.Actor = env.blocks[attn]
            while dist_to_goal > 1e-2:
                pose = block.get_pose()
                dense_obs=np.zeros(10)
                dense_obs[:3]=curr_xyz
                dense_obs[3:]=np.hstack([pose.p, pose.q])
                observations.append(
                    dense_obs
                )
                deltas = (xyz - curr_xyz) * 10e-2
                new_xyz = curr_xyz + deltas
                robot.set_qpos(np.array([*new_xyz, 0, 0]))
                env.pointmass.set_pose(pose=sapien.Pose(np.array([*new_xyz])))
                if grasping:
                    
                    pose.set_p(pose.p + deltas)
                    block.set_pose(pose)
                dist_to_goal = np.linalg.norm(new_xyz - xyz)
                curr_xyz = new_xyz
                attns.append(attn)
                
                if render:
                    
                    if mode == 'rgb_array':
                        vids.append(env.render(mode))
                    else:
                        env.render(mode)
        else:
            while True:
                robot_qpos=env.get_articulations()[0].get_qpos()
                delta_xyz=target - robot_qpos[:3]
                robot_to_target_dist=np.linalg.norm(delta_xyz)
                vel=np.linalg.norm(env.unwrapped._agent._robot.get_qvel()[:3])
                if robot_to_target_dist < 0.01 and vel < 0.05:
                    break
                action=np.zeros(4)
                action[:3]=delta_xyz.copy()
                # action[-1]=gripper_action
                # print(action)
                if 'NormalizeActionWrapper' in str(env):
                    action[:3]=action[:3] / env.env.action_space.high[0]
                    # gripper always +1
                    action[-1]=+1
                # collect observations
                obs_dict=env.get_obs()
                dense_obs=np.zeros(10)
                dense_obs[:3]=obs_dict['agent']['qpos'][:3]
                dense_obs[3:]=obs_dict['extra']['obs_blocks']['absolute'][attn * 7: attn * 7 + 7]
                observations.append(dense_obs)
                attns.append(attn)
                # attentions.append(attn)
                # print("{}: dist {:.4f}, act {}, cur xyz {}, target xyz {}".format(
                #     i_step, robot_to_target_dist, action, robot_qpos[:3], xyz
                # ))
                assert action[-1] == +1
                env.step(action)

                if render:
                    if mode == 'rgb_array':
                        vids.append(env.render(mode))
                    else:
                        
                        env.render(mode)
                i_step+=1
                if i_step > 200:
                    return False
        
        magic_traj['observations']+=observations
        magic_traj["attns"] += attns
        return True
    # goto(xyz=init_xyz+init_offset,gripper_action=None, attn=0)
    magic_traj=dict(
        observations=[],
        attns=[]
    )
    for i in range(len(env.blocks)):
        block=env.blocks[i]
        block_pos=env.blocks[i].pose.p

        
        
        grasp_pos=[block_pos[0], block_pos[1], 0.06]
        itmd = [init_xyz[0], init_xyz[1], 0.35]
        # if not goto(xyz=itmd, gripper_action=None, attn=i): return None
        # if not goto(xyz=above_pos, gripper_action=None, attn=i): return None
        if not goto(xyz=grasp_pos, gripper_action=None, attn=i): return None
        if not dreamed:
            env.magic_grasp(block)
        grasping=True
        goal_pos=env.goal_coords[i]
        above_pos=[(block_pos[0]+goal_pos[0])/2, 0.24, 0.1]
        if np.random.random() > 0.5:
            above_pos[1] *= -1
        if not goto(xyz=above_pos, gripper_action=None, attn=i): return None

        goal_pos=env.goal_coords[i]
        goal_above_pos=[goal_pos[0], goal_pos[1], goal_pos[2] + 0.096 + 0.05+ change_in_goal_height + release_dist]
        release_pos=[goal_pos[0], goal_pos[1], goal_pos[2] + 0.06]

        # if not goto(xyz=goal_above_pos, gripper_action=None, attn=i): return None
        if not goto(xyz=release_pos, gripper_action=None, attn=i): return None
        if not dreamed:
            env.magic_release()
        grasping=False
        # if not goto(xyz=goal_above_pos, gripper_action=None, attn=i): return None

        # itmd = [init_xyz[0], init_xyz[1], 0.35]
        # if not goto(xyz=itmd, gripper_action=None, attn=i): return None

        # # back to init
        # if not goto(xyz = init_xyz, gripper_action=None, attn=i): return None
        # # last observation and attention:
        # if dreamed:
        #     # we simply step forward in env just to get the next block spawned
        #     env.step(np.zeros(4))
    obs_dict=env.get_obs()
    dense_obs=np.zeros(10)
    dense_obs[:3]=obs_dict['agent']['qpos'][:3]
    dense_obs[3:]=obs_dict['extra']['obs_blocks']['absolute'][i * 7:i * 7 + 7]
    if dreamed:
        # dense_obs[:3] = curr_xyz
        # dense_obs[3:6] = block_pos
        pass
    else:
        magic_traj['observations'].append(dense_obs)

    magic_traj['attns'] = np.array(magic_traj['attns'], dtype=int)
    magic_traj['observations'] = np.array(magic_traj['observations'], dtype=np.float32)
    # assert len(float_traj['attentions']) == len(float_traj['observations'])
    if render and mode == 'rgb_array':
        return magic_traj, vids

    if not render:
        return magic_traj
    return magic_traj


if __name__ == '__main__':
    cfg = OmegaConf.from_cli()
    from tqdm import tqdm
    np.set_printoptions(suppress=True, precision=3)
    save_path = cfg.save_path
    # size = cfg.size
    N = 2
    if "n" in cfg:
        N = cfg.n
    render =False
    if "render" in cfg:
        render = cfg.render
    env_name = 'BlockStackMagic-v0'
    env = gym.make(
        env_name,
        reward_mode='sparse',
        obs_mode='state_dict',
        num_blocks=1,
        goal='pick_and_place_silo',
        spawn_all_blocks=False,
    )
    env = ManiSkillActionWrapper(env)
    env = NormalizeActionWrapper(env)
    dataset = dict(teacher=dict())
    all_ids = []
    all_lens = []
    dreamed=True
    for i in tqdm(range(N)):
        env.reset(seed=i)
        traj = collect_magic_teacher_traj(env, render=render, mode='human', change_in_goal_height=0.2, change_in_height=0.2, dreamed=dreamed)
        if traj is None: continue
        max_block_id = traj["attns"].max()
        all_obs = []
        attns = []
        for b_id in sorted(np.unique(traj["attns"])):
            obs = traj["observations"][np.where(traj["attns"] == b_id)]
            obs = resample_teacher_trajectory(obs, max_dist=4e-2)
            # dataset["teacher"][f"{i}-{b_id}"] = dict(observations=obs)
            attns.append(np.ones(len(obs), dtype=int)*b_id)
            all_obs.append(obs)
        all_obs = np.concatenate(all_obs)
        attns = np.concatenate(attns)
        dataset["teacher"][f"{i}"] = dict(observations=all_obs)
        all_ids.append(i)
        all_lens += [len(all_obs)]
    Path(osp.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    with open(osp.join(save_path), "wb") as f:
        pickle.dump(dataset, f)
    ids = sorted(all_ids)
    np.random.shuffle(ids)
    with open(osp.join(osp.dirname(save_path), "meta.json"), "w") as f:
        json.dump(dict(
            avg_traj_length=np.mean(all_lens),
            # sparsity=sparsity
        ), f)
    print("Avg traj length", np.mean(all_lens))
    np.save(osp.join(osp.dirname(save_path), f"dataset_train_ids_64.npy"), ids[:64])
    np.save(osp.join(osp.dirname(save_path), f"dataset_train_ids_128.npy"), ids[:128])
    np.save(osp.join(osp.dirname(save_path), f"dataset_train_ids_512.npy"), ids[:512])
    np.save(osp.join(osp.dirname(save_path), f"dataset_test_ids.npy"), ids[512:])
    print(f"=== Generated {len(dataset['teacher'])} teacher trajectories ===")