from multiprocessing import Pool
from pathlib import Path
from omegaconf import OmegaConf
from tr2.envs.blockstacking import BlockStackFloatPandaEnv
import gym
import numpy as np
from mani_skill2.utils.wrappers import ManiSkillActionWrapper, NormalizeActionWrapper
from tr2.envs.blockstacking import BlockStackFloatPandaEnv, BlockStackMagicPandaEnv
import gym
import numpy as np
from mani_skill2.utils.wrappers import ManiSkillActionWrapper, NormalizeActionWrapper

from tr2.utils.sampling import resample_teacher_trajectory


def animate(imgs, filename="animation.mp4", _return=True, fps=10):
    if isinstance(imgs, dict):
        imgs=imgs["image"]
    print(f"animating {filename}")
    from moviepy.editor import ImageSequenceClip

    imgs=ImageSequenceClip(imgs, fps=fps)
    imgs.write_videofile(filename, fps=fps)
    if _return:
        from IPython.display import Video

        return Video(filename, embed=True)


def collect_heuristic_student_traj(env: BlockStackFloatPandaEnv, render=False, mode='human', change_in_height=0, change_in_goal_height=0, init_offset=np.zeros(3)):

    assert env.goal_coords is not None
    ee_init_pos=env.unwrapped._agent.hand_link.get_pose().p

    robot_arm_traj=dict(

        observations=[],
        actions=[],
    )
    vids=[]
    if render and mode == 'human':
        env.render()
        env.get_viewer().paused=True

   

    def goto(xyz, gripper_action):
        nonlocal  block
        i_step=0
        target=np.array(xyz)
        observations=[]  ## during this goto()
        actions=[]
        while True:
            ee_pos=env.unwrapped._agent.hand_link.get_pose().p
            delta_xyz=target - ee_pos
            robot_to_target_dist=np.linalg.norm(delta_xyz)
            vel=np.linalg.norm(env.unwrapped._agent._robot.get_qvel()[:3])

            grasp_flag=env.unwrapped._agent.check_grasp(block)
            gripper_flag=grasp_flag != (gripper_action > 0)

            if robot_to_target_dist < 0.01 and vel < 0.05 and gripper_flag:
                break
            action=np.zeros(4)
            action[:3]=delta_xyz.copy()
            action[-1]=gripper_action
            # print(action)
            if 'NormalizeActionWrapper' in str(env):
                action[:3]=action[:3] / env.env.action_space.high[0]
                action[-1]=1 if gripper_action > 0 else -1
            # print(action)
            # print("{}: dist {:.4f}, act {}, cur xyz {}, target xyz {}".format(
            #     i_step, robot_to_target_dist, action, ee_pos, xyz
            # ))
            # print(env.unwrapped._agent.hand_link.get_pose())
            obs_dict=env.get_obs()
            dense_obs=np.zeros(38)
            dense_obs[:9]=obs_dict['agent']['qpos']
            dense_obs[9:18]=obs_dict['agent']['qvel']
            # only one block -- so
            dense_obs[18:25]=obs_dict['extra']['obs_blocks']['absolute']
            panda_hand=None
            for link in env.get_articulations()[0].get_links():
                if link.name == 'panda_hand':
                    panda_hand=link
            dense_obs[25:28]=panda_hand.pose.p
            dense_obs[28:32]=panda_hand.pose.q
            block_velocity=env.blocks[0].get_velocity()
            block_angular_velocity=env.blocks[0].get_angular_velocity()
            dense_obs[32:35]=block_velocity
            dense_obs[35:38]=block_angular_velocity
            observations.append(dense_obs)
            actions.append(action)

            env.step(action)

            if render:
                if mode == 'rgb_array':
                    vids.append(env.render(mode))
                else:
                    env.render(mode)
            i_step+=1
            if i_step > 200:
                return False
        robot_arm_traj['observations']+=observations
        robot_arm_traj['actions']+=actions
        return True

    block=env.blocks[0]
    if not goto(xyz=ee_init_pos + init_offset, gripper_action=0.04): return None




    ## delte previous goto info, was for init
    robot_arm_traj=dict(

        observations=[],
        actions=[],
    )
    env_init_state = env.get_state()
    for i in range(len(env.blocks)):
        block=env.blocks[i]
        block_pos=env.blocks[i].pose.p
        above_pos=[block_pos[0], block_pos[1], 0.25 + change_in_height]
        grasp_pos=[block_pos[0], block_pos[1], 0.113]
        itmd = [ee_init_pos[0], ee_init_pos[1], 0.35]
        if not goto(xyz=itmd, gripper_action=0.04): return None
        if not goto(xyz=above_pos, gripper_action=0.04): return None
        if not goto(xyz=grasp_pos, gripper_action=0.04): return None
        if not goto(xyz=grasp_pos, gripper_action=0): return None
        if not goto(xyz=above_pos, gripper_action=0): return None

        goal_pos=env.goal_coords[i]
        goal_above_pos=[goal_pos[0], goal_pos[1], goal_pos[2] + 0.096 + 0.09+ change_in_goal_height]
        release_pos=[goal_pos[0], goal_pos[1], goal_pos[2] + 0.096]

        if not goto(xyz=goal_above_pos, gripper_action=0): return None
        if not goto(xyz=release_pos, gripper_action=0): return None
        if not goto(xyz=release_pos, gripper_action=0.04): return None
        if not goto(xyz=goal_above_pos, gripper_action=0.04): return None
        itmd = [ee_init_pos[0], ee_init_pos[1], 0.35]
        if not goto(xyz=itmd, gripper_action=0.04): return None
        # go back to init
        if not goto(xyz=ee_init_pos, gripper_action=0.04): return None
    # last observation
    obs_dict=env.get_obs()
    dense_obs=np.zeros(38)
    dense_obs[:9]=obs_dict['agent']['qpos']
    dense_obs[9:18]=obs_dict['agent']['qvel']
    # only one block -- so
    dense_obs[18:25]=obs_dict['extra']['obs_blocks']['absolute']
    panda_hand=None
    for link in env.get_articulations()[0].get_links():
        if link.name == 'panda_hand':
            panda_hand=link
    dense_obs[25:28]=panda_hand.pose.p
    dense_obs[28:32]=panda_hand.pose.q
    block_velocity=env.blocks[0].get_velocity()
    block_angular_velocity=env.blocks[0].get_angular_velocity()
    dense_obs[32:35]=block_velocity
    dense_obs[35:38]=block_angular_velocity
    robot_arm_traj['observations'].append(dense_obs)
    robot_arm_traj['env_init_state'] = env_init_state
    if render and mode == 'rgb_array':
        return vids
    if not render:
        return robot_arm_traj
    return robot_arm_traj


def collect_magic_teacher_traj(env: BlockStackMagicPandaEnv, render=False, mode='human', change_in_height=0, change_in_goal_height=0,init_offset=np.zeros(3)):
    assert env.connected == False

    assert env.goal_coords is not None
    init_xyz = env.get_articulations()[0].get_qpos()[:3]
    magic_traj=dict(
        observations=[],
    )
    vids=[]
    if render and mode == 'human':
        env.render()
        env.get_viewer().paused=True
    
    def goto(xyz, gripper_action, attn=0):
        assert gripper_action is None
        nonlocal block
        i_step=0
        target=np.array(xyz)
        observations=[]  # during this goto()
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
        return True
    block = env.blocks[0]
    goto(xyz=init_xyz+init_offset,gripper_action=None, attn=0)

    magic_traj=dict(
        observations=[],
    )
    for i in range(len(env.blocks)):
        block=env.blocks[i]
        block_pos=env.blocks[i].pose.p

        above_pos=[block_pos[0], block_pos[1], 0.25+change_in_height]
        grasp_pos=[block_pos[0], block_pos[1], 0.113]
        itmd = [init_xyz[0], init_xyz[1], 0.35]
        if not goto(xyz=itmd, gripper_action=None, attn=i): return None
        if not goto(xyz=above_pos, gripper_action=None, attn=i): return None
        if not goto(xyz=grasp_pos, gripper_action=None, attn=i): return None
        env.magic_grasp(block)
        if not goto(xyz=above_pos, gripper_action=None, attn=i): return None

        goal_pos=env.goal_coords[i]
        goal_above_pos=[goal_pos[0], goal_pos[1], goal_pos[2] + 0.096 + 0.05+ change_in_goal_height]
        release_pos=[goal_pos[0], goal_pos[1], goal_pos[2] + 0.096]

        if not goto(xyz=goal_above_pos, gripper_action=None, attn=i): return None
        if not goto(xyz=release_pos, gripper_action=None, attn=i): return None
        env.magic_release()
        if not goto(xyz=goal_above_pos, gripper_action=None, attn=i): return None

        itmd = [init_xyz[0], init_xyz[1], 0.35]
        if not goto(xyz=itmd, gripper_action=None, attn=i): return None

        # back to init
        if not goto(xyz = init_xyz, gripper_action=None, attn=i): return None
        # last observation and attention:
    obs_dict=env.get_obs()
    dense_obs=np.zeros(10)
    dense_obs[:3]=obs_dict['agent']['qpos'][:3]
    dense_obs[3:]=obs_dict['extra']['obs_blocks']['absolute'][i * 7:i * 7 + 7]
    # float_traj['attentions'].append(i) ## here 'i' is the last block
    magic_traj['observations'].append(dense_obs)

    # assert len(float_traj['attentions']) == len(float_traj['observations'])
    if render and mode == 'rgb_array':
        return vids

    if not render:
        return magic_traj
    return magic_traj
import os.path as osp
import os
from tqdm import tqdm

def generate_dataset(id_range=(0,100),render=False,task_range='small'):
    dataset=dict(
        student=dict(),
        teacher=dict(),
    )

    teacher_env_name='BlockStackMagic-v0'
    teacher_env: BlockStackMagicPandaEnv=gym.make(
        teacher_env_name,
        reward_mode='sparse',
        obs_mode='state_dict',
        num_blocks=1,
        goal='pick_and_place_train',
        task_range=task_range
    )
    teacher_env=ManiSkillActionWrapper(teacher_env)
    teacher_env=NormalizeActionWrapper(teacher_env)

    student_env_name='BlockStackArm-v0'
    student_env: BlockStackFloatPandaEnv=gym.make(
        student_env_name,
        reward_mode='sparse',
        obs_mode='state_dict',
        num_blocks=1,
        goal='pick_and_place_train',
        controller="ee",
        task_range=task_range,

    )
    student_env=ManiSkillActionWrapper(student_env)
    student_env=NormalizeActionWrapper(student_env)

    for i in tqdm(range(id_range[0], id_range[1])):
        np.random.seed(i)
        student_env.reset(seed=i)
        teacher_env.reset(seed=i)
        assert np.all(student_env.blocks[0].pose.p == teacher_env.blocks[0].pose.p)
        assert np.all(student_env.goal_coords.flatten() == student_env.goal_coords.flatten())

        student_panda_hand = None
        for link in student_env.get_articulations()[0].get_links():
            if link.name == 'panda_hand':
                student_panda_hand = link
        assert np.allclose(student_panda_hand.get_pose().p, teacher_env.get_articulations()[0].get_qpos()[:3], atol=2e-3)

        change_height = np.random.uniform(0,0.32)
        change_goal_height = np.random.uniform(0,0.16)
        init_offset = np.random.uniform(-0.07,0.07,3)
        # print(change_height)
        # print(change_goal_height)
        student_trajectory=collect_heuristic_student_traj(student_env, render=render, change_in_height=change_height, change_in_goal_height=change_goal_height, init_offset=init_offset)
        if student_trajectory is None: 
            print(f"== traj {i} unsuccesful")
            continue
        teacher_trajectory=collect_magic_teacher_traj(teacher_env, render=False, change_in_height=change_height, change_in_goal_height=change_goal_height, init_offset=init_offset)
        if teacher_trajectory is None:
            print(f"== traj {i} unsuccesful due to teacher malformed")
            continue
        student_panda_hand = None
        for link in student_env.get_articulations()[0].get_links():
            if link.name == 'panda_hand':
                student_panda_hand = link

        pose_close = np.allclose(student_panda_hand.get_pose().p,teacher_env.get_articulations()[0].get_qpos()[:3], atol=1e-2)
        if not pose_close:
            print(f"== traj {i} unsuccesful - hand and teacher not close")
            continue

        teacher_observation=np.array(teacher_trajectory['observations'])
        teacher_observation = resample_teacher_trajectory(teacher_observation, max_dist=4e-2)
        student_observation=np.array(student_trajectory['observations'])
        student_actions=np.array(student_trajectory['actions'])
        assert len(student_actions) + 1 == len(student_observation)
        dataset['teacher'][str(i)]=dict(
            observations=teacher_observation,
            env_init_state=student_trajectory['env_init_state']
        )
        dataset['student'][str(i)]=dict(
            observations=student_observation,
            actions=student_actions
        )

    return dataset
def generate_dataset_helper(args):
    return generate_dataset(*args)

if __name__ == '__main__':
    # python 'scripts/blockstack/collect_dataset_train.py' save_path=datasets/blockstack_v3/dataset.pkl n=4000 cpu=16
    cfg = OmegaConf.from_cli()
    save_path = cfg.save_path
    # size = cfg.size
    N = 2
    if "n" in cfg:
        N = cfg.n
    cpu = 16
    if 'cpu' in cfg:
        cpu = cfg['cpu']
    ids_per_proc = N // cpu
    args = []
    for x in range(cpu):
        arg_rng = ((x * ids_per_proc, (x+1) * ids_per_proc), False, 'large')
        args.append(arg_rng)
        print(arg_rng)
    # generate_dataset((0,100), render=True)
    # exit()
    with Pool(cpu) as p:
        datasets = list(tqdm(p.imap(generate_dataset_helper, args), total=N))

    dataset = dict(student=dict(), teacher=dict())
    for d in datasets:
        for traj_id in d['teacher']:
            dataset['teacher'][traj_id] = d['teacher'][traj_id]
            dataset['student'][traj_id] = d['student'][traj_id]
    Path(osp.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    np.save(osp.join(osp.dirname(save_path), f"dataset_train_ids.npy"), sorted(list(d['teacher'].keys())))
    print(f"=== Generated {len(dataset['teacher'])} teacher trajectories ===")
    
    dataset_file=open(save_path, "wb")
    import pickle
    pickle.dump(dataset, dataset_file)
    dataset_file.close()
    exit()

    from tqdm import tqdm
    teacher_env_name='BlockStackMagic-v0'
    teacher_env: BlockStackMagicPandaEnv=gym.make(
        teacher_env_name,
        reward_mode='sparse',
        obs_mode='state_dict',
        num_blocks=1,
        goal='pick_and_place_train',
    )
    teacher_env=ManiSkillActionWrapper(teacher_env)
    teacher_env=NormalizeActionWrapper(teacher_env)

    student_env_name='BlockStackArm-v0'
    student_env: BlockStackFloatPandaEnv=gym.make(
        student_env_name,
        reward_mode='sparse',
        obs_mode='state_dict',
        num_blocks=1,
        goal='pick_and_place_train',
    )
    student_env=ManiSkillActionWrapper(student_env)
    student_env=NormalizeActionWrapper(student_env)
    vids = []
    np.random.seed(0)
    for i in tqdm(range(0,15)):
        np.random.seed(i)
        student_env.reset(seed=i)
        teacher_env.reset(seed=i)
        assert np.all(student_env.blocks[0].pose.p == teacher_env.blocks[0].pose.p)
        assert np.all(student_env.goal_coords.flatten() == student_env.goal_coords.flatten())

        student_panda_hand = None
        for link in student_env.get_articulations()[0].get_links():
            if link.name == 'panda_hand':
                student_panda_hand = link
        print(student_panda_hand.get_pose().p, teacher_env.get_articulations()[0].get_qpos()[:3])
        assert np.all(student_panda_hand.get_pose().p == teacher_env.get_articulations()[0].get_qpos()[:3])


        change_height = np.random.uniform(0,0.2)
        change_goal_height = np.random.uniform(0,0.2)
        init_offset = np.random.uniform(-0.07,0.07,3)
        print(change_height + 0.25)
        imgs = student_trajectory=collect_heuristic_student_traj(student_env, change_in_height=change_height, render=True, mode='rgb_array',change_in_goal_height=change_goal_height,init_offset=init_offset)
        vids += imgs


        imgs = teacher_trajectory=collect_magic_teacher_traj(teacher_env, change_in_height=change_height, render=True, mode='rgb_array',change_in_goal_height=change_goal_height, init_offset=init_offset)
        vids += imgs

    animate(vids, 'dataset_content.mp4',fps=30)

