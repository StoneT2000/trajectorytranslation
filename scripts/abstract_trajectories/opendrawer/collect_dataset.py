import gym
from multiprocessing import Pool
from pathlib import Path
import os.path as osp
import pickle
import numpy as np
from skilltranslation.utils.sampling import resample_teacher_trajectory
from omegaconf import OmegaConf
try:
    import mani_skill.env
    from mani_skill.env.open_cabinet_door_drawer import OpenCabinetDrawerMagicEnv
except:
    print( "#" * 15, "no Maniskill 1", "#" * 15,)
from tqdm import tqdm
from skilltranslation.utils.animate import animate
drawer_idx = [
1000 ,
1004 ,
1005 ,
1016 ,
1021 ,
1024 ,
1027 ,
1032 ,
1033 ,
1038 ,
1040 ,
1044 ,
1045 ,
1052 ,
1054 ,
1061 , #1
1063 , #1
1066 , #1
1067 , #1
1076 , #4
1079 , #3
1082  #3
]

two_drawer_idx = [1000, 1004, 1005, 1016,1024,1033,1044, 1076,1079, 1082]
# reset z-base as well 

def sanity_check():
    env = gym.make('OpenCabinetDrawerMagic-v0')
    env.set_env_mode(obs_mode='state', reward_type='sparse')
    env.reset(level=0)
    for i in range(200):
        env.step(np.zeros(5))
    print("sanity check passed")

def collect_heuristic_teacher_traj(env: OpenCabinetDrawerMagicEnv, render=False, mode='color_image', obs_mode='state', task='open', end_move_away_dist=0.0):
    # env should be configured, seeded and reset outside
    ## checks
    if env.connected is not False:
        print("warning not connected")
        return
    if env.magic_drive is not None:
        print("magic drive missing")
        return
    
    def parse_teacher_obs(obs):
        if obs_mode == 'state' or obs_mode == 'custom':
            return np.hstack([
            obs[:3], obs[10:]
            ])
        elif obs_mode == 'pointcloud':
            center = obs['pointcloud']['xyz'].mean(0)
            return np.hstack([
                obs['agent'][:3], center # 6dims
            ])
        else:
            raise NotImplementedError()
        # bbox mode
        
    
    vids = []
    if render and mode =='human':
        env.render()
        env.viewer.paused=True
    curr_xyz = env.agent.robot.get_qpos()[:3]
    robot = env.agent.robot
    handle_bbox = parse_teacher_obs(env.get_obs())[3:]
    cabinet_qpos = env.cabinet.get_qpos()
    robot_qpos = robot.get_qpos()
    grasping = False
    teacher_traj = dict(observations=[
        np.hstack([curr_xyz, handle_bbox])
    ])
    mins, maxs = env.get_aabb_for_min_x(env.target_link)
    end_qpos = cabinet_qpos.copy()
    [[lmin, lmax]] = env.target_joint.get_limits()
    end_qpos[env.target_index_in_active_joints] = lmax
    # interpolate
    env.cabinet.set_qpos(end_qpos)
    endmins, endmaxs = env.get_aabb_for_min_x(env.target_link)
    qpos_extent_dist = maxs[0] - endmaxs[0]
    # reset cabinet
    # print(qpos_extent_dist, lmax, lmin)
    env.cabinet.set_qpos(cabinet_qpos)

    def goto(xyz, gripper_action, dist_threshold=0.01, scale_action=1):
        nonlocal curr_xyz, handle_bbox, robot, grasping, cabinet_qpos
        assert gripper_action is None
        i_step = 0
        if type(xyz) != np.ndarray:
            xyz = np.array(xyz)
        observations = []
        dist_to_goal = np.linalg.norm(xyz - curr_xyz)
        while dist_to_goal > 1e-2:
            deltas = (xyz - curr_xyz) * 5e-2
            new_xyz = curr_xyz + deltas
            dist_to_goal = np.linalg.norm(new_xyz - xyz)
            if grasping:
                handle_bbox = (handle_bbox.reshape(2,3) + deltas).flatten()
                qpos_m_ratio = (lmax - lmin) / (qpos_extent_dist) # multiply this by vector the agent moves to get delta qpos
                cabinet_qpos[env.target_index_in_active_joints] += qpos_m_ratio * np.linalg.norm(deltas)
                if cabinet_qpos[env.target_index_in_active_joints] > lmax:
                    cabinet_qpos[env.target_index_in_active_joints] = lmax
            if render:
                if mode == 'color_image':
                    img_dict = env.render(mode)
                    img=img_dict['world']['rgb']
                    vids.append(img * 255)
                else:
                    env.render("human")
            qpos = robot.get_qpos()
            qpos[:3] = new_xyz
            robot.set_qpos(qpos)
            env.cabinet.set_qpos(cabinet_qpos)
            curr_xyz = new_xyz
            obs = np.zeros(9)
            obs[:3] = curr_xyz
            obs[3:] = handle_bbox
            observations.append(obs)
        teacher_traj['observations']+=(observations)
        return True

    
    xyz = [mins[0]-0.1, (mins[1] + maxs[1])/2, (mins[2] + maxs[2]) / 2 ] ## if set_qpos -0.11 is too close, but works in this code
    init_xyz = list(env.agent.robot.get_qpos()[:3]).copy()
    end_noise = np.random.rand(3) * end_move_away_dist
    end_noise[0] = end_noise[0] * -1
    
    goal1 = np.array(init_xyz + [0.04, 0.04])
    goal1[2] = xyz[2].copy()

    goal2 = xyz.copy() + [0.04, 0.04]
    goal3 = goal2.copy()
    goal3[0] -= env.target_qpos * 1.2


    if task == 'open':
        end_xyz = goal3[:3] + end_noise
        if not goto(xyz=goal1[:3], gripper_action=None): return None

        if not goto(xyz=goal2[:3], gripper_action=None): return None
        env.magic_grasp()
        grasping=True
        if not goto(xyz=goal3[:3], gripper_action=None, dist_threshold=0.04, scale_action=2): return None
        env.magic_release()
        grasping=False
        if not goto(xyz=end_xyz, gripper_action=None): return None
        target_qpos = env.target_qpos
        actual_qpos = env.cabinet.get_qpos()[env.target_index_in_active_joints]
        if actual_qpos < target_qpos:
            print("Task Failed, Open Extent {} smaller than tagret {}".format(actual_qpos, target_qpos))
            return None
        post_teacher_xyz=env.agent.robot.get_qpos()[:3]
        returned_to_rest = np.allclose(post_teacher_xyz, end_xyz, atol=5e-2)
        if not returned_to_rest:
            print("Did no return to rest: failed")
            return None
    elif task == 'close':
        end_xyz = goal3[:3] + end_noise
        goal1 = np.array(init_xyz + [0.04, 0.04])
        goal1[2] = xyz[2].copy()
        goal2 = xyz.copy() + [0.04, 0.04]
        goal3 = goal2.copy()
        goal3[0] += env.full_open_qpos * 1.2
        if not goto(xyz=goal1[:3], gripper_action=None): return None
        if not goto(xyz=goal2[:3], gripper_action=None): return None
        env.magic_grasp()
        # print("grasped, going to ", goal3)
        if not goto(xyz=goal3[:3], gripper_action=None, dist_threshold=0.05, scale_action=2): return None
        env.magic_release()
        if not goto(xyz=goal1[:3], gripper_action=None): return None
        if not goto(xyz=end_xyz, gripper_action=None): return None
        target_qpos = env.target_qpos
        actual_qpos = env.cabinet.get_qpos()[env.target_index_in_active_joints]
        if actual_qpos > target_qpos:
            print("Close Task Failed, Open Extent {} greater than {}".format(actual_qpos, target_qpos))
            return None
        post_teacher_xyz=env.agent.robot.get_qpos()[:3]
        returned_to_rest = np.allclose(post_teacher_xyz, end_xyz, atol=5e-2)
        if not returned_to_rest:
            print("Did no return to rest: failed")
            return None

    if render and mode == 'color_image':
        return vids

    if not render:
        return teacher_traj
    return teacher_traj

def collect_dataset(mode, id_range=(0,100), drawer_rng=(0,10), render=False, obs_mode='state'): # not including mode for simplicity
    dataset = dict(
        teacher=dict()
    )
    drawer_id_list = drawer_idx
    if mode == 'opentwo':
        drawer_id_list = two_drawer_idx
    for idx in drawer_id_list[drawer_rng[0]:drawer_rng[1]]:
        teacher_env_name = 'OpenCabinetDrawerMagic_' + str(idx) + '-v0'
        student_env_name = f'OpenCabinetDrawer_{idx}-v0'

        teacher_env = gym.make(teacher_env_name)
        student_env = gym.make(student_env_name)
        teacher_env.set_env_mode(obs_mode=obs_mode, reward_type='sparse')
        student_env.set_env_mode(obs_mode=obs_mode, reward_type='sparse')

        for level in tqdm(range(id_range[0], id_range[1])):
            teacher_env.reset(level=level)
            student_env.reset(level=level)
            state = student_env.get_state()
            if teacher_env.task == 'close': continue # skip close tasks for now
            config=dict(
                target_link_id=teacher_env.target_index,
                cabinet_id=idx,
                level=level,
                task=teacher_env.task,
                env_init_state=state
            )
            # print(config)
            # print(student_env.get_obs()[33:], teacher_env.get_obs()[10:])
            student_env.set_state(state)
            if obs_mode == 'state' or obs_mode=='custom':
                init_teacher_qpos = np.hstack([student_env.get_obs()[26:29],np.array([0.04,0.04])])
            else:
                init_teacher_qpos = np.hstack([student_env.get_obs()['agent'][26:29],np.array([0.04,0.04])])

            teacher_env.agent.robot.set_qpos(init_teacher_qpos)
            
            # if render:
            #     vids = collect_heuristic_teacher_traj(teacher_env, render=True, mode='color_image', obs_mode=obs_mode, task=config['task'], end_move_away_dist=0.5)
            #     assert vids is not None
            #     animate(vids, 'env{}level{}.mp4'.format(idx, level), fps=40)
            # else
            traj_dict = collect_heuristic_teacher_traj(teacher_env, render=render, mode='human', obs_mode=obs_mode, task=config['task'], end_move_away_dist=0.25)
            if traj_dict is None:
                print(f"== env {idx} seed {level} unsuccesful")
                continue
            # print("###",len(traj_dict["observations"]))
            observations = resample_teacher_trajectory(np.array(traj_dict['observations']), max_dist=5e-2)
            dataset['teacher'][f"{idx}-{level}"] = dict(
                config=config,
                observations=observations,
                )

    return dataset
def collect_dataset_helper(args):
    return collect_dataset(args[0], args[1], args[2], args[3], args[4])
if __name__ == '__main__':
    
    sanity_check()
    cfg = OmegaConf.from_cli()
    save_path = cfg.save_path
    # size = cfg.size

    N = 2
    if "n" in cfg:
        N = cfg.n
    cpu = 16
    mode = 'train'
    obs_mode = 'state'
    if 'obs_mode' in cfg:
        obs_mode = cfg['obs_mode']
    if 'mode' in cfg:
        mode = cfg['mode']
    if 'cpu' in cfg:
        cpu = cfg['cpu']
    ids_per_proc = N // cpu
    args = []
    if mode == 'train':
        drawer_rng = (0,8)
    elif mode == 'test':
        drawer_rng = (8,22)
    elif mode == 'opentwo':
        drawer_rng = (0, len(two_drawer_idx))
    for x in range(cpu):
        arg_rng = (x * ids_per_proc, (x+1) * ids_per_proc)
        args.append((mode, arg_rng, drawer_rng, False, obs_mode))
        print(arg_rng)
    # ds = collect_dataset(id_range=(7,8), drawer_rng=(6,8), render=True, obs_mode='custom')
    # exit()
    try:
        from multiprocessing import set_start_method
        set_start_method('spawn')
    except RuntimeError:
        print('Cannot set start method to spawn')

    with Pool(cpu) as p:
        datasets = list(tqdm(p.imap(collect_dataset_helper, args), total=N))
    
    dataset = dict(student=dict(), teacher=dict())
    for d in datasets:
        for traj_id in d['teacher']:
            dataset['teacher'][traj_id] = d['teacher'][traj_id]
    Path(osp.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    np.save(osp.join(osp.dirname(save_path), f"dataset_{mode}_sim_ids.npy"), sorted(list(dataset['teacher'].keys())))
    print(f"=== Generated {len(dataset['teacher'])} teacher trajectories ===")
    
    dataset_file=open(save_path, "wb")
    import pickle
    pickle.dump(dataset, dataset_file)
    dataset_file.close()
    exit()