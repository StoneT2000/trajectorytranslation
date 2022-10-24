import numpy as np
from mani_skill.env.open_cabinet_door_drawer import (
    OpenCabinetDrawerEnv_CabinetSelection, OpenCabinetDrawerMagicEnv)

from tr2.planner.base import HighLevelPlanner
from tr2.utils.sampling import resample_teacher_trajectory


class OpenDrawerPlanner(HighLevelPlanner):


    def __init__(self,
                 replan_threshold=1e-5,
                 # hand_threshold=0.02,
                 # block_diff_threshold=0.02,


                 **kwargs):
        super().__init__(**kwargs)
        self.replan_threshold = replan_threshold
        # self.hand_threshold = hand_threshold
        # self.block_diff_threshold = block_diff_threshold

    def need_replan(self, curr_state, student_obs, teacher_trajectory, env):
        env: OpenCabinetDrawerEnv_CabinetSelection
        if len(student_obs['observation'].shape) == 2:
            dense_obs = student_obs['observation'][-1]
        else:
            dense_obs = student_obs['observation']
        # check opened enough, and check far away enough
        eval_flag_dict = env.compute_eval_flag_dict()
        target_opened = eval_flag_dict["open_enoughs"][env.target_index]

        if eval_flag_dict["open_enough"]:
            # just wait for cabinet static, dont replan again
            return False

        far_enough = False
        # import pdb;pdb.set_trace()
        dist_to_rest = np.linalg.norm(dense_obs[26:29] - teacher_trajectory[-1][:3])
        # large margin since its not super important to return exactly and our policies when trained for just 1k steps haven't learned to return to rest
        far_enough = dist_to_rest < 4e-1 
        match_end = (np.linalg.norm(dense_obs[26:29] - teacher_trajectory[:, :3], axis=1)).argmin() >= len(teacher_trajectory) - 2
        return target_opened #and far_enough and match_end
    def generate_magic_teacher_traj_for_env(self, obs, render):
        # is_grasped = obs['is_grasped']
        planning_env = obs['planning_env']
        magic_traj = collect_heuristic_teacher_traj(planning_env, render=render, obs_mode='custom', task='open')
        observations = magic_traj["observations"]
        observations = resample_teacher_trajectory(np.array(observations), max_dist=5e-2)
        return observations
    def act(self, obs):
        pass


def collect_heuristic_teacher_traj(env: OpenCabinetDrawerMagicEnv, render=False, mode='human', obs_mode='state', task='open', end_move_away_dist=0.0):
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
        viewer = env.render(mode)
        viewer.paused=True
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
    end_noise = np.zeros(3) #np.random.rand(3) * end_move_away_dist
    end_noise[0] = 0.1
    end_noise[0] = end_noise[0] * -1
    
    goal1 = np.array(init_xyz + [0.04, 0.04])
    goal1[2] = xyz[2].copy()

    goal2 = xyz.copy() + [0.04, 0.04]
    goal3 = goal2.copy()
    goal3[0] -= env.target_qpos * 1.2


    if task == 'open':
        end_xyz = init_xyz#goal3[:3] + end_noise
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