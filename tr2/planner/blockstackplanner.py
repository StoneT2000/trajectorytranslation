import numpy as np

from tr2.envs.blockstacking import (BlockStackFloatPandaEnv,
                                                 BlockStackMagicPandaEnv)
from tr2.planner.base import HighLevelPlanner
from tr2.utils.animate import animate
from tr2.utils.sampling import resample_teacher_trajectory


class BlockStackPlanner(HighLevelPlanner):


    def __init__(self,
                 replan_threshold=1e-5,
                 # hand_threshold=0.02,
                 # block_diff_threshold=0.02,
                release_height=0.05,

                 **kwargs):
        super().__init__(**kwargs)
        self.replan_threshold = replan_threshold
        self.release_height = release_height
        # self.hand_threshold = hand_threshold
        # self.block_diff_threshold = block_diff_threshold

    def need_replan(self, curr_state, student_obs, teacher_trajectory, env):
        if isinstance(curr_state, dict):
            # from traj env? so dict?
            # import pdb;pdb.set_trace()
            if len(student_obs['observation'].shape) == 2:
                dense_obs = student_obs['observation'][-1]
            else:
                dense_obs = student_obs['observation']
            teacher_student_coord_diff = np.linalg.norm(teacher_trajectory[:, :3] - dense_obs[25:28],axis=1)
            teacher_student_world_diff = np.linalg.norm(teacher_trajectory[:, -7:-4] - dense_obs[18:25][:3],axis=1)
            max_world_diff = 0.05
            if env.real_pose_est:
                max_world_diff = 0.08
            if teacher_student_coord_diff[-1] < 0.05 and teacher_student_world_diff[-1] < max_world_diff:
                env.allow_next_stage(True)

                return True
            env.allow_next_stage(False)
            return False
        else:
            # qpos-9 qvel-9 block-7 hand.p=3 hand.q=4 block.vel-3 block.avel-3
            student_hand_coord = curr_state[25:28]
            student_block_coord = curr_state[18:21] # only pos no quat
            teacher_hand_coord = teacher_trajectory[:, :3]
            teacher_block_coord = teacher_trajectory[:, 10:13]
            teacher_student_hand_diff = np.linalg.norm(
                teacher_hand_coord - student_hand_coord, axis=1
            )
            replan=True
            return replan
    def generate_magic_teacher_traj_for_env(self, obs, render=False, save_video=False, plan_id=0):
        is_grasped = obs['is_grasped']
        planning_env = obs['planning_env']
        change_in_height=0.2
        change_in_goal_height=0.2
        release_dist = 0.025#self.release_height
        magic_traj, imgs = collect_magic_teacher_traj(planning_env, render=render,save_video=save_video,is_grasped=is_grasped,change_in_goal_height=change_in_goal_height,change_in_height=change_in_height,release_dist=release_dist)
        if save_video:
            animate(imgs, f"plan_{plan_id}.mp4", fps=150)
        
        return magic_traj['observations']
    def act(self, obs):
        pass
def collect_magic_teacher_traj(env: BlockStackMagicPandaEnv, is_grasped=False, render=False, save_video=False, mode='human', change_in_height=0, change_in_goal_height=0,dreamed=True, release_dist=0.0,init_offset=np.zeros(3)):
    # assert env.connected == is_g
    
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
    imgs = []
    def goto(xyz, gripper_action, attn=0):
        assert gripper_action is None
        nonlocal grasping, curr_xyz, robot
        i_step = 0
        target=np.array(xyz)
        observations=[]  # during this goto()
        attns = []

        if dreamed:
            dist_to_goal = np.linalg.norm(curr_xyz - xyz)
            block = env.blocks[attn]
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
                pm_pose = env.grasp_site.pose
                env.pointmass.set_pose(pm_pose)
                if grasping:
                    
                    pose.set_p(pose.p + deltas)
                    block.set_pose(pose)
                dist_to_goal = np.linalg.norm(new_xyz - xyz)
                curr_xyz = new_xyz
                attns.append(attn)
                if save_video:
                    img = env.render("rgb_array")
                    imgs.append(img)
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
        env.magic_release()
        block_pos=env.blocks[i].pose.p
        goal_pos=env.goal_coords[i]
        dist_to_goal = np.linalg.norm(goal_pos - block_pos)
        # print(f"block-{i}: {dist_to_goal}")
        fac = 1.25
        if env.real_pose_est:
            fac = 1.8
        if dist_to_goal < env.block_half_size*fac:
            continue
        
        above_pos=[block_pos[0], block_pos[1], 0.25+change_in_height]
        grasp_pos=[block_pos[0], block_pos[1], 0.143]
        itmd = [init_xyz[0], init_xyz[1], 0.35]
        if not is_grasped:
            if not goto(xyz=itmd, gripper_action=None, attn=i): return None
            if not goto(xyz=above_pos, gripper_action=None, attn=i): return None
            if not goto(xyz=grasp_pos, gripper_action=None, attn=i): return None
        if not dreamed:
            env.magic_grasp(block)
        grasping=True
        if not goto(xyz=above_pos, gripper_action=None, attn=i): return None

        
        goal_above_pos=[goal_pos[0], goal_pos[1], goal_pos[2] + 0.096 + 0.05+ change_in_goal_height + release_dist]
        release_pos=[goal_pos[0], goal_pos[1], goal_pos[2] + 0.096 + release_dist]

        if not goto(xyz=goal_above_pos, gripper_action=None, attn=i): return None
        if not goto(xyz=release_pos, gripper_action=None, attn=i): return None
        if not dreamed:
            env.magic_release()
        grasping=False
        if not goto(xyz=goal_above_pos, gripper_action=None, attn=i): return None

        itmd = [init_xyz[0], init_xyz[1], 0.35]
        if not goto(xyz=itmd, gripper_action=None, attn=i): return None

        # back to init
        if not goto(xyz = init_xyz, gripper_action=None, attn=i): return None
        break
        # last observation and attention:
    obs_dict=env.get_obs()
    dense_obs=np.zeros(10)
    dense_obs[:3]=obs_dict['agent']['qpos'][:3]
    dense_obs[3:]=obs_dict['extra']['obs_blocks']['absolute'][i * 7:i * 7 + 7]
    magic_traj['observations'].append(dense_obs)

    magic_traj['attns'] = np.array(magic_traj['attns'], dtype=int)
    magic_traj['observations'] = np.array(magic_traj['observations'], dtype=np.float32)
    magic_traj['observations'] = resample_teacher_trajectory(magic_traj['observations'], 4e-2)
    # assert len(float_traj['attentions']) == len(float_traj['observations'])
    if save_video:
        return magic_traj, imgs
    if render and mode == 'rgb_array':
        return magic_traj, vids
    return magic_traj, None


def collect_magic_teacher_traj_already_grasped(env: BlockStackMagicPandaEnv, render=False, mode='human'):
    assert env.connected == True

    assert env.goal_coords is not None
    assert env.obs_mode == 'state_dict'
    magic_traj=dict(
        observations=[],
    )
    vids=[]
    if render and mode == 'human':
        env.render()
        env.get_viewer().paused=True
    i_step=0

    def goto(xyz, gripper_action, attn=0):
        assert gripper_action is None
        nonlocal i_step, block
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
            dense_obs=np.zeros(17)
            dense_obs[:5]=obs_dict['agent']['qpos']
            dense_obs[5:10]=obs_dict['agent']['qvel']
            dense_obs[10:]=obs_dict['extra']['obs_blocks']['absolute'][attn * 7: attn * 7 + 7]
            observations.append(dense_obs)

            assert action[-1] == +1
            env.step(action)

            if render:
                if mode == 'rgb_array':
                    vids.append(env.render(mode))
                else:
                    env.render(mode)
            i_step+=1
        magic_traj['observations']+=observations


    for i in range(len(env.blocks)):
        block=env.blocks[i]
        block_pos=env.blocks[i].pose.p
        # above_pos=[block_pos[0], block_pos[1], 0.25]
        # grasp_pos=[block_pos[0], block_pos[1], 0.04 + 0.112]
        # above_pos=[block_pos[0], block_pos[1], 0.25]
        # grasp_pos=[block_pos[0], block_pos[1], 0.113]
        #
        # goto(xyz=above_pos, gripper_action=None, attn=i)
        # goto(xyz=grasp_pos, gripper_action=None, attn=i)
        # env.magic_grasp(block)
        # goto(xyz=above_pos, gripper_action=None, attn=i)

        goal_pos=env.goal_coords[i]
        # goal_above_pos=[goal_pos[0], goal_pos[1], 0.25]
        # release_pos=[goal_pos[0], goal_pos[1], goal_pos[2] + 0.132]
        goal_above_pos=[goal_pos[0], goal_pos[1], 0.25]
        release_pos=[goal_pos[0], goal_pos[1], goal_pos[2] + 0.096]

        goto(xyz=goal_above_pos, gripper_action=None, attn=i)
        goto(xyz=release_pos, gripper_action=None, attn=i)
        env.magic_release()
        goto(xyz=goal_above_pos, gripper_action=None, attn=i)
        # last observation and attention:
    obs_dict=env.get_obs()
    dense_obs=np.zeros(17)
    dense_obs[:5]=obs_dict['agent']['qpos']
    dense_obs[5:10]=obs_dict['agent']['qvel']
    dense_obs[10:]=obs_dict['extra']['obs_blocks']['absolute'][i * 7:i * 7 + 7]
    # float_traj['attentions'].append(i) ## here 'i' is the last block
    magic_traj['observations'].append(dense_obs)

    # assert len(float_traj['attentions']) == len(float_traj['observations'])
    if render and mode == 'rgb_array':
        return vids

    if not render:
        return magic_traj
    return None



