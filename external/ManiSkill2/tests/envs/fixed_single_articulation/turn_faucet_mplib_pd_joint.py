import argparse

import cv2
import gym
import mplib
import numpy as np
import open3d as o3d
import sapien.core as sapien
import transforms3d as t3d
from tqdm import tqdm

from mani_skill2 import ASSET_DIR
from mani_skill2.envs.experimental.turn_faucet import TurnFaucetPandaEnv
from mani_skill2.utils.geometry import transform_points
from mani_skill2.utils.misc import print_dict
from mani_skill2.utils.wrappers import ManiSkillActionWrapper


def convert_traj_to_action(traj_qpos, gripper_pos: float) -> np.ndarray:
    return np.concatenate(
        (
            traj_qpos,
            np.ones(1) * gripper_pos,
        )
    )


def main():
    np.set_printoptions(suppress=True, precision=8)

    env: TurnFaucetPandaEnv = gym.make(
        "TurnFaucetPanda-v0",
        model_ids=[
            # "693",  # wrong contact
            # "1011",
            # "1370",
            # "1667",
            # "1935",
            # "2054",  # don't know reason why it does not move...
            "908",
        ],
        obs_mode="state_dict",
        reward_mode="dense",
    )

    env = ManiSkillActionWrapper(env, "pd_joint_pos")
    env.reset()

    print("Action space", env.action_space)
    print("Control mode", env.control_mode)
    print("Reward mode", env.reward_mode)

    print("link info: ")
    print(
        "name: ",
        env.target_link.name,
        "; scale: ",
        env.model_scale,
        "; mass: ",
        env.target_link.get_mass(),
        "; inertia: ",
        env.target_link.get_inertia(),
    )

    viewer = env.render()
    control_time_step = env.control_time_step

    H_robot_base = env.agent._robot.get_root_pose().to_transformation_matrix()

    def world_mat_to_robot(pose_mat: np.ndarray) -> np.ndarray:
        return np.linalg.inv(H_robot_base) @ pose_mat

    def mat_to_pose(pose_mat: np.ndarray) -> list:
        pose = sapien.Pose.from_transformation_matrix(world_mat_to_robot(pose_mat))
        return list(pose.p) + list(pose.q)

    obs = env.get_obs()
    print("obs: ", print_dict(obs))

    scene_points = env.gen_scene_pcd(int(1e5))
    scene_points = transform_points(np.linalg.inv(H_robot_base), scene_points)

    curr_link_pcd = obs["extra"]["curr_link_pcd"]
    target_link_pcd = obs["extra"]["target_link_pcd"]
    curr_angle = obs["extra"]["curr_angle"]
    target_angle = obs["extra"]["target_angle"]
    joint_position = obs["extra"]["joint_position"]
    joint_axis = obs["extra"]["joint_axis"]

    rot_angle = target_angle - curr_angle
    R = t3d.axangles.axangle2mat(joint_axis, rot_angle)
    target_link_pcd_computed = (
        R @ (curr_link_pcd.T - joint_position[:, None]) + joint_position[:, None]
    ).T

    # curr_pcd = o3d.geometry.PointCloud()
    # curr_pcd.points = o3d.utility.Vector3dVector(curr_link_pcd)
    # o3d.io.write_point_cloud('curr_pcd.pcd', curr_pcd)
    #
    # target_pcd = o3d.geometry.PointCloud()
    # target_pcd.points = o3d.utility.Vector3dVector(target_link_pcd)
    # o3d.io.write_point_cloud('target_pcd.pcd', target_pcd)
    #
    # target_pcd_computed = o3d.geometry.PointCloud()
    # target_pcd_computed.points = o3d.utility.Vector3dVector(target_link_pcd_computed)
    # o3d.io.write_point_cloud('target_link_pcd_computed.pcd', target_pcd_computed)

    print("Press [c] to start")
    while True:
        if viewer.window.key_down("c"):
            break
        env.render()

    # set up motion planning
    link_names = [link.get_name() for link in env.agent._robot.get_links()]
    joint_names = [joint.get_name() for joint in env.agent._robot.get_active_joints()]
    planner = mplib.Planner(
        urdf=str(ASSET_DIR / "descriptions/panda_v1.urdf"),
        srdf=str(ASSET_DIR / "descriptions/panda_v1.srdf"),
        user_link_names=link_names,
        user_joint_names=joint_names,
        move_group="grasp_site",
        joint_vel_limits=np.ones(7),
        joint_acc_limits=np.ones(7),
    )
    OPEN_GRIPPER_POS = 0.04
    CLOSE_GRIPPER_POS = 0.0
    planner.update_point_cloud(scene_points)

    # find farthest point to the joint axis
    joint_axis_dist = curr_link_pcd - joint_position[None, :]
    joint_axis_dist = (
        joint_axis_dist
        - (joint_axis_dist * joint_axis[None, :]).sum(1, keepdims=True)
        * joint_axis[None, :]
    )
    joint_axis_dist = np.linalg.norm(joint_axis_dist, axis=1)

    max_idx = np.argmax(joint_axis_dist)
    farthest_point = curr_link_pcd[max_idx]
    push_direction = np.cross(joint_axis, farthest_point - joint_position)
    push_direction /= np.linalg.norm(push_direction)
    if target_angle < curr_angle:
        push_direction *= -1

    PUSH_OFFSET = 0.07
    AXIS_OFFSET = 0.03
    RADIUS_OFFSET = 0.03
    NUM_STEP = 4

    # compute gripper pose
    push_point = (
        farthest_point
        - PUSH_OFFSET * push_direction
        - AXIS_OFFSET * joint_axis
        - RADIUS_OFFSET * np.cross(push_direction, joint_axis)
    )
    push_mat = env.agent.build_grasp_pose(
        -joint_axis, push_direction, push_point
    ).to_transformation_matrix()

    mid_mat = push_mat.copy()
    mid_mat[:3, 3] += joint_axis * AXIS_OFFSET * 2
    mid_pose = mat_to_pose(mid_mat)

    plan = planner.plan(
        mid_pose,
        env.agent._robot.get_qpos(),
        time_step=control_time_step,
        use_point_cloud=True,
    )
    trajs = plan["position"]
    for i in tqdm(range(len(trajs))):
        env.render()
        action = convert_traj_to_action(trajs[i], gripper_pos=CLOSE_GRIPPER_POS)
        obs, rew, done, info = env.step(action)

    print("Press [c] to continue")
    while True:
        if viewer.window.key_down("c"):
            break
        env.render()

    push_pose = mat_to_pose(push_mat)

    plan = planner.plan(
        push_pose, env.agent._robot.get_qpos(), time_step=control_time_step
    )
    trajs = plan["position"]
    for i in tqdm(range(len(trajs))):
        env.render()
        action = convert_traj_to_action(trajs[i], gripper_pos=CLOSE_GRIPPER_POS)
        obs, rew, done, info = env.step(action)

    print("Press [c] to continue")
    while True:
        if viewer.window.key_down("c"):
            break
        env.render()

    for i in range(NUM_STEP):
        R_step = t3d.axangles.axangle2mat(joint_axis, rot_angle / NUM_STEP * (i + 1))
        target_farthest_point = (
            R_step @ (farthest_point - joint_position) + joint_position
        )
        target_push_point = R_step @ (push_point - joint_position) + joint_position
        target_push_direction = R_step @ push_direction
        target_push_mat = env.agent.build_grasp_pose(
            -joint_axis, target_push_direction, target_push_point
        ).to_transformation_matrix()
        target_push_pose = mat_to_pose(target_push_mat)
        plan = planner.plan_screw(
            target_push_pose, env.agent._robot.get_qpos(), time_step=control_time_step
        )
        trajs = plan["position"]
        for i in tqdm(range(len(trajs))):
            env.render()
            action = convert_traj_to_action(trajs[i], gripper_pos=CLOSE_GRIPPER_POS)
            obs, rew, done, info = env.step(action)
            tcp_wrench = env.agent.get_tcp_wrench()
            print(f"{i:04d}: reward: {rew}, done: {done}, tcp_wrench: {tcp_wrench}")

    print("Press [c] to continue")
    while True:
        if viewer.window.key_down("c"):
            break
        env.render()


if __name__ == "__main__":
    main()
