import argparse

import cv2
import gym
import mplib
import numpy as np
import open3d as o3d
import sapien.core as sapien
from tqdm import tqdm

from mani_skill2 import ASSET_DIR
from mani_skill2.envs.assembly.plug_charger import (
    PlugChargerFixedXmate3RobotiqEnv,
    PlugChargerPandaEnv,
)
from mani_skill2.utils.misc import print_dict
from mani_skill2.utils.wrappers import ManiSkillActionWrapper


def convert_traj_to_action(traj_qpos, gripper_pos: float) -> np.ndarray:
    return np.concatenate(
        (
            traj_qpos,
            np.ones(1) * gripper_pos,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, required=True, choices=["xmate3", "panda"])
    args = parser.parse_args()
    return args


def main():
    np.set_printoptions(suppress=True, precision=8)

    args = parse_args()

    if args.robot == "xmate3":
        env: PlugChargerFixedXmate3RobotiqEnv = gym.make(
            "PlugChargerFixedXmate3Robotiq-v0",
            obs_mode="state_dict",
            reward_mode="dense",
        )
    else:
        env: PlugChargerPandaEnv = gym.make(
            "PlugChargerPanda-v0", obs_mode="state_dict", reward_mode="dense"
        )

    env = ManiSkillActionWrapper(env, "pd_joint_pos")
    env.reset()

    print("Observation space", env.observation_space)
    print("Action space", env.action_space)
    print("Control mode", env.control_mode)
    print("Reward mode", env.reward_mode)

    viewer = env.render()
    control_time_step = env.control_time_step

    H_robot_base = env.agent._robot.get_root_pose().to_transformation_matrix()

    def world_mat_to_robot(pose_mat: np.ndarray) -> np.ndarray:
        return np.linalg.inv(H_robot_base) @ pose_mat

    obs = env.get_obs()
    print("obs: ", print_dict(obs))

    print("Press [c] to start")
    while True:
        if viewer.window.key_down("c"):
            break
        env.render()

    # set up motion planning
    link_names = [link.get_name() for link in env.agent._robot.get_links()]
    joint_names = [joint.get_name() for joint in env.agent._robot.get_active_joints()]
    if args.robot == "xmate3":
        planner = mplib.Planner(
            urdf=str(ASSET_DIR / "descriptions/fixed_xmate3_robotiq.urdf"),
            srdf=str(ASSET_DIR / "descriptions/fixed_xmate3_robotiq.srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="grasp_convenient_link",
            joint_vel_limits=np.ones(7),
            joint_acc_limits=np.ones(7),
        )
        OPEN_GRIPPER_POS = 0.045
        CLOSE_GRIPPER_POS = 0.068
    else:
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

    # compute grasp pose
    forward = np.array([0.0, 0.0, -1.0])
    flat = np.array([0.0, -1.0, 0.0])

    grasp_mat = env.agent.build_grasp_pose(
        forward, flat, obs["env_state"]["charger_pos"]
    ).to_transformation_matrix()

    midpoint_mat = grasp_mat.copy()
    midpoint_mat[2, 3] += 0.05
    midpoint_pose = sapien.Pose.from_transformation_matrix(
        world_mat_to_robot(midpoint_mat)
    )
    midpoint_pose = list(midpoint_pose.p) + list(midpoint_pose.q)

    plan = planner.plan(
        midpoint_pose, env.agent._robot.get_qpos(), time_step=control_time_step
    )
    trajs = plan["position"]
    for i in tqdm(range(len(trajs))):
        env.render()
        action = convert_traj_to_action(trajs[i], gripper_pos=OPEN_GRIPPER_POS)
        obs, rew, done, info = env.step(action)

    print("Press [e] to continue")
    while True:
        if viewer.window.key_down("e"):
            break
        env.render()

    # reach charger
    grasp_pose = sapien.Pose.from_transformation_matrix(world_mat_to_robot(grasp_mat))
    grasp_pose = list(grasp_pose.p) + list(grasp_pose.q)

    plan = planner.plan_screw(
        grasp_pose, env.agent._robot.get_qpos(), time_step=control_time_step
    )
    trajs = plan["position"]
    for i in tqdm(range(len(trajs))):
        env.render()
        action = convert_traj_to_action(trajs[i], gripper_pos=OPEN_GRIPPER_POS)
        obs, rew, done, info = env.step(action)

    print("Press [c] to continue")
    while True:
        if viewer.window.key_down("c"):
            break
        env.render()

    # close gripper
    for i in tqdm(range(30)):
        env.render()
        action = convert_traj_to_action(trajs[-1], gripper_pos=CLOSE_GRIPPER_POS)
        obs, rew, done, info = env.step(action)
        # print(
        #     f'{i:04d}: charger_pos: { obs["env_state"]["charger_pos"]}'
        # )

    print("Press [e] to continue")
    while True:
        if viewer.window.key_down("e"):
            break
        env.render()

    # lift to goal
    grasp_mat = env.agent.build_grasp_pose(
        forward, flat, obs["env_state"]["goal_pos"]
    ).to_transformation_matrix()
    goal_mat = grasp_mat.copy()
    goal_mat[0, 3] -= 0.05
    if args.robot == "xmate3":
        goal_mat[2, 3] += 0.001
    goal_pose = sapien.Pose.from_transformation_matrix(world_mat_to_robot(goal_mat))
    goal_pose = list(goal_pose.p) + list(goal_pose.q)
    plan = planner.plan(
        goal_pose, env.agent._robot.get_qpos(), time_step=control_time_step
    )
    trajs = plan["position"]
    for i in range(len(trajs)):
        env.render()
        action = convert_traj_to_action(trajs[i], gripper_pos=CLOSE_GRIPPER_POS)
        obs, rew, done, info = env.step(action)
        tcp_wrench = env.agent.get_tcp_wrench()
        print(
            f"{i:04d}: obs: {print_dict(obs)} reward: {rew}, done: {done}, info: {info}, tcp_wrench: {tcp_wrench}"
        )
    print("Press [c] to continue")
    while True:
        if viewer.window.key_down("c"):
            break
        env.render()
    # plug
    goal_mat = grasp_mat.copy()
    # goal_mat[0, 3] += 0.005
    if args.robot == "xmate3":
        goal_mat[2, 3] += 0.001
    goal_pose = sapien.Pose.from_transformation_matrix(world_mat_to_robot(goal_mat))
    goal_pose = list(goal_pose.p) + list(goal_pose.q)
    plan = planner.plan_screw(
        goal_pose, env.agent._robot.get_qpos(), time_step=control_time_step
    )
    trajs = plan["position"]
    for i in range(len(trajs)):
        env.render()
        action = convert_traj_to_action(trajs[i], gripper_pos=CLOSE_GRIPPER_POS)
        obs, rew, done, info = env.step(action)
        tcp_wrench = env.agent.get_tcp_wrench()
        print(
            f"{i:04d}: obs: {print_dict(obs)} reward: {rew}, done: {done}, info: {info}, tcp_wrench: {tcp_wrench}"
        )

    print("Press [e] to continue")
    while True:
        if viewer.window.key_down("e"):
            break
        env.render()

    # open gripper
    for i in tqdm(range(30)):
        env.render()
        action = convert_traj_to_action(trajs[-1], gripper_pos=OPEN_GRIPPER_POS)
        obs, rew, done, info = env.step(action)

    print("Press [c] to finish")
    while True:
        if viewer.window.key_down("c"):
            break
        env.render()


if __name__ == "__main__":
    main()
