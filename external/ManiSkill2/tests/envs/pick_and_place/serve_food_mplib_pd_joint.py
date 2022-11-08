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
from mani_skill2.envs.pick_and_place.serve_food import (
    ServeFoodFixedXmate3RobotiqEnv,
    ServeFoodPandaEnv,
)
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, required=True, choices=["xmate3", "panda"])
    args = parser.parse_args()
    return args


def main():
    np.set_printoptions(suppress=True, precision=8)

    args = parse_args()

    if args.robot == "xmate3":
        env: ServeFoodFixedXmate3RobotiqEnv = gym.make(
            "ServeFoodFixedXmate3Robotiq-v0",
            container_model_ids=[
                "bowl",
                # "plate",
            ],
            food_model_ids=[
                "014_lemon",
            ],
            obs_mode="state_dict",
            reward_mode="dense",
        )
    else:
        env: ServeFoodPandaEnv = gym.make(
            "ServeFoodPanda-v0",
            container_model_ids=[
                "bowl",
                "plate",
            ],
            food_model_ids=[
                "014_lemon",
            ],
            obs_mode="state_dict",
            reward_mode="dense",
        )
    env = ManiSkillActionWrapper(env, "pd_joint_pos")
    env.reset()

    print("Action space", env.action_space)
    print("Control mode", env.control_mode)
    print("Reward mode", env.reward_mode)

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
    # scene_pcd = o3d.geometry.PointCloud()
    # scene_pcd.points = o3d.utility.Vector3dVector(scene_points)
    # o3d.io.write_point_cloud('scene_pcd.pcd', scene_pcd)

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

    planner.update_point_cloud(scene_points)

    # compute grasp pose
    if obs["extra"]["container_idx"] == 0:
        # plate
        forward = np.array([1.0, 0.0, 0.0])
        flat = np.array([0.0, 0.0, -1.0])
        grasp_point = np.array([-0.11, 0.0, 0.0])
    elif obs["extra"]["container_idx"] == 1:
        # bowl
        forward = np.array([0.6, 0.0, -0.8])
        flat = np.array([0.8, 0.0, 0.6])
        grasp_point = np.array([-0.08, 0.0, 0.019])
    else:
        print("Unkown container idx")
        exit()

    grasp_mat = env.agent.build_grasp_pose(
        forward,
        flat,
        grasp_point + obs["extra"]["container_pos"],
    ).to_transformation_matrix()
    mid_mat = grasp_mat.copy()
    mid_mat[:3, 3] -= forward * 0.15

    mid_pose = mat_to_pose(mid_mat)
    plan = planner.plan(
        mid_pose,
        env.agent._robot.get_qpos(),
        time_step=control_time_step,
        # use_point_cloud=True,
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

    grasp_pose = mat_to_pose(grasp_mat)
    plan = planner.plan_screw(
        grasp_pose,
        env.agent._robot.get_qpos(),
        time_step=control_time_step,
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
    print("Press [c] to continue")
    while True:
        if viewer.window.key_down("c"):
            break
        env.render()

    # lift
    mid_mat = grasp_mat.copy()
    mid_mat[2, 3] += 0.15
    mid_pose = mat_to_pose(mid_mat)
    plan = planner.plan_screw(
        mid_pose, env.agent._robot.get_qpos(), time_step=control_time_step
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

    place_mat = env.agent.build_grasp_pose(
        forward, flat, grasp_point + obs["extra"]["goal_pos"]
    ).to_transformation_matrix()
    mid_mat = place_mat.copy()
    mid_mat[2, 3] += 0.15
    mid_goal_pose = mat_to_pose(mid_mat)

    success_plan = False
    mid_pose2 = np.array(mid_pose[:3]) + np.array(mid_goal_pose[:3])
    mid_pose2 /= 2.0
    for i in tqdm(range(10000)):
        mid_pose_tmp = mid_pose2 + np.concatenate(
            [np.random.randn(2) * 0.15, np.random.rand(1) * 0.15]
        )
        mid_pose_tmp = list(mid_pose_tmp) + mid_goal_pose[3:]

        plan = planner.plan_screw(
            mid_pose_tmp, env.agent._robot.get_qpos(), time_step=control_time_step
        )
        if "position" not in plan:
            continue
        plan2 = planner.plan_screw(
            mid_goal_pose,
            np.concatenate([plan["position"][-1], env.agent._robot.get_qpos()[-2:]]),
            time_step=control_time_step,
        )
        if "position" in plan2:
            success_plan = True
            break

    if not success_plan:
        print("CAN NOT PLAN!")
        exit()

    trajs = plan["position"]
    for i in tqdm(range(len(trajs))):
        env.render()
        action = convert_traj_to_action(trajs[i], gripper_pos=CLOSE_GRIPPER_POS)
        obs, rew, done, info = env.step(action)

    trajs = plan2["position"]
    for i in tqdm(range(len(trajs))):
        env.render()
        action = convert_traj_to_action(trajs[i], gripper_pos=CLOSE_GRIPPER_POS)
        obs, rew, done, info = env.step(action)

    print("Press [c] to continue")
    while True:
        if viewer.window.key_down("c"):
            break
        env.render()

    place_pose = mat_to_pose(place_mat)
    plan = planner.plan_screw(
        place_pose, env.agent._robot.get_qpos(), time_step=control_time_step
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


if __name__ == "__main__":
    main()
