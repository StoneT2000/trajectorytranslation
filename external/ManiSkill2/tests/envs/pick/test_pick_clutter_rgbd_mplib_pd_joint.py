import argparse

import cv2
import gym
import mplib
import numpy as np
import open3d as o3d
import sapien.core as sapien
from tqdm import tqdm

from mani_skill2 import ASSET_DIR
from mani_skill2.envs.pick_and_place.base_env import PandaEnv
from mani_skill2.utils.cv_utils import depth2pts_np
from mani_skill2.utils.geometry import (
    get_oriented_bounding_box_for_2d_points,
    transform_points,
)
from mani_skill2.utils.misc import print_dict
from mani_skill2.utils.wrappers import ManiSkillActionWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    np.set_printoptions(suppress=True, precision=4)

    args = parse_args()

    def convert_traj_to_action(traj_qpos, gripper_pos: float) -> np.ndarray:
        return np.concatenate(
            (
                traj_qpos,
                np.ones(1) * gripper_pos,
            )
        )

    if args.robot == "panda":
        env: PandaEnv = gym.make(
            "PickClutterPanda-v0",
            json_path=str(ASSET_DIR / "ocrtoc/episodes/train_ycb_box_v0_1000.json.gz"),
            obs_mode="rgbd",
            reward_mode="dense",
        )
    elif args.robot == "xmate3":
        env: PandaEnv = gym.make(
            "PickClutterFixedXmate3Robotiq-v0",
            json_path=str(ASSET_DIR / "ocrtoc/episodes/train_ycb_box_v0_1000.json.gz"),
            obs_mode="rgbd",
            reward_mode="dense",
        )
    else:
        print("Unknown robot")
        exit()
    env = ManiSkillActionWrapper(env, control_mode="pd_joint_pos")
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

    hand_pts = depth2pts_np(
        obs["hand_camera"]["depth"],
        obs["hand_camera"]["camera_intrinsic"],
        obs["hand_camera"]["camera_extrinsic_base_frame"],
    )
    hand_pts = transform_points(H_robot_base, hand_pts)

    hand_obj_mask = obs["hand_camera"]["obj_mask"]
    cv2.imwrite("hand_obj_mask.png", (hand_obj_mask * 255).astype(np.uint8))

    third_pts = depth2pts_np(
        obs["third_view"]["depth"],
        obs["third_view"]["camera_intrinsic"],
        obs["third_view"]["camera_extrinsic_world_frame"],
    )
    third_obj_mask = obs["third_view"]["obj_mask"]

    cv2.imwrite("third_obj_mask.png", (third_obj_mask * 255).astype(np.uint8))

    np.savetxt("hand_pts.xyz", hand_pts)
    np.savetxt("third_pts.xyz", third_pts)

    third_pts_obj = third_pts[third_obj_mask.reshape(-1)]
    np.savetxt("third_pts_obj.xyz", third_pts_obj)

    hand_pts_obj = hand_pts[hand_obj_mask.reshape(-1)]
    np.savetxt("hand_pts_obj.xyz", hand_pts_obj)

    hand_rgb = (obs["hand_camera"]["rgb"] * 255).astype(np.uint8)
    cv2.imwrite("hand_rgb.png", hand_rgb[..., ::-1])

    third_rgb = (obs["third_view"]["rgb"] * 255).astype(np.uint8)
    cv2.imwrite("third_rgb.png", third_rgb[..., ::-1])
    #
    print("Press [c] to continue")
    while True:
        if viewer.window.key_down("c"):
            break
        env.render()

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
        OPEN_GRIPPER_POS = 0.0
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

    planner.update_point_cloud(transform_points(np.linalg.inv(H_robot_base), third_pts))

    obj_pts = np.concatenate([third_pts_obj, hand_pts_obj])
    max_z = obj_pts[:, 2].max()
    finger_length = 0.002

    obb_dict = get_oriented_bounding_box_for_2d_points(obj_pts[:, :2], resolution=0.001)
    center = obb_dict["center"]
    half_size = obb_dict["half_size"]
    axes = obb_dict["axes"]
    corners = obb_dict["corners"]
    # visualize in open3d
    VISUALIZE = False
    if VISUALIZE:
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(obj_pts)
        vect = np.concatenate([axes, np.zeros((1, 2))], axis=0)
        vect = np.concatenate([vect, np.array([0, 0, 1]).reshape(3, 1)], axis=1)
        trans = np.eye(4)
        trans[:3, 3] = center
        trans[:3, :3] = vect
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        frame.transform(trans)
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        corners = np.concatenate([corners, np.ones((4, 1)) * max_z], axis=1)
        corners_pcd = o3d.geometry.PointCloud()
        corners_pcd.points = o3d.utility.Vector3dVector(corners)
        colors = np.array([1, 0, 0]).reshape(1, 3).repeat(4, 0)
        corners_pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([obj_pcd, frame, world_frame, corners_pcd])

    axes_3d = np.concatenate([axes, np.zeros((1, 2))], axis=0)
    center = np.concatenate([center, np.array([max_z - finger_length])])

    forward = np.array([0.0, 0.0, -1.0])
    if half_size[0] < half_size[1]:
        flat = axes_3d[:, 0]
    else:
        flat = axes_3d[:, 1]

    grasp_mat = env.agent.build_grasp_pose(
        forward, flat, center
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

    # reach object
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

    print("Press [e] to continue")
    while True:
        if viewer.window.key_down("e"):
            break
        env.render()

    # lift to goal
    goal_mat = grasp_mat.copy()
    goal_mat[:3, 3] = obs["goal_pos"]
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

    print("Press [e] to finish")
    while True:
        if viewer.window.key_down("e"):
            break
        env.render()


if __name__ == "__main__":
    main()
