import argparse
import time

import gym
import mplib
import numpy as np
import sapien.core as sapien
import transforms3d as t3d
from tqdm import tqdm

from mani_skill2 import ASSET_DIR
from mani_skill2.envs.fixed_single_articulation.open_cabinet_door import OpenCabinetDoor
from mani_skill2.envs.fixed_single_articulation.open_cabinet_drawer import (
    OpenCabinetDrawer,
)
from mani_skill2.utils.contrib import apply_pose_to_points, o3d_to_trimesh
from mani_skill2.utils.geometry import transform_points
from mani_skill2.utils.misc import print_dict
from mani_skill2.utils.o3d_utils import merge_mesh, np2mesh
from mani_skill2.utils.sapien_utils import get_entity_by_name
from mani_skill2.utils.wrappers import ManiSkillActionWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["door", "drawer"], required=True)
    parser.add_argument(
        "--obs-mode", choices=["state", "state_dict", "rgbd"], default="state_dict"
    )
    parser.add_argument("--reward-mode", choices=["sparse", "dense"], default="dense")
    parser.add_argument(
        "--control-mode",
        choices=["pd_joint_pos", "imp_joint_pos", "imp_ee_pos"],
        required=True,
    )
    args = parser.parse_args()
    return args


def main():
    np.set_printoptions(suppress=True, precision=4)

    # for impedance control
    KP_REACH = 2000.0
    KD_REACH = 1.0
    KP_PULL = 1000.0
    KD_PULL = 1.0

    args = parse_args()

    if args.env == "drawer":
        env: OpenCabinetDrawer = gym.make(
            "FixedOpenCabinetDrawer-v0",
            articulation_config_path=ASSET_DIR
            / "partnet_mobility_configs/fixed_cabinet_drawer_filtered/1005-3-0.yml",
            obs_mode=args.obs_mode,
            reward_mode=args.reward_mode,
        )
    else:
        env: OpenCabinetDoor = gym.make(
            "FixedOpenCabinetDoor-v0",
            articulation_config_path=ASSET_DIR
            / "partnet_mobility_configs/fixed_cabinet_door_filtered/1000-0-2.yml",
            obs_mode=args.obs_mode,
            reward_mode=args.reward_mode,
        )
    env = ManiSkillActionWrapper(env, control_mode=args.control_mode)
    env.reset()

    print("Observation space", env.observation_space)
    print("Action space", env.action_space)
    print("Control mode", env.control_mode)
    print("Reward mode", env.reward_mode)

    z = env.articulation.get_root_pose().p[2]
    x = 0.33
    y = 0.14
    rz = 0.14
    env.articulation.set_root_pose(
        sapien.Pose([x, y, z], [np.sqrt(1 - rz ** 2), 0, 0, rz])
    )
    viewer = env.render()
    print("Press [e] to continue")
    while True:
        if viewer.window.key_down("e"):
            break
        env.render()

    robot_root_pose_mat = env.agent._robot.get_root_pose().to_transformation_matrix()
    start_link = get_entity_by_name(env.agent._robot.get_links(), "xmate3_base")
    end_link = get_entity_by_name(env.agent._robot.get_links(), "xmate3_link7")

    def convert_traj_to_action(
        traj_qpos, gripper_pos: float, kp: float, kd: float
    ) -> np.ndarray:

        if args.control_mode == "pd_joint_pos":
            return np.concatenate(
                (
                    traj_qpos,
                    np.ones(1) * gripper_pos,
                )
            )
        elif args.control_mode == "imp_joint_pos":
            return np.concatenate(
                (
                    traj_qpos,
                    np.ones_like(traj_qpos) * kp,
                    np.ones_like(traj_qpos) * kd,
                    np.ones(1) * gripper_pos,
                )
            )
        else:
            curr_qpos = env.agent._robot.get_qpos()
            env.agent._robot.set_qpos(np.concatenate([traj_qpos, np.zeros(2)]))
            rel_pose = start_link.get_pose().inv() * end_link.get_pose()
            rel_mat = rel_pose.to_transformation_matrix()
            axangle = t3d.axangles.mat2axangle(rel_mat[:3, :3])
            ac = np.concatenate([rel_mat[:3, 3], axangle[0] * axangle[1]])
            env.agent._robot.set_qpos(curr_qpos)

            return np.concatenate(
                (
                    ac,
                    np.ones(6) * kp,
                    np.ones(6) * kd,
                    np.ones(1) * gripper_pos,
                )
            )

    def world_mat_to_robot(pose_mat: np.ndarray) -> np.ndarray:
        return np.linalg.inv(robot_root_pose_mat) @ pose_mat

    control_time_step = env.control_time_step
    target_pose = env.target_link.get_pose() * env.handle_info["grasp"][0]
    print("target grasp pose: ", target_pose)

    link_names = [link.get_name() for link in env.agent._robot.get_links()]
    joint_names = [joint.get_name() for joint in env.agent._robot.get_active_joints()]

    planner = mplib.Planner(
        urdf=str(ASSET_DIR / "descriptions/fixed_xmate3_robotiq.urdf"),
        srdf=str(ASSET_DIR / "descriptions/fixed_xmate3_robotiq.srdf"),
        user_link_names=link_names,
        user_joint_names=joint_names,
        move_group="grasp_convenient_link",
        joint_vel_limits=np.ones(7),
        joint_acc_limits=np.ones(7),
    )

    # extract point cloud from cabinet for collision avoidance
    cabinet: sapien.Articulation = env.articulation
    meshes = []
    for link in cabinet.get_links():
        for visual_body in link.get_visual_bodies():
            for render_shape in visual_body.get_render_shapes():
                vertices = apply_pose_to_points(
                    render_shape.mesh.vertices * visual_body.scale,
                    link.get_pose() * visual_body.local_pose,
                )
                mesh = np2mesh(vertices, render_shape.mesh.indices.reshape(-1, 3))
                meshes.append(mesh)
    mesh = merge_mesh(meshes)
    mesh = o3d_to_trimesh(mesh)
    cabinet_pcd = mesh.sample(10000)
    cabinet_pcd = transform_points(
        np.linalg.inv(robot_root_pose_mat), cabinet_pcd
    )  # transform to robot base frame
    planner.update_point_cloud(cabinet_pcd)

    # plan 0, reach midpoint
    curr_mat = env.agent.grasp_site.get_pose().to_transformation_matrix()
    print("curr mat: ", curr_mat)
    midpoint_mat = target_pose.to_transformation_matrix()
    midpoint_mat[0, 3] -= 0.1
    print("midpoint mat: ", midpoint_mat)
    midpoint_pose = sapien.Pose.from_transformation_matrix(
        world_mat_to_robot(midpoint_mat)
    )
    midpoint_pose = list(midpoint_pose.p) + list(midpoint_pose.q)
    plan0 = planner.plan(
        midpoint_pose,
        env.agent._robot.get_qpos(),
        time_step=control_time_step,
        use_point_cloud=True,
    )
    trajs = plan0["position"]

    for i in tqdm(range(len(trajs))):
        env.render()
        action = convert_traj_to_action(trajs[i], 0, KP_REACH, KD_REACH)
        obs, rew, done, info = env.step(action)
    print("Press [c] to continue")
    while True:
        if viewer.window.key_down("c"):
            break
        env.render()

    # reach handle
    midpoint_mat = target_pose.to_transformation_matrix()
    midpoint_mat[0, 3] -= 0.03
    midpoint_pose = sapien.Pose.from_transformation_matrix(
        world_mat_to_robot(midpoint_mat)
    )
    midpoint_pose = list(midpoint_pose.p) + list(midpoint_pose.q)
    plan1 = planner.plan_screw(
        midpoint_pose,
        env.agent._robot.get_qpos(),
        time_step=control_time_step,
        use_point_cloud=True,
    )

    trajs = plan1["position"]
    for i in tqdm(range(len(trajs))):
        env.render()
        action = convert_traj_to_action(trajs[i], 0, KP_REACH, KD_REACH)
        obs, rew, done, info = env.step(action)

    print("Press [e] to continue")
    while True:
        if viewer.window.key_down("e"):
            break
        env.render()

    # close gripper
    for i in tqdm(range(30)):
        env.render()
        action = convert_traj_to_action(trajs[-1], 0.068, KP_PULL, KD_PULL)
        obs, rew, done, info = env.step(action)

    print("Press [c] to continue")
    while True:
        if viewer.window.key_down("c"):
            break
        env.render()

    # pull out
    midpoint_mat = target_pose.to_transformation_matrix()
    midpoint_mat[0, 3] -= 0.1
    # add some noise
    midpoint_mat[2, 3] += 0.01
    print("midpoint mat: ", midpoint_mat)
    midpoint_pose = sapien.Pose.from_transformation_matrix(
        world_mat_to_robot(midpoint_mat)
    )
    midpoint_pose = list(midpoint_pose.p) + list(midpoint_pose.q)
    plan2 = planner.plan_screw(
        midpoint_pose, env.agent._robot.get_qpos(), time_step=control_time_step
    )
    trajs = plan2["position"]
    print("Pulling out...")
    for i in range(len(trajs)):
        env.render()
        action = convert_traj_to_action(trajs[i], 0.068, KP_PULL, KD_PULL)
        obs, rew, done, info = env.step(action)
        tcp_wrench = env.agent.get_tcp_wrench()
        print(
            f"{i:04d}: obs: {print_dict(obs)}; reward: {rew}, done: {done}, info: {info}, tcp_wrench: {tcp_wrench}"
        )

    print("Press [e] to finish")
    while True:
        if viewer.window.key_down("e"):
            break
        env.render()


if __name__ == "__main__":
    main()
