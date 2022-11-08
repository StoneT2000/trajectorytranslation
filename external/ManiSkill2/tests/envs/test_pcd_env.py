import gym
import open3d as o3d

from mani_skill2.envs.pick_and_place.lift_cube import LiftCubeFixedXmate3RobotiqEnv
from mani_skill2.utils.misc import print_dict
from mani_skill2.utils.wrappers import PointCloudPreprocessObsWrapper


def main():
    human_view = False
    env = gym.make("LiftCubePanda_pointcloud-v0")
    env = PointCloudPreprocessObsWrapper(env)
    env.reset()

    pcd = o3d.geometry.PointCloud()

    if human_view:
        viewer = env.render()
        while not viewer.closed:
            env.render()
            obs, rew, done, info = env.step(None)
            print("obs: ", print_dict(obs))
            print(
                "obs point bbox: ",
                obs["pointcloud"]["xyz"].min(0),
                obs["pointcloud"]["xyz"].max(0),
            )
            pcd.points = o3d.utility.Vector3dVector(obs["pointcloud"]["xyz"])
            pcd.colors = o3d.utility.Vector3dVector(obs["pointcloud"]["rgb"] / 255.0)

            o3d.visualization.draw_geometries([pcd])
    else:
        obs, rew, done, info = env.step(None)
        print("obs: ", print_dict(obs))
        print(
            "obs point bbox: ",
            obs["pointcloud"]["xyz"].min(0),
            obs["pointcloud"]["xyz"].max(0),
        )
        pcd.points = o3d.utility.Vector3dVector(obs["pointcloud"]["xyz"])
        pcd.colors = o3d.utility.Vector3dVector(obs["pointcloud"]["rgb"] / 255.0)

        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
