import argparse

import gym
import numpy as np

import mani_skill2.envs.experimental
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.visualization.cv2 import OpenCVViewer
from mani_skill2.utils.wrappers import ManiSkillActionWrapper, NormalizeActionWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--obs-mode", type=str, default="state_dict")
    parser.add_argument("--reward-mode", type=str, default="dense")
    parser.add_argument("--control-mode", type=str, default="pd_ee_delta_pos")
    parser.add_argument("--render-mode", type=str, default="cameras")
    parser.add_argument("--print-obs", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    args, opts = parser.parse_known_args()
    return args, opts


def main():
    np.set_printoptions(suppress=True, precision=3)
    args, opts = parse_args()

    print("opts:", opts)
    env_kwargs = dict((x, eval(y)) for x, y in zip(opts[0::2], opts[1::2]))
    print("env_kwargs:", env_kwargs)

    env: BaseEnv = gym.make(
        args.env, obs_mode=args.obs_mode, reward_mode=args.reward_mode, **env_kwargs
    )
    env = ManiSkillActionWrapper(env, args.control_mode)
    env = NormalizeActionWrapper(env)

    env.reset()

    print("Observation space", env.observation_space)
    print("Action space", env.action_space)
    print("Control mode", env.control_mode)
    print("Reward mode", env.reward_mode)

    sapien_viewer = env.render(mode="human")
    viewer = OpenCVViewer()
    gripper_action = -1

    while True:
        env.render(mode="human")
        render_frame = env.render(mode=args.render_mode)

        key = viewer.imshow(render_frame)

        if args.control_mode == "pd_ee_delta_pos":
            ee_action = np.zeros([3])
        elif args.control_mode == "pd_ee_delta_pose":
            ee_action = np.zeros([6])
        else:
            raise NotImplementedError

        if key == "i":  # forward
            ee_action[0] = 1
        elif key == "k":  # backward
            ee_action[0] = -1
        elif key == "j":  #
            ee_action[1] = 1
        elif key == "l":
            ee_action[1] = -1
        elif key == "u":  # up
            ee_action[2] = 1
        elif key == "o":  # down
            ee_action[2] = -1
        elif key == "1":  # rot (axis-angle)
            ee_action[3:6] = (0, 0, 1)
        elif key == "2":
            ee_action[3:6] = (0, 0, -1)
        elif key == "3":  # rot (axis-angle)
            ee_action[3:6] = (0, 1, 0)
        elif key == "4":
            ee_action[3:6] = (0, -1, 0)
        elif key == "f":  # close gripper
            gripper_action = 1
        elif key == "g":  # open gripper
            gripper_action = -1
        elif key == "e":
            while True:
                sapien_viewer = env.render(mode="human")
                if sapien_viewer.window.key_down("e"):
                    break
        elif key == "r":
            env.reset()
        elif key == "s":
            viewer.save_to_video(".", video_name=args.env)

        obs, reward, done, info = env.step(np.hstack([ee_action, gripper_action]))
        if args.print_obs:
            if args.obs_mode == "state_dict":
                print(obs)
        print(reward, done, info)


if __name__ == "__main__":
    main()
