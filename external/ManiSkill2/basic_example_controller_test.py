import gym
import numpy as np

import mani_skill.env

env = gym.make(
    "OpenCabinetDoor-v0",
    max_episode_steps=100,
    frame_skip=100,
)
# full environment list can be found in available_environments.txt

env.set_env_mode(obs_mode="state", reward_type="sparse")
# obs_mode can be 'state', 'pointcloud' or 'rgbd'
# reward_type can be 'sparse' or 'dense'
print(
    env.observation_space
)  # this shows the structure of the observation, openai gym's format
print(env.action_space)  # this shows the action space, openai gym's format

for level_idx in range(1):  # level_idx is a random seed
    obs = env.reset(level=level_idx)
    print("#### Level {:d}".format(level_idx))
    for i_step in range(100000):
        env.render(
            "human"
        )  # a display is required to use this function, rendering will slower the running speed
        # action = env.action_space.sample()  # take a random action
        # action = np.zeros(24)
        # action[2:9] = 1.
        # action[9:16] = 1.
        action = np.zeros(21)
        action[2] = 1
        action[3] = -1
        action[5:8] = 0.7
        # action[0] = 1
        obs, reward, done, info = env.step(action)
        print("{:d}: reward {:.4f}, done {}".format(i_step, reward, done))
        # print(env.agent.robot.get_links()[1].get_velocity())
        print(env.agent.robot.get_qpos())
        if done:
            break
env.close()
