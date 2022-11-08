import faulthandler

import gym

import mani_skill.env

faulthandler.enable()

mode = 'Self_Door'
# mode = 'Self_Drawer'
# mode = 'Default'


def random_action(env_name):
    env = gym.make(env_name)
    # full environment list can be found in available_environments.txt

    env.set_env_mode(obs_mode='state', reward_type='sparse')
    # obs_mode can be 'state', 'pointcloud' or 'rgbd'
    # reward_type can be 'sparse' or 'dense'
    print(env.observation_space) # this shows the structure of the observation, openai gym's format
    print(env.action_space) # this shows the action space, openai gym's format

    for level_idx in range(0, 5): # level_idx is a random seed
        obs = env.reset(level=level_idx)
        print('#### Level {:d}'.format(level_idx))
        for i_step in range(100000):
            env.render('human') # a display is required to use this function, rendering will slower the running speed
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action) # take a random action
            print('{:d}: reward {:.4f}, done {}'.format(i_step, reward, done))
            if done:
                break
    env.close()


if __name__ == "__main__":
    env_name = None
    if mode == 'Self_Door':
        env_name = 'OpenCabinetDoor_0-v0'
    elif mode == 'Self_Drawer':
        env_name = 'OpenCabinetDrawer_500-v0'
    elif mode == 'Default':
        env_name = 'OpenCabinetDoor_1000-v0'  # 'OpenCabinetDrawer_1013-v0'
    else:
        print('Not valid mode!')
    random_action(env_name)
