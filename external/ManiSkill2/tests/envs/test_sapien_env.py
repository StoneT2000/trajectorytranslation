import itertools

import gym
import pytest

from mani_skill2.envs.sapien_env import BaseEnv, SapienEnv


def test_warning():
    with pytest.warns(None) as record:
        env = SapienEnv(2, 1)
    assert len(record) == 0, len(record)
    with pytest.warns(UserWarning):
        env = SapienEnv(1, 2)


def test_envs():
    ENV_IDS = [
        "LiftCubePanda-v0",
        "PickCubePanda-v0",
        "StackCubePanda-v0",
        "PickSinglePanda-v0",
        "PickClutterPanda-v0",
    ]
    OBS_MODES = ["state_dict", "state", "rgbd", "pointcloud"]
    REWARD_MODES = ["sparse", "dense"]
    for env_id, obs_mode, reward_mode in itertools.product(
        ENV_IDS, OBS_MODES, REWARD_MODES
    ):
        env: BaseEnv = gym.make(env_id, obs_mode=obs_mode, reward_mode=reward_mode)
        env.reset()
        action_space: gym.Space = env.action_space[env.control_mode]
        for _ in range(5):
            env.step(action_space.sample())
