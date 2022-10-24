import numpy as np

import tr2.envs.blockstacking.test_configs as test_configs


def get_block_spawn_loc(goal: str, cfg: dict, episode_rng: np.random.RandomState, origin_xy, min_dist_to_goal):
    if goal == 'tower':
        # origin_xy = episode_rng.uniform(*test_configs.tower_configs[self.goal_height]['offset'])
        # for i in range(self.goal_height):
        #     self.goal_coords[i] = np.array([
        #         0, 0, (2*i+1)*t,
        #     ])
        # for i in range(len(self.blocks)):
        dist = -1
        while dist < min_dist_to_goal:
            generate_spawn_location=lambda rng: [-0.02 + rng.rand() * 0.04, -0.11 + rng.rand()*0.04]
            loc = generate_spawn_location(episode_rng) #cfg['generate_spawn_location'](episode_rng)
            dist = np.linalg.norm(origin_xy - loc)
        return loc
        # self.TEST_BLOCK_SPAWN_LOCATIONS.append(
        #     loc
        # )