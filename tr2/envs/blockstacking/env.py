import pdb
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import sapien.core as sapien
from mani_skill2.agents.camera import (get_camera_images, get_camera_pcd,
                                       get_camera_rgb)
from mani_skill2.utils.common import register_gym_env
from mani_skill2.utils.sapien_utils import (get_pairwise_contact_impulse,
                                            vectorize_pose)
from sapien.core import Pose

from tr2.envs.blockstacking.block_spawns import \
    get_block_spawn_loc
from tr2.utils.blockstack_utils.goal_state import *
from tr2.utils.pose_est import \
    estimate_single_block_pose_realsense

this_file = Path(__file__).parent.resolve()

AGENT_CONFIG_DIR = this_file / '../../assets/blockstack/agent_config'
import os

from mani_skill2.agents.camera import get_camera_rgb
from mani_skill2.utils.sapien_utils import get_entity_by_name, look_at
from mani_skill2.utils.visualization.misc import (observations_to_images,
                                                  tile_images)
from matplotlib.cm import get_cmap

import tr2.envs.blockstacking.test_configs as test_configs
from tr2.envs.blockstacking.agents.panda import Panda
# from .base_env import PandaEnv, FloatPandaEnv
# from .agents.panda import Panda
from tr2.envs.blockstacking.base_env import (FloatPandaEnv,
                                                          PandaEnv)


def get_color_palette(n):
    # For visualization
    # Note that COLOR_PALETTE is a [C, 3] np.uint8 array.
    cmap=get_cmap('rainbow', n)
    COLOR_PALETTE=np.array([cmap(i)[:3] for i in range(n)])
    COLOR_PALETTE=np.array(COLOR_PALETTE * 255, dtype=np.uint8)
    return COLOR_PALETTE


def get_seg_visualization(seg):
    # assume seg is (n, n)
    assert len(seg.shape) == 2
    COLOR_PALETTE=get_color_palette(np.max(seg) + 1)
    img=COLOR_PALETTE[seg]
    return img

class UniformSampler:
    """Uniform placement sampler.

    Args:
        ranges: ((low1, low2, ...), (high1, high2, ...))
        rng (np.random.RandomState): random generator
    """

    def __init__(
        self, ranges: Tuple[List[float], List[float]], rng: np.random.RandomState
    ) -> None:
        assert len(ranges) == 2 and len(ranges[0]) == len(ranges[1])
        self._ranges = ranges
        self._rng = rng
        self._fixtures = []

    def add_external_fixtures(self, fixtures):
        for f in fixtures:
            self._fixtures.append(f)

    def sample(self, min_radius, max_trials, max_radius=9999, append=True):
        """Sample a position.

        Args:
            radius (float): collision radius.
            max_trials (int): maximal trials to sample.
            append (bool, optional): whether to append the new sample to fixtures. Defaults to True.
            verbose (bool, optional): whether to print verbosely. Defaults to False.

        Returns:
            np.ndarray: a sampled position.
        """
        if len(self._fixtures) == 0:
            pos = self._rng.uniform(*self._ranges)
        else:
            fixture_pos = np.array([x[0] for x in self._fixtures])
            fixture_radius = np.array([x[1] for x in self._fixtures])
            for i in range(max_trials):
                pos = self._rng.uniform(*self._ranges)
                dist = np.linalg.norm(pos - fixture_pos, axis=1)
                if np.all(dist > fixture_radius + min_radius) and np.all(dist < fixture_radius + max_radius):
                    break
        if append:
            self._fixtures.append((pos, min_radius))
        return pos



@register_gym_env("BlockStackArm-v0")
@register_gym_env("BlockStackArm_200-v0", max_episode_steps=200)
class BlockStackPandaEnv(PandaEnv):
    def __init__(self,
                 obs_mode=None,
                 reward_mode=None,
                 num_blocks=2,
                 goal='pick_and_place',
                 task_range='small',
                 controller='arm',
                 always_done = False,
                 show_goal_visuals = True,
                 intervene_count = 0,
                 spawn_all_blocks = False,
                 real_pose_est=False,
                 ):

        self.sim_real_ratio = (4 / 5.08)
        self.show_goal_visuals = show_goal_visuals
        self._num_blocks = num_blocks
        self._goal = goal
        self.goal_coords = None
        self.magic = False
        self._always_done = always_done
        self.controller = controller
        self.real_pose_est = real_pose_est
        self._target_blocks_idx = None
        self.completed_blocks = None
        self.goal_visual = None
        self.spawn_all_blocks = spawn_all_blocks
        self.init_area_visual = None
        self.goal_area_visual = None
        self.goal_fixtures = []
        self.intervene_count = intervene_count
        self.intervene_steps = []
        self.can_go_to_next_stage = True # something the planner / agent controls. tells someone to put the next block in.
        self.task_range = task_range
        if '-' in goal:
            self.test_task = True
            # print('## Warning: num_blocks will be overrided!')
            
            k = goal.find('-')
            h = int(goal[k+1:])
            self.goal_height = h
            self._goal = goal[:k]
            # print("GOAL", goal)
            if 'realtower2' in goal:
                self._num_blocks = h * 2
            elif 'realtower' in goal:
                self._num_blocks = h
            elif 'tower' in goal:
                self._num_blocks = h
            elif 'pyramid' in goal or 'realpyramid' in goal:
                self._num_blocks = int(h * (h+1) / 2)
            elif 'bridge' in goal:
                self._num_blocks = (h-1)*2+1
                self.goal_height = h-1
            elif 'minecraft_villagerhouse' in goal:
                self._num_blocks = h
                self.goal_variant = self._num_blocks
            elif 'minecraft_creeper' in goal:
                self._num_blocks = h
                self.goal_variant = self._num_blocks
            elif 'custom_mc_scene' in goal:
                self._num_blocks = h
                self.goal_variant = self._num_blocks
            else:
                raise NotImplementedError('Unknown goal: {:s}'.format(goal))
        else:
            self.test_task = False

        super(BlockStackPandaEnv, self).__init__(obs_mode=obs_mode, reward_mode=reward_mode)
    def _load_agent(self):
        if self.controller == "arm":
            cfg = "panda_blockstack.yml"
        elif self.controller == "ee":
            cfg = "panda_ee_control.yml"
        elif self.controller == "float_panda":
            cfg = "float_panda.yml"
        elif self.controller == "pointmass":
            cfg = "pointmass.yml"
            builder=self._scene.create_actor_builder()
            builder.add_sphere_visual(radius=0.015, color=[0, 0, 1])
            # self.visual_goals.append()
            self.pointmass: sapien.Actor = builder.build_kinematic('pointmass_agent')
        else:
            raise ValueError(f"Not a valid controller: {self.controller}")
        self._agent = Panda.from_config_file(
            AGENT_CONFIG_DIR / cfg, self._scene, self._control_freq
        )
        self.grasp_site: sapien.Link = get_entity_by_name(
            self._agent._robot.get_links(), "grasp_site"
        )
    def _load_actors(self):
        # if self.test_task:
        self.visual_goals = []
        if self.show_goal_visuals:
            for i in range(self._num_blocks):
                builder=self._scene.create_actor_builder()
                builder.add_sphere_visual(radius=0.015, color=[1, 1, 1])
                self.visual_goals.append(builder.build_static('goal_{:d}'.format(i)))
            
        self._add_ground()
        # self.block_half_size = 0.025/2
        self.block_half_size = 0.02
        self.blocks = []
        for i in range(self._num_blocks):
            builder = self._scene.create_actor_builder()
            half_size = [self.block_half_size for _ in range(3)]
            # builder.add_box_collision(half_size=half_size, material=self.block_material)
            if self.test_task and "bridge" in self._goal:
                if i == self._num_blocks - 1:
                    half_size[0] = self.block_half_size * 3
            builder.add_box_collision(half_size=half_size)
            builder.add_box_visual(half_size=half_size, color=[1, 0, 0])
            # self.cubeA = builder.build("block")
            self.blocks.append(builder.build('block_'+str(i)))
        if self._goal == 'pick_and_place_train':
            self.completed_blocks = []
            for i in range(3):
                builder=self._scene.create_actor_builder()
                half_size=[self.block_half_size for _ in range(3)]
                # builder.add_box_collision(half_size=half_size, material=self.block_material)
                builder.add_box_collision(half_size=half_size)
                builder.add_box_visual(half_size=half_size, color=[1, 0, 0])
                # self.cubeA = builder.build("block")
                self.completed_blocks.append(builder.build('completed_block_' + str(i)))

            
        elif self._goal == 'pick_and_place_silo':
            self.completed_blocks = []
            builder=self._scene.create_actor_builder()
            half_size=[0.25, 0.01, 0.2]
            # builder.add_box_collision(half_size=half_size, material=self.block_material)
            builder.add_box_collision(half_size=half_size)
            builder.add_box_visual(half_size=half_size, color=[0.8, 0.4, 0.4])
            self.walls: List[sapien.Actor] = []
            self.walls += [builder.build_kinematic('left_wall')]
            self.walls[0].set_pose(sapien.Pose(np.array([-0.05, 0.24, 0.1])))
            self.walls += [builder.build_kinematic('right_wall')]
            self.walls[1].set_pose(sapien.Pose(np.array([-0.05, -0.24, 0.1])))
            # for i in range(1):
            #     builder=self._scene.create_actor_builder()
            #     half_size=[self.block_half_size for _ in range(3)]
            #     # builder.add_box_collision(half_size=half_size, material=self.block_material)
            #     builder.add_box_collision(half_size=half_size)
            #     builder.add_box_visual(half_size=half_size, color=[1, 0, 0])
            #     # self.cubeA = builder.build("block")
            #     self.completed_blocks.append(builder.build('completed_block_' + str(i)))

    def _initialize_actors(self):
        self.cur_block_idx = 0
        if self.test_task and not self.spawn_all_blocks:
            for i in range(self._num_blocks):
                self.blocks[i].set_pose(Pose([10+i*0.05, 10, self.block_half_size]))
            self.blocks[0].set_pose(Pose([self.TEST_BLOCK_SPAWN_LOCATIONS[0][0], self.TEST_BLOCK_SPAWN_LOCATIONS[0][1], self.block_half_size]))
            return
        

        ranges = [[-0.055, -0.15], [0.1, 0.15]]
        if self.task_range == 'large':
            ranges = [[-0.2, -0.25], [0.22, 0.25]]
        
        sampler = UniformSampler(ranges, self._episode_rng)
        if self._goal == 'pick_and_place_train' or self.test_task:
            sampler.add_external_fixtures(self.goal_fixtures)
            sampler.add_external_fixtures([(self.goal_coords[0][:2], (self.block_half_size))])
        # if 
        if self._goal == 'pick_and_place_silo':
            # block is always placed in the same location according to silo paper
            self.blocks[0].set_pose(Pose([-0.09, 0, self.block_half_size]))
        else:
            for block in self.blocks:
                dist_to_core = 9999
                while dist_to_core < 0.35 or dist_to_core > 0.72:
                    pos = pos = sampler.sample(min_radius=self.block_half_size*4, max_radius=999, max_trials=200)
                    dist_to_core = np.linalg.norm(pos - np.array([-0.56, 0.0]))
                block.set_pose(Pose([pos[0],  pos[1], self.block_half_size]))

    def _initialize_complete_blocks(self):

        temp_offset = 4
        if self._goal == 'pick_and_place_train':
            for block in self.completed_blocks:
                block.set_pose(Pose([temp_offset, temp_offset, self.block_half_size]))
                temp_offset += 0.5


    def _initialize_agent(self):
        qpos = np.array(
            [0, np.pi / 16, 0, -np.pi * 5 / 6, 0, np.pi - 0.2, np.pi / 4, 0, 0]
        )

        self._agent.reset(qpos)
        self._agent._robot.set_pose(Pose([-0.56, 0, 0]))

    def initialize_episode(self):
        self._initialize_complete_blocks()
        self._set_goal()
        super().initialize_episode()
        if self.show_goal_visuals:
            self.draw_goal_site()

    def _get_obs_extra(self):
        if self._obs_mode in ["rgbd", "pointcloud"]:
            return OrderedDict(
                gripper_pose=vectorize_pose(self.grasp_site.pose),
            )
        else:
            obs_blocks=self._get_obs_blocks()
            obs_blocks['absolute'] = np.hstack(obs_blocks['absolute'])
            assert len(obs_blocks['absolute']) == self._num_blocks * 7
            extra_obs=OrderedDict(
                obs_blocks=obs_blocks,
            )
            return extra_obs
    def _get_obs_blocks(self):
        '''this method returns blocks observation including all of their absolute poses (p,q)
        and relative pos from gripper to blocks

        :param include_relative: wheter to include relative vectors
        :return:
        '''
        absolute = []
        for block in self.blocks:
            # get absolute block pose
            block_pose: Pose = block.pose
            stacked_block_pose = np.hstack([block_pose.p, block_pose.q])
            absolute.append(stacked_block_pose)

        return OrderedDict(
            absolute=np.array(absolute)
        )

    def _set_goal(self):
        '''this function perturb the goal state ( +/- 0.1)
        '''
        if self.test_task:
            self.intervene_steps = []
            for i in range(self.intervene_count):
                idx=int(self._episode_rng.rand()*len(self.blocks))
                self.intervene_steps.append(idx)
            self.intervene_steps = sorted(self.intervene_steps)

            self.goal_coords = np.zeros((self._num_blocks, 3))
            t = self.block_half_size
            

            min_dist_to_goal = (self.block_half_size * 0.008) * 1.414 + self.block_half_size * 6
            if self._goal == 'tower':
                origin_xy = self._episode_rng.uniform(*test_configs.tower_configs[self.goal_height]['offset'])
                self.TEST_BLOCK_SPAWN_LOCATIONS = []
                for i in range(self.goal_height):
                    self.goal_coords[i] = np.array([
                        0, 0, (2*i+1)*t,
                    ])
                for i in range(len(self.blocks)):
                    dist = -1
                    while dist < min_dist_to_goal:
                        loc = test_configs.tower_configs[self.goal_height]['generate_spawn_location'](self._episode_rng)
                        dist = np.linalg.norm(self.goal_coords[i][:2] + origin_xy - loc)
                    self.TEST_BLOCK_SPAWN_LOCATIONS.append(
                        loc
                    )
            elif self._goal == 'realtower':
                origin_xy = self._episode_rng.uniform(*test_configs.realtower[self.goal_height]['offset'])
                self.TEST_BLOCK_SPAWN_LOCATIONS = []
                for i in range(self.goal_height):
                    self.goal_coords[i] = np.array([
                        0, 0, (2*i+1)*t,
                    ])
                for i in range(len(self.blocks)):
                    dist = -1
                    # while dist < min_dist_to_goal:
                    #     loc = test_configs.realtower[self.goal_height]['generate_spawn_location'](self._episode_rng)
                    #     dist = np.linalg.norm(self.goal_coords[i][:2] + origin_xy - loc)
                    loc=np.zeros(2)
                    self.TEST_BLOCK_SPAWN_LOCATIONS.append(
                        loc
                    )
            elif self._goal == 'realtower2':
                origin_xy = self._episode_rng.uniform(*test_configs.realtower[self.goal_height]['offset'])
                self.TEST_BLOCK_SPAWN_LOCATIONS = []
                # self.goal_coords = np.zeros((self.goal_height * 2, 3))
                for i in range(self.goal_height):
                    self.goal_coords[i] = np.array([
                        0.01, -0.04, (2*i+1)*t,
                    ])
                for i in range(self.goal_height):
                    self.goal_coords[i + self.goal_height] = np.array([
                        0.08, 0, (2*i+1)*t,
                    ])
                for i in range(len(self.blocks)):
                    dist = -1
                    self.TEST_BLOCK_SPAWN_LOCATIONS.append(
                        np.zeros(2)
                    )
            elif self._goal == 'pyramid':
                k = 0
                origin_xy = self._episode_rng.uniform(*test_configs.pyramid_configs[self.goal_height]['offset'])
                self.TEST_BLOCK_SPAWN_LOCATIONS = []
                for _ in range(len(self.blocks)):
                    self.TEST_BLOCK_SPAWN_LOCATIONS.append(
                        test_configs.pyramid_configs[self.goal_height]['generate_spawn_location'](self._episode_rng)
                    )
                gap_dist = self.block_half_size*1.4*5.0/4
                for i in range(self.goal_height):
                    for j in range(self.goal_height-i):
                        n = self.goal_height-i
                        self.goal_coords[k] = np.array([
                            0, (n-1-j*2)*t - gap_dist*(j+i/2), (2*i+1)*t,
                        ])
                        k += 1
            elif self._goal == 'realpyramid':
                k = 0
                origin_xy = self._episode_rng.uniform(*test_configs.pyramid_configs[self.goal_height]['offset'])
                self.TEST_BLOCK_SPAWN_LOCATIONS = []
                for _ in range(len(self.blocks)):
                    self.TEST_BLOCK_SPAWN_LOCATIONS.append(
                        test_configs.pyramid_configs[self.goal_height]['generate_spawn_location'](self._episode_rng)
                    )
                gap_dist = self.block_half_size*1.5 * 5.0/4
                for i in range(self.goal_height):
                    for j in range(self.goal_height-i):
                        n = self.goal_height-i
                        x = self.goal_height - i - j - 1
                        self.goal_coords[k] = np.array([
                            0, (n-1-x*2)*t - gap_dist*(x+i/2), (2*i+1)*t,
                        ])
                        k += 1
            elif self._goal == 'minecraft_villagerhouse':
                origin_xy = test_configs.minecraft_villagerhouse_configs[self.goal_variant]['offset'][0]
                self.TEST_BLOCK_SPAWN_LOCATIONS = []
                for _ in range(len(self.blocks)):
                    self.TEST_BLOCK_SPAWN_LOCATIONS.append(
                        test_configs.minecraft_villagerhouse_configs[self.goal_variant]['generate_spawn_location'](self._episode_rng)
                    )
                idx = 0
                gap_dist = 0.45
                for i in range(3):
                    for j in range(3):
                        self.goal_coords[idx] = np.array([
                            i*t*2 + i*t*gap_dist, j*t*gap_dist+j*t*2, t
                        ])
                        idx += 1
            elif self._goal == 'minecraft_creeper':
                origin_xy = test_configs.minecraft_creeper_configs[self.goal_variant]['offset'][0]
                self.TEST_BLOCK_SPAWN_LOCATIONS = []
                for _ in range(len(self.blocks)):
                    self.TEST_BLOCK_SPAWN_LOCATIONS.append(
                        test_configs.minecraft_creeper_configs[self.goal_variant]['generate_spawn_location'](self._episode_rng)
                    )
                idx = 0
                gap_dist = 0.5
                self.goal_coords[0] = np.array([-t - t*gap_dist, -t - t*gap_dist, t])
                self.goal_coords[1] = np.array([t + t*gap_dist, -t - t*gap_dist, t])
                self.goal_coords[2] = np.array([t + t*gap_dist, t + t*gap_dist, t])
                self.goal_coords[3] = np.array([-t - t*gap_dist, t + t*gap_dist, t])
                
                for i in range(3):
                    self.goal_coords[4+i*2] = np.array([-t, 0, (2*(i+1)+1)*t])
                    self.goal_coords[4+i*2+1] = np.array([t, 0, (2*(i+1)+1)*t])
            elif self._goal == 'realcustom_mc_scene_1':
                origin_xy = test_configs.custom_mc_scene_1[self.goal_variant]['offset'][0]
                self.TEST_BLOCK_SPAWN_LOCATIONS = []
                for _ in range(len(self.blocks)):
                    self.TEST_BLOCK_SPAWN_LOCATIONS.append(
                        test_configs.custom_mc_scene_1[self.goal_variant]['generate_spawn_location'](self._episode_rng)
                    )
                gap_dist = self.block_half_size * 0.5
                self.goal_coords[0] = np.array([-t - gap_dist, 0, t])
                self.goal_coords[1] = np.array([t + gap_dist, 0, t])
                for i in range(3):
                    self.goal_coords[i+2] = np.array([
                        0, 0, (2*i+1+2)*t,
                    ])
            elif self._goal == 'realcustom_mc_scene_2':
                origin_xy = test_configs.realcustom_mc_scene_2[self.goal_variant]['offset'][0]
                self.TEST_BLOCK_SPAWN_LOCATIONS = []
                for _ in range(len(self.blocks)):
                    self.TEST_BLOCK_SPAWN_LOCATIONS.append(
                        test_configs.realcustom_mc_scene_2[self.goal_variant]['generate_spawn_location'](self._episode_rng)
                    )
                gap_dist = self.block_half_size * 0.5
                self.goal_coords[0] = np.array([-t - gap_dist, 0, t])
                self.goal_coords[1] = np.array([t + gap_dist, 0, t])
                self.goal_coords[2] = np.array([-t - gap_dist, -t*3.5+gap_dist, t])
                self.goal_coords[3] = np.array([t + gap_dist, -t*3.5+gap_dist, t])
                self.goal_coords[4] = np.array([-t - gap_dist, 0, t*3])
                self.goal_coords[5] = np.array([t + gap_dist, 0, t*3])
                self.goal_coords[7] = np.array([-t - gap_dist, 0, t*5])
            elif self._goal == 'realcustom_mc_scene_3':
                origin_xy = test_configs.realcustom_mc_scene_3[self.goal_variant]['offset'][0]
                self.TEST_BLOCK_SPAWN_LOCATIONS = []
                for _ in range(len(self.blocks)):
                    self.TEST_BLOCK_SPAWN_LOCATIONS.append(
                        test_configs.realcustom_mc_scene_3[self.goal_variant]['generate_spawn_location'](self._episode_rng)
                    )
                gap_dist = self.block_half_size * 0.5
                self.goal_coords[0] = np.array([-t - gap_dist, 0, t])
                self.goal_coords[1] = np.array([t + gap_dist, 0, t])
                self.goal_coords[2] = np.array([-t*1.15 - gap_dist, 0, t*3])
                self.goal_coords[3] = np.array([t*1.15 + gap_dist, 0, t*3])
                self.goal_coords[4] = np.array([0, 0, t*5])
                self.goal_coords[5] = np.array([0, 0, t*7])
                self.goal_coords[6] = np.array([0, 0, t*9])
            elif self._goal == 'realcustom_mc_scene_4':
                origin_xy = test_configs.realcustom_mc_scene_4[self.goal_variant]['offset'][0]
                self.TEST_BLOCK_SPAWN_LOCATIONS = []
                for _ in range(len(self.blocks)):
                    self.TEST_BLOCK_SPAWN_LOCATIONS.append(
                        test_configs.realcustom_mc_scene_4[self.goal_variant]['generate_spawn_location'](self._episode_rng)
                    )
                gap_dist = self.block_half_size * 0.6
                y_gap = 1.5
                idx = 0
                # build effectively 8 towers of size (num_blocks - 4 )// 8 and add 4 blocks on each corner
                max_height = (self._num_blocks - 4) // 8
                for h in range(max_height):
                    for i in range(2,-1, -1):
                        for j in range(2,-1, -1):
                            if i == 1 and j == 1: continue
                            self.goal_coords[idx] = np.array([(-t - gap_dist)*2 + (t+gap_dist) * 2 * i, (-t - gap_dist*y_gap)*2 + (t+gap_dist*y_gap) * 2 * j, t + (2*h*t)])
                            idx += 1
                
                for i in range(2,-1, -1):
                    for j in range(2,-1, -1):
                        if i == 1 or j == 1: continue
                        self.goal_coords[idx] = np.array([(-t - gap_dist)*2 + (t+gap_dist) * 2 * i, (-t - gap_dist*y_gap)*2 + (t+gap_dist*y_gap) * 2 * j, t + (2*max_height*t)])
                        idx += 1
                # self.goal_coords[1] = np.array([(t + gap_dist)*3, 0, t])
                # self.goal_coords[2] = np.array([-t*1.15 - gap_dist, 0, t*3])
                # self.goal_coords[3] = np.array([t*1.15 + gap_dist, 0, t*3])
                # self.goal_coords[4] = np.array([0, 0, t*5])
                # self.goal_coords[5] = np.array([0, 0, t*7])
                # self.goal_coords[6] = np.array([0, 0, t*9])
            elif self._goal == 'bridge':
                origin_xy = self._episode_rng.uniform([0.04,-0.06],[0.06, 0.06])
                k = 0
                # gap_dist = self.block_half_size*0.45
                for i in range(self.goal_height):
                    self.goal_coords[i] = np.array([
                        -0.05, 0, (2*i+1)*t,
                    ])
                for i in range(self.goal_height):
                    self.goal_coords[i+self.goal_height] = np.array([
                        0.05, 0, (2*i+1)*t,
                    ])
                # self.goal_coords[self.goal_height-1][2] += 1e-2
                # self.goal_coords[self.goal_height-1][2] += 1e-2
                # self.goal_coords[0] = np.array([0.04,0.0, self.block_half_size])
                # self.goal_coords[1] = np.array([0.12,0.0, self.block_half_size])
                self.goal_coords[-1] = np.array([0, 0, self.goal_height * self.block_half_size*2 + self.block_half_size*2])
            self.goal_coords[:, :2] += origin_xy
            self.config_origin_xy = origin_xy
            if self.show_goal_visuals:
                for i in range(self._num_blocks):
                    self.visual_goals[i].set_pose(Pose(self.goal_coords[i]))

            if not self.magic and self.real_pose_est:
                first_block_pose = estimate_single_block_pose_realsense()
                first_block_T = first_block_pose[:3, 3]
                print("ESTIMATED BLOCK T", first_block_T)
                xy=np.array([first_block_T[0] + test_configs.ROBOT_X_OFFSET, first_block_T[1]+ test_configs.ROBOT_Y_OFFSET])
                xy *= self.sim_real_ratio
                self.TEST_BLOCK_SPAWN_LOCATIONS[0][:2] = xy
                print("ESTIMATED BLOCK XY", xy)
            return

        if self.task_range == 'small':
            random_xy = self._episode_rng.uniform([-0.02,-0.07],[0.1,0.07]) # old range
        
        elif self.task_range == 'large':
            dist_to_core = 9999
            # while dist_to_core < 0.35 or dist_to_core > 0.72:
            while dist_to_core < 0.45 or dist_to_core > 0.5:
                random_xy = self._episode_rng.uniform([-0.2, -0.25], [0.22, 0.25])
                dist_to_core = np.linalg.norm(random_xy - np.array([-0.56, 0]))
        if self._goal == 'tower':
            assert self._num_blocks == 9
            self.goal_coords=TOWER_GOAL_COORD.copy()
            self.goal_coords[:, :2] += random_xy

        elif self._goal == 'pick_and_place':
            assert self._num_blocks == 2
            self.goal_coords=SIMPLE_GOAL_COORD_25.copy()
            self.goal_coords[:,:2] += random_xy

        elif self._goal == 'simple_tower':
            assert self._num_blocks == 4
            self.goal_coords = SIMPLE_TOWER_GOAL_COORD.copy()
            self.goal_coords[:, :2] += random_xy

        elif self._goal == 'random_simple_tower':
            assert self._num_blocks >= 4
            self._target_blocks_idx = self._episode_rng.choice(self._num_blocks, 4, replace=False)
            self.goal_coords = np.zeros((self._num_blocks, 3))
            target_blocks_goal_coords = SIMPLE_TOWER_GOAL_COORD.copy()
            target_blocks_goal_coords[:, :2] += random_xy
            self.goal_coords[self._target_blocks_idx] = target_blocks_goal_coords

            for i in range(self._num_blocks):
                if i not in self._target_blocks_idx:
                    self.goal_coords[i] = self._blocks[i].pose.p
        elif self._goal == 'pick_and_place_silo':
            self.goal_coords=np.array([[0.08, 0., 0.08]])
            # paper doesn'tmention varying how far out to move block
            # self.goal_coords[:, 1] += 
            # self._episode_rng.uniform(-0.05, 0.05)
            # self.(self._episode_rng.randint(0, 2) * -1)
            self.goal_coords[:, 2] += self._episode_rng.uniform(0, 0.01) # range z by 0.06-0.07

        elif self._goal == 'pick_and_place_train':
            

            assert self._num_blocks == 1
            sub_task_indicator = -1
            original_goal = SIMPLE_GOAL_COORD_TRAIN.copy() if sub_task_indicator > 0 else STACK_OF_THREE_TRAIN.copy()
            #original_goal = SIMPLE_GOAL_COORD_TRAIN.copy()
            num_completed = self._episode_rng.choice(3,1)[0]
            completed_blocks_coords = original_goal[:num_completed]
            completed_blocks_coords[:,:2] += random_xy
            self.goal_coords=np.array([original_goal[num_completed]])
            self.goal_coords[:, :2]+=random_xy

            for i in range(num_completed):
                pose = completed_blocks_coords[i]
                self.completed_blocks[i].set_pose(Pose(pose))

            # add fixture info for later initializing actor
            self.goal_fixtures = [] # remove past fixtures
            if sub_task_indicator > 0:
                # never used
                # pyramid case
                for i in range(min(num_completed,3)):
                    pos1 = completed_blocks_coords[i][:2]
                    pos2 = completed_blocks_coords[i][:2] + np.array([0,0.025])
                    pos3 = completed_blocks_coords[i][:2] + np.array([0,-0.025])
                    radius = self.block_half_size*2
                    self.goal_fixtures.append((pos1,radius))
                    self.goal_fixtures.append((pos2,radius))
                    self.goal_fixtures.append((pos3,radius))
            else :
                if num_completed != 0:
                    # tower case:
                    pos1 = completed_blocks_coords[0][:2]
                    pos2 = completed_blocks_coords[0][:2]+ np.array([0,0.04])
                    pos3 = completed_blocks_coords[0][:2]+ np.array([0,-0.04])
                    radius= self.block_half_size*2
                    self.goal_fixtures.append((pos1, radius))
                    self.goal_fixtures.append((pos2, radius))
                    self.goal_fixtures.append((pos3, radius))

                    # build up to 3 towers
                    # if num_completed > 4:
                    #     x = pos1[0]
                    #     while x < 0.08:
                    #         pos1=completed_blocks_coords[0][:2] + np.array([0.04,0])
                    #         pos2=completed_blocks_coords[0][:2] + np.array([0.04, 0.03])
                    #         pos3=completed_blocks_coords[0][:2] + np.array([0.04, -0.03])
                    #         radius=(self.block_half_size + 0.008) * 1.414
                    #         self.goal_fixtures.append((pos1, radius))
                    #         self.goal_fixtures.append((pos2, radius))
                    #         self.goal_fixtures.append((pos3, radius))
                    #         x += 0.04

            # self.goal_coords = np.array([originoral_goal[num_completed]])
            # self.goal_coords[:,:2] += random_xy
            #
            # self.goal_coords = SIMPLE_GOAL_COORD_TRAIN.copy()
            # self.goal_coords[:, :2] += random_xy
        else:
            return NotImplementedError
        if self.show_goal_visuals:
            for i in range(self._num_blocks):
                self.visual_goals[i].set_pose(Pose(self.goal_coords[i]))
    def draw_goal_site(self):
        for i in range(len(self.goal_coords)):
            self.visual_goals[i].set_pose(Pose(self.goal_coords[i]))
        # self.goal_area_visual.set_pose(Pose(np.array([0.09, 0, 0.2])))
        # self.init_area_visual.set_pose(Pose(np.array([-0.0475,0, 0.025])))

    def _load_complete_blocks(self):
        assert self._goal == 'pick_and_place_train'

    def check_success(self) -> bool:
        if self.test_task:
            blocks_pose = np.array([
                b.pose.p for b in self.blocks
            ])
            max_dist = np.linalg.norm(self.goal_coords - blocks_pose, axis=1).max()
            min_gripper_dist = np.linalg.norm(self.grasp_site.pose.p - blocks_pose, axis=1).min()
            far_enough = min_gripper_dist > self.block_half_size*4
            success_allowance_factor = 1.25
            if 'real' in self._goal:
                success_allowance_factor =1.8
            return max_dist < self.block_half_size*success_allowance_factor and far_enough
        if self._goal == 'pick_and_place_silo':
            block = self.blocks[0]
            goal_pos = self.goal_coords[0]
            block_pos = block.pose.p
            block_to_goal_dist = np.linalg.norm(goal_pos - block_pos)
            reached = False
            if block_to_goal_dist <= self.block_half_size:
                reached = True
            return reached
        if self._goal == 'pick_and_place_train':
            gripper_pos = self.grasp_site.pose.p
            reached , far_enough = False, False
            block = self.blocks[0]
            block_pos = block.pose.p
            goal_pos = self.goal_coords[0]

            block_to_goal_dist = np.linalg.norm(goal_pos - block_pos)
            if block_to_goal_dist <= self.block_half_size:
                reached = True
            far_enough = False
            cubeA_pos = self.blocks[0].pose.p
            block_pos = np.hstack([cubeA_pos[0:2], cubeA_pos[2]])
            block_gripper_dist = np.linalg.norm(gripper_pos - block_pos)
            far_enough = block_gripper_dist > self.block_half_size * 4
            return reached and far_enough

    def get_viewer(self):
        return self._viewer


    def render(self, mode="human"):
        cam = self.state_visual
        if mode == 'state_visual':
            self.update_render()
            cam.take_picture()

            rgbd = self._get_camera_images(
                self.state_visual, rgb=True, depth=True, visual_seg=True, actor_seg=True,
            )

            #pdb.set_trace()
            return rgbd

        else:
            return super(BlockStackPandaEnv, self).render(mode=mode)



    @property
    def agent(self):
        return self._agent

    def compute_dense_reward(self):
        reward = 0.0
        # if self.check_success():
        #     reward = 2.25 + 1
        # else:
        # reaching object reward
        gripper_pos = self.grasp_site.pose.p
        cubeA_pos = self.blocks[0].pose.p
        cubeB_pos = self.goal_coords[0]#self.cubeB.pose.p
        goal_xyz = cubeB_pos
        cubeA_to_goal_dist = np.linalg.norm(goal_xyz - cubeA_pos)

        # grasping reward
        is_cubeA_grasped = self._agent.check_grasp(self.blocks[0])

        block_at_target = cubeA_to_goal_dist <= self.block_half_size
        if not block_at_target:
            target_gripper_pos = np.hstack([cubeA_pos[0:2], cubeA_pos[2]])
            cubeA_to_gripper_dist = np.linalg.norm(gripper_pos - target_gripper_pos)
            reaching_reward = 1 - np.tanh(10.0 * cubeA_to_gripper_dist)
            reward += reaching_reward
            if is_cubeA_grasped:
                reward += 0.5
            # stage 2, hold on and move block
            if is_cubeA_grasped:
                reaching_reward2 = 1 - np.tanh(10.0 * cubeA_to_goal_dist)
                reward += reaching_reward2
                # max reward here is 2.5
        else:
            reward += 2.5 + 1 # min reward here is 3.5, +1 more than previous stage
            # encourage to leave block where ispawn_all_blocks=Truet is basically
            # [-1.0194329e-01  1.2922101e-08  9.2596292e-02]
            if not is_cubeA_grasped:
                # if not grasping / we released the block, give even more reward
                reward += 10
                target_gripper_pos = np.array([-0.1, 0, 0.1])
                agent_to_rest_dist = np.linalg.norm(gripper_pos - target_gripper_pos)
                reaching_reward3 = 1 - np.tanh(10.0 * agent_to_rest_dist)
                reward += reaching_reward3*2
                # max reward is 5.25+2 = 7.5

        return reward

    def get_state(self):
        if self.test_task:
            goal_info = self.goal_coords.copy()

            blocks_info = dict(
                xyz = [],
                quat = [],
                vel = [],
                ang_vel = [],
            )
            for b in self.blocks:
                b:sapien.Actor
                b_xyz, b_quat = b.pose.p, b.pose.q
                b_vel = b.get_velocity()
                b_ang_vel = b.get_angular_velocity()
                blocks_info['xyz'].append(b_xyz)
                blocks_info['quat'].append(b_quat)
                blocks_info['vel'].append(b_vel)
                blocks_info['ang_vel'].append(b_ang_vel)

            hand = None
            for link in self.get_articulations()[0].get_links():
                if link.name == 'panda_hand':
                    hand = link
            hand_xyz = hand.pose.p
            agent: Panda = self.agent

            is_grasped = agent.check_grasp(self.blocks[0])


            cur_block_idx = self.cur_block_idx
            return dict(
                blocks_info=blocks_info,
                goal_info=goal_info,
                hand_xyz=hand_xyz,
                is_grasped=is_grasped,
                cur_block_idx=cur_block_idx
            )

        # get student state
        complete_blocks_info = dict(
            xyz = [],
        )

        blocks_info = dict(
            xyz = [],
            quat = [],
            vel = [],
            ang_vel = [],
        )

        goal_info = self.goal_coords

        for cb in self.completed_blocks:
            cb: sapien.Actor
            cb_pose = cb.pose.p
            complete_blocks_info['xyz'].append(cb_pose)
        for b in self.blocks:
            b:sapien.Actor
            b_xyz, b_quat = b.pose.p, b.pose.q
            b_vel = b.get_velocity()
            b_ang_vel = b.get_angular_velocity()
            blocks_info['xyz'].append(b_xyz)
            blocks_info['quat'].append(b_quat)
            blocks_info['vel'].append(b_vel)
            blocks_info['ang_vel'].append(b_ang_vel)

        # articulation_info, here we only need to get studet robot hand state
        hand = None
        for link in self.get_articulations()[0].get_links():
            if link.name == 'panda_hand':
                hand = link
        hand_xyz = hand.pose.p
        arm_qpos = self.get_articulations()[0].get_qpos()
        arm_qvel = self.get_articulations()[0].get_qvel()
        arm_qacc = self.get_articulations()[0].get_qacc()
        agent: Panda = self.agent
        is_grasped = agent.check_grasp(self.blocks[0])
        return dict(
            complete_blocks_info=complete_blocks_info,
            blocks_info=blocks_info,
            goal_info=goal_info,
            hand_xyz=hand_xyz,
            is_grasped=is_grasped,
            arm_qpos=arm_qpos,
            arm_qvel=arm_qvel,
            arm_qacc=arm_qacc,
        )
    def set_state(self, state):
        if 'complete_blocks_info' in state:
            complete_blocks_info= state['complete_blocks_info']
            for i in range(len(self.completed_blocks)):
                cb: sapien.Actor=self.completed_blocks[i]
                cb.set_pose(Pose(p=complete_blocks_info['xyz'][i]))
        if 'cur_block_idx' in state:
            self.cur_block_idx = state['cur_block_idx']
        blocks_info=state['blocks_info']
        goal_info=state['goal_info']
        if 'arm_qpos' in state:
            arm_qpos = state['arm_qpos']
            arm_qvel = state['arm_qvel']
            arm_qacc = state['arm_qacc']
            self.get_articulations()[0].set_qpos(arm_qpos)
            self.get_articulations()[0].set_qvel(arm_qvel)
            self.get_articulations()[0].set_qacc(arm_qacc)

        is_grasped=state['is_grasped']


        for i in range(len(self.blocks)):
            b:sapien.Actor = self.blocks[i]
            b.set_pose(Pose(p=blocks_info['xyz'][i], q=blocks_info['quat'][i]))
            b.set_velocity(blocks_info['vel'][i])
            b.set_angular_velocity(blocks_info['ang_vel'][i])
        self.goal_coords = goal_info

        # if self.show_goal_visuals is not None and self.goal_visual is not None:
        #     self.goal_visual.set_pose(Pose(self.goal_coords[0]))

    def allow_next_stage(self, toggle):
        self.can_go_to_next_stage = toggle

    def step(self, *args, **kwargs):
        if self.controller == 'pointmass':
            pm_pose = self.grasp_site.pose
            self.pointmass.set_pose(pm_pose)

        if self.test_task and self.cur_block_idx < self._num_blocks:
            cur_block = self.blocks[self.cur_block_idx]
            cur_goal = self.goal_coords[self.cur_block_idx]
            d = np.linalg.norm(cur_block.pose.p - cur_goal)
            success_allowance_factor = 1.25
            if 'real' in self._goal:
                success_allowance_factor = 1.8
            gripper_pos = self.grasp_site.pose.p
            gripper_dist_to_rest = np.linalg.norm(gripper_pos - np.array([-0.1, 0, 0.14]))
            if d < self.block_half_size*success_allowance_factor and not self.agent.check_grasp(cur_block) and self.can_go_to_next_stage:
                # if correct, now perform intervention by replacing block somewhere
                if len(self.intervene_steps) > 0 and self.intervene_steps[0] == self.cur_block_idx:

                    min_dist_to_goal = (self.block_half_size * 0.008) * 1.414 + self.block_half_size * 6
                    pos = get_block_spawn_loc(self._goal, test_configs.tower_configs[self.goal_height], self._episode_rng, self.config_origin_xy, min_dist_to_goal=min_dist_to_goal)
                    self.blocks[self.cur_block_idx].set_pose(Pose([pos[0], pos[1], self.block_half_size]))
                    self.intervene_steps = self.intervene_steps[1:]
                else:
                    self.cur_block_idx += 1
                    if self.real_pose_est:
                        user_input = input("Ready for next pose estimation?")
                        if user_input != "n":
                            guess_block_T = []
                            for i in range(3):
                                block_pose = estimate_single_block_pose_realsense()
                                guess_block_T.append(block_pose[:3, 3])
                            guess_block_T = np.array(guess_block_T)
                            block_T = np.median(guess_block_T, 0)
                            xy=np.array([block_T[0] + test_configs.ROBOT_X_OFFSET, block_T[1]+ test_configs.ROBOT_Y_OFFSET])
                            xy *= self.sim_real_ratio
                            self.TEST_BLOCK_SPAWN_LOCATIONS[self.cur_block_idx][:2] = xy
                            print("ESTIMATED XY", xy)
                    if self.cur_block_idx < self._num_blocks and not self.spawn_all_blocks:
                        self.blocks[self.cur_block_idx].set_pose(Pose([self.TEST_BLOCK_SPAWN_LOCATIONS[self.cur_block_idx][0], self.TEST_BLOCK_SPAWN_LOCATIONS[self.cur_block_idx][1], self.block_half_size]))

        r = super().step(*args, **kwargs)
        # self.render("human")
        return r


    # def get_done(self):
    #     return self.check_success()

@register_gym_env("BlockStackFloat-v0")
@register_gym_env("BlockStackFloat_200-v0", max_episode_steps=200)
class BlockStackFloatPandaEnv(BlockStackPandaEnv, FloatPandaEnv):
    def _initialize_agent(self):
        qpos = np.array(
            # [0, 0, 0.2, 0, 0, 1.5707963267948966, 0.04, 0.04]
            [0, 0, 0.2, 0.04, 0.04]
        )
        # qpos[:3] += self._episode_rng.normal(0, 0.02, 3)
        # adjust to arm graps site pos
        qpos[:3] += np.array([-1.15014605e-01 , 2.14204192e-08 ,-5.23643196e-03])
        self._agent.reset(qpos)


@register_gym_env("BlockStackMagic-v0")
@register_gym_env("BlockStackMagic_200-v0", max_episode_steps=200)
class BlockStackMagicPandaEnv(BlockStackPandaEnv, FloatPandaEnv):
    def __init__(self,
                 obs_mode=None,
                 reward_mode=None,
                 num_blocks=2,
                 goal='pick_and_place',
                 task_range='small',
                 spawn_all_blocks=False,
                 always_done = False):

        self._magic_drive = None
        self._connected = False
        self.magic = True
        # same super init()
        super(BlockStackMagicPandaEnv, self).__init__(obs_mode, reward_mode, num_blocks, goal, always_done=always_done, controller="pointmass", task_range=task_range, spawn_all_blocks=spawn_all_blocks)
    def _initialize_agent(self):
        qpos = np.array(
            # [0, 0, 0.2, 0, 0, 1.5707963267948966, 0.04, 0.04]
            [0, 0, 0.2, 0.04, 0.04]
        )
        #qpos[:3] += self._episode_rng.normal(0, 0.02, 3)
        # adjust to arm graps site pos
        qpos[:3] += np.array([-1.15014605e-01 , 2.14204192e-08 ,-5.23643196e-03])
        self._agent.reset(qpos)


    def magic_grasp(self, block):
        # grasp site
        actor1 = self.grasp_site
        #actor1 = self.get_articulations()[0].get_links()[-1]
        pose1 = sapien.Pose([-0,0,0])

        # block actor
        actor2 = block
        # gripper pose -> wTg, block pose -> wTb, want gripper pose under block frame -> gTb = wTg^-1 @ wTb
        pose2=actor1.pose.inv().to_transformation_matrix() @ actor2.pose.to_transformation_matrix()
        pose2=sapien.Pose.from_transformation_matrix(pose2)
        magic_drive=self._scene.create_drive(actor1, pose1, actor2, pose2)
        magic_drive.set_x_properties(stiffness=5e4, damping=3e3)
        magic_drive.set_y_properties(stiffness=5e4, damping=3e3)
        magic_drive.set_z_properties(stiffness=5e4, damping=3e3)
        magic_drive.set_slerp_properties(stiffness=5e4, damping=3e3)

        self._connected = True
        self._magic_drive = magic_drive


    def magic_release(self):
        if self._magic_drive is None: return
        self._scene.remove_drive(self._magic_drive)
        self._connected=False
        self._magic_drive=None

    @property
    def connected(self):
        return self._connected


    def set_state(self, state: dict):
        if 'complete_blocks_info' in state:
            complete_blocks_info= state['complete_blocks_info']
            for i in range(len(self.completed_blocks)):
                cb: sapien.Actor=self.completed_blocks[i]
                cb.set_pose(Pose(p=complete_blocks_info['xyz'][i]))
        if 'cur_block_idx' in state:
            self.cur_block_idx = state['cur_block_idx']
        blocks_info=state['blocks_info']
        goal_info=state['goal_info']
        hand_xyz=state['hand_xyz']
        is_grasped=state['is_grasped']


        for i in range(len(self.blocks)):
            b:sapien.Actor = self.blocks[i]
            b.set_pose(Pose(p=blocks_info['xyz'][i], q=blocks_info['quat'][i]))
            b.set_velocity(blocks_info['vel'][i])
            b.set_angular_velocity(blocks_info['ang_vel'][i])
        self.goal_coords = goal_info

        # set robot pos
        qpos = np.hstack([hand_xyz, np.array([0.04,0.04])])
        self.get_articulations()[0].set_qpos(qpos)
        if is_grasped:
            self.magic_grasp(self.blocks[0])
        else:
            # always call this no risk
            self.magic_release()
        if self.goal_visual is not None:
            self.goal_visual.set_pose(Pose(self.goal_coords[0]))
        self.draw_goal_site()


if __name__ == '__main__':

    def animate(imgs, filename="animation.mp4", _return=True, fps=10):
        if isinstance(imgs, dict):
            imgs=imgs["image"]
        print(f"animating {filename}")
        from moviepy.editor import ImageSequenceClip

        imgs=ImageSequenceClip(imgs, fps=fps)
        imgs.write_videofile(filename, fps=fps)
        if _return:
            from IPython.display import Video

            return Video(filename, embed=True)

    import gym
    student_env_name='BlockStackArm-v0'
    student_env: BlockStackFloatPandaEnv=gym.make(
        student_env_name,
        reward_mode='sparse',
        obs_mode='state_dict',
        num_blocks=1,
        goal='pick_and_place_train',
    )
    student_env.reset(seed=0)
    for _ in range(30):
        student_env.step(action=np.ones(4))
    print('truth', student_env.get_articulations()[0].get_qpos())
    state = student_env.get_state()
    student_env.reset(seed=1)
    print('new_env', student_env.get_articulations()[0].get_qpos())

    student_env.set_state(state)
    print('after setstate', student_env.get_articulations()[0].get_qpos())

    print('done')
    exit()
    imgs = []
    rgb = []
    d = []
    seg = []
    for _ in range(200):
        student_env.step(action=np.zeros(8))
        img_dict = student_env.render('state_visual')
        # rgb.append(np.clip(img_dict['rgb']*255,0,255))
        # d.append(np.repeat(np.clip(img_dict['depth']*255,0,255).astype(np.uint8), 3, axis=-1))
        # seg.append(np.repeat(img_dict['seg'].astype(np.uint8)[..., np.newaxis], 3, axis=-1))
        #pdb.set_trace()
        #
        # img = observations_to_images(img_dict)
        #
        # real_img = tile_images(img)
        # real_img[256:256+128,0:]=get_seg_visualization(np.array(real_img[256:256+128,0:]))
        seg_img = get_seg_visualization(img_dict['visual_seg'])
        imgs.append(seg_img)
        # imgs.append(img)
    #pdb.set_trace()
    animate(imgs, 'test_cam_seg.mp4',fps=20)
