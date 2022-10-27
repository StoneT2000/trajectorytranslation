from copy import deepcopy

import gym
import numpy as np
import sapien.core as sapien
from gym import spaces
from sapien.utils.viewer import Viewer
from transforms3d.quaternions import axangle2quat, quat2axangle

from tr2.envs.couchmoving.building import add_walls
from tr2.envs.sapien_env import SapienEnv
from tr2.envs.world_building import add_ball, add_target

COLORS = {
    "GREEN": [0.1662, 0.5645, 0.2693],
    "RED": [0.9350, 0.0694, 0.0956],
    "BLUE_DARK": [0.05, 0.01, 0.99],
    "BLUE": [0.45, 0.52, 0.93],
}


class CouchmovingEnv(SapienEnv):
    def __init__(
        self,
        target_radius=0.025,
        agent_type="point",
        obs_mode="dict",
        reward_type="sparse",
        fixed_env=False,
        patch_size=3,
        exclude_target_state=False,
        max_walks=3,
        walk_dist_range=(12, 25),
        world_size=50,
        skip_map_gen=False,
        random_init=False,
        target_next_chamber=True,
        force_target_next_chamber=False,
        start_from_chamber=False,
        repeat_actions=1,
        **kwargs,
    ):  
        """
        Parameters
        ----------

        target_next_chamber: bool
            Whether the target is always the next chamber / end of the maze    
        """
        self.target_radius = target_radius
        self.point_agent_radius = 0.02
        self.obs_mode = obs_mode
        self.agent_type = agent_type
        self.fixed_env = fixed_env
        self.patch_size = patch_size
        self.max_walks = max_walks
        self.walk_dist_range = walk_dist_range
        self.world_size = world_size
        self.skip_map_gen = skip_map_gen
        self.random_init = random_init
        self.target_next_chamber = target_next_chamber
        self.start_from_chamber = start_from_chamber
        self.repeat_actions = repeat_actions
        self.agent: sapien.Actor = None
        self.target = None
        self.set_env_mode(obs_mode, agent_type, reward_type)
        self.agent_config = {
            "shape": {
                "type": "couch",
                "sizes": [0.02, 0.11, 0.02],
            },
            "speed": 2500,
            "angle_speed": 100
        }
        # for larger world size 200, use this config
        if self.world_size == 200:
            self.agent_config = {
                "shape": {
                    "type": "couch",
                    "sizes": [0.02, 0.11, 0.02],
                },
                "speed": 30,
                "angle_speed": 80
            }
        self.exclude_target_state = exclude_target_state
        timestep = 1e-2
        if self.world_size == 200:
            timestep = 0.5e-2
        super().__init__(control_freq=1, timestep=timestep,  ccd=True, contact_offset=0.1, **kwargs)

        self.agent_angle = 0
        self.visuals = []
        self.walls = []
        self.path_locs = []
        self.world_map = []
        self.chamber_locs = []
        self.debug_map = False

        self.reached_chambers = 0
        self.previous_chamber = -1


    def set_env_mode(
        self, obs_mode="dict", agent_type="2D", reward_type="sparse"
    ):
        """
        set the environment observation, interaction, and reward modes
        """
        self.obs_mode = obs_mode
        self.agent_type = agent_type
        self.reward_type = reward_type
        if agent_type == "point":
            agent_info_size = 2 #x, y
        elif agent_type == "rect" or agent_type == "couch":
            agent_info_size = 3 #x, y, angle
        else:
            raise ValueError(f"agent_type provided is {agent_type}, must be point or rect")
        if self.obs_mode == "dict":
            shared = {
                "target": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=[2],
                    dtype=np.float32,
                )
            }
            if agent_type == "point":
                shared["agent"] = spaces.Box(
                    low=np.array([-np.inf, -np.inf]),
                    high=np.array([np.inf, np.inf]),
                    shape=[2],
                    dtype=np.float32,
                )
            elif agent_type == "rect":
                shared["agent"] = spaces.Box(
                    low=np.array([-np.inf, -np.inf, -1]),
                    high=np.array([np.inf, np.inf, +1]),
                    shape=[3],
                    dtype=np.float32,
                )
            elif agent_type == "couch":
                # see a local patch.
                shared["agent"] = spaces.Box(
                    low=-1,
                    high=1,
                    shape=[4 + (self.patch_size ** 2)],
                    dtype=np.float32
                )
            self.observation_space = spaces.Dict(
                shared
            )
        elif self.obs_mode == "dense":
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=[agent_info_size + 2],
            )
        if agent_type == "point":
            # dx, dy
            self.action_space = spaces.Box(
                low=-1,
                high=+1,
                shape=[2],
                dtype=np.float32,
            )
        elif agent_type == "rect":
            # dx, dy, d_angle
            self.action_space = spaces.Box(
                low=-1, high=+1, shape=[3]
            )
        elif agent_type == "couch":
            # dx, dy, d_angle
            self.action_space = spaces.Box(
                low=-1, high=+1, shape=[3]
            )

    def _build_world(self):
        ground_mtl = self._scene.create_physical_material(0. ,0., 0.0)
        self._scene.add_ground(altitude=0, material=ground_mtl)
        self.path_locs = []
        
        self.walls = []
        self._clear_world_map()
        
        self._add_actors()
    def _clear_world_map(self):
        self.world_map = []
        self.chamber_locs = []
        for x in range(self.world_size+1):
            self.world_map.append([])
            for y in range(self.world_size+1):
                self.world_map[-1].append(2) # 0 for free space, 1 for walls, 2 for void
    def _add_visual_target(self, x, y, color=[0.2, 0.6, 0.9]):
        ball_mtl = self._scene.create_physical_material(0., 0., 0.0)
        world_size_scale = 50/ self.world_size
        b = add_target(
                self._scene,
                target_id=len(self.visuals) + 10,
                physical_material=ball_mtl,
                radius=self.point_agent_radius * world_size_scale,
                color=color,
                pose=sapien.Pose([x, y, 0])
            )
        self.visuals.append(
            b
        )
    def _unscale_world(self, v):
        x = v[0]
        y = v[1]
        world_size_half = self.world_size / 2
        return np.array([
            np.round((x*world_size_half)+world_size_half),
            np.round((y*world_size_half) +world_size_half)
        ],dtype=int)
    def _scale_world(self, v):
        x = v[0]
        y = v[1]
        world_size_half = self.world_size / 2
        return np.array([(x - world_size_half)/world_size_half, (y-world_size_half) /world_size_half])
    def _gen_map(self):
        # world cartesian from [0, 0] to [100, 100], which is scaled to -1, 1 to 1, 1
        done_generating = False
        while not done_generating:
            self._clear_world_map()
            agent_xy = [self.np_random.randint(2, self.world_size - 1), self.np_random.randint(2, self.world_size - 1)]
            path_locs = [agent_xy]
            self.world_map[agent_xy[1]][agent_xy[0]] = 0
            prob = self.np_random.rand()
            move_delta = (1, 0)
            if prob < 0.25:
                move_delta = (-1, 0)
            elif prob < 0.5:
                move_delta = (1, 0)
            elif prob < 0.75:
                move_delta = (0, 1)
            else:  
                move_delta = (0, -1)
            self.movement_directions = []
            bad_path = False
            main_path_locs = set()
            while len(path_locs) <= self.max_walks:
                walk_dist = self.np_random.randint(self.walk_dist_range[0], self.walk_dist_range[1])
                prob = self.np_random.rand()
                path_loc = path_locs[-1]
                sx, sy = path_loc[0], path_loc[1]
                if prob < 0.5:
                    pot_move_delta = ((move_delta[0] + 1), (move_delta[1] + 1))
                else:
                    pot_move_delta = ((move_delta[0] - 1), (move_delta[1] - 1))
                pot_move_delta = (np.sign(pot_move_delta[0]) * (pot_move_delta[0] % 2), np.sign(pot_move_delta[1]) * (pot_move_delta[1] % 2))
                new_path_loc = [path_loc[0] + walk_dist * pot_move_delta[0], path_loc[1] + walk_dist * pot_move_delta[1]]
                if new_path_loc[0] < 2 or new_path_loc[1] < 2 or new_path_loc[0] > self.world_size - 2 or new_path_loc[1] > self.world_size - 2:
                    # don't let path be near map border or off it
                    continue

                for i in range(walk_dist):
                    sx += pot_move_delta[0] * 1
                    sy += pot_move_delta[1] * 1
                    open_c = 0
                    if i > 0:
                        # check orthogonal direction and if we are too close to another path, break
                        for k in range(-3, 3):
                            if k == 0: continue
                            if sy + k*pot_move_delta[0] < 2 or sy + k*pot_move_delta[0] > self.world_size - 2: continue
                            if sx + k*pot_move_delta[1] < 2 or sx + k*pot_move_delta[1] > self.world_size - 2: continue
                            
                            if self.world_map[sy + k*pot_move_delta[0]][sx + k*pot_move_delta[1]] == 0:
                                bad_path = True
                                break
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            if self.world_map[sy+dy][sx+dx] == 0:
                                open_c += 1
                    if self.world_map[sy][sx] == 0:
                        # print("maze map_gen warning: intersected past path, redoing")
                        bad_path = True
                        break
                if bad_path:
                    break
                move_delta = pot_move_delta
                self.movement_directions.append(move_delta)
                sx, sy = path_loc[0], path_loc[1]
                dist_to_chamber = self.np_random.randint(4, walk_dist - 4)
                for i in range(walk_dist):
                    sx += move_delta[0] * 1
                    sy += move_delta[1] * 1
                    if (sx, sy) in main_path_locs: 
                        # print("maze map_gen warning: intersected past traversed path, redoing")
                        bad_path = True
                        break
                    self.world_map[sy][sx] = 0
                    main_path_locs.add((sx, sy))
                    if i == dist_to_chamber - 1:
                        for dx in range(-1, 2):
                            for dy in range(-1, 2):
                                self.world_map[sy+dy][sx+dx] = 0
                        self.chamber_locs.append([sx, sy])
                path_locs.append(new_path_loc)
                if len(path_locs) == 2:
                    agent_xy = [agent_xy[0] + move_delta[0], agent_xy[1] + move_delta[1]]
            path_locs[0] = agent_xy
            if bad_path:
                continue
            done_generating=True
        
        simulated_walls = set()

        for y in range(self.world_size+1):
            for x in range(self.world_size+1):
                if self.world_map[y][x] == 0:
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            ny, nx = y+dy, x+dx
                            if ny < 0 or nx < 0 or ny > self.world_size or nx > self.world_size: continue
                            if self.world_map[ny][nx] == 2:
                                self.world_map[ny][nx] = 1
                                if (nx, ny) not in simulated_walls:
                                    w_x, w_y = self._scale_world([nx, ny])
                                    # add_wall(w_x, w_y, half_sizes)
                                    self.walls.append((nx, ny))
                                    simulated_walls.add((nx,ny))
        if self.debug_map:
            for y in range(self.world_size+1):
                for x in range(self.world_size+1):
                    symb = "□"
                    if self.world_map[y][x] == 2:
                        symb = "□"
                    if self.world_map[y][x] == 1:
                        symb = "▪"
                    elif self.world_map[y][x] == 0:
                        symb = " "
                    if y == agent_xy[1] and x == agent_xy[0]:
                        symb = "x"
                    if self.debug_map: print(symb,end="")
                if self.debug_map: print()
            if self.debug_map: print()

        target_xy = [path_locs[-1][0] - move_delta[0], path_locs[-1][1] - move_delta[1]]
        target_xy = self._scale_world(target_xy)
        agent_xy = self._scale_world(agent_xy)
        return agent_xy, target_xy, self.walls, path_locs
        
    def _add_actors(self):
        if not self.skip_map_gen:
            agent_xy, target_xy, walls, path_locs = self._gen_map()
            self.path_locs = path_locs
            self.walls = walls
            first_move_dir = self.movement_directions[0]
            add_walls(self, walls)
        else:
            agent_xy = [0, 0]
            target_xy = [0, 0]
            first_move_dir = (1, 0)
        # add targets and boxes
        ball_mtl = self._scene.create_physical_material(0., 0., 0.0)
        self.target = add_target(
            self._scene,
            sapien.Pose(
                [target_xy[0], target_xy[1], 0], q=axangle2quat([0, 1, 0], np.deg2rad(90))
            ),
            radius=1/self.world_size,
            physical_material=ball_mtl,
            target_id=1,
            color=COLORS["BLUE_DARK"],
        )
        if self.agent_type == "point":
            self.agent = add_ball(
                self._scene,
                ball_id=0,
                material=ball_mtl,
                half_size=self.point_agent_radius,
                density=1000.0,
                color=[0.1, 0.1, 0.1],
            )
            self.agent.set_pose(sapien.Pose([agent_xy[0],agent_xy[1], 0.02]))
        elif self.agent_type == "couch":
            builder: sapien.ActorBuilder = self._scene.create_actor_builder()
            
            thickness = .3 / self.world_size
            # box_length = 7.5
            box_length = 7 if self.world_size == 50 else 7.65
            density=100000.0
            box_height = 0.01 if self.world_size == 50 else 0.007
            builder.add_box_visual(half_size=[thickness, thickness*box_length, box_height], color=[0.1,0.1,0.1])
            builder.add_box_collision(
                half_size=[thickness, thickness*box_length, box_height], material=ball_mtl, density=density
            )
            builder.add_box_collision(
                pose=sapien.Pose([thickness*2, thickness*(box_length-1), 0]), half_size=[thickness,thickness,box_height], material=ball_mtl, density=density*box_length/2
            )
            builder.add_box_collision(
                pose=sapien.Pose([thickness*2, -thickness*(box_length-1), 0]), half_size=[thickness,thickness,box_height], material=ball_mtl, density=density*box_length/2
            )
            builder.add_box_visual(pose=sapien.Pose([thickness*2, thickness*(box_length-1), 0]), half_size=[thickness,thickness,box_height], color=[0.9,0.1,0.1])
            builder.add_box_visual(pose=sapien.Pose([thickness*2, -thickness*(box_length-1), 0]), half_size=[thickness,thickness,box_height], color=[0.9,0.1,0.1])

            ball = builder.build(name=f"ball_{0}")
            ball.set_pose(sapien.Pose(p=[0, 0, thickness], q=[1, 0, 0, 0]))
            self.agent = ball
            # if self.world_size == 50:
            self.agent.set_damping(30, 30)
            # else:
            #     self.agent.set_damping(500,500)
            cx, cy = 0, 0
            if first_move_dir == (1, 0) or first_move_dir == (-1, 0):
                if self.np_random.rand() < 0.5:
                    self.agent_angle = np.deg2rad(90)
                    cy = -thickness
                else:
                    self.agent_angle = -np.deg2rad(90)
                    cy = thickness
            else:
                if self.np_random.rand() < 0.5:
                    self.agent_angle = -np.deg2rad(0)
                    cx = -thickness
                else:
                    self.agent_angle = np.deg2rad(180)
                    cx = thickness
            self.agent_offsets = thickness, thickness
            self.agent.set_pose(sapien.Pose([agent_xy[0] + cx, agent_xy[1] + cy, thickness + 1e-3], 
                q=axangle2quat([0, 0, 1], self.agent_angle))
                )        
        elif self.agent_type == "rect":
            self.agent = add_ball(
                self._scene,
                ball_id=0,
                material=ball_mtl,
                half_size=self.agent_config["shape"]["sizes"],
                density=100000.0,
                color=[0.1, 0.1, 0.1],
            )
            self.agent.set_damping(10, 10)
            self.agent.set_pose(sapien.Pose([agent_xy[0],agent_xy[1], 0.02], 
                q=axangle2quat([0, 0, 1], np.deg2rad(self.np_random.randint(360))))
            )
        
        if self.random_init and not self.skip_map_gen:
            has_contact=True
            while has_contact:
                def therescontact():
                    for c in self._scene.get_contacts():
                        for p in c.points:
                            if np.linalg.norm(p.impulse) > 0:
                                if c.actor0.name == "ball_0" and "ground" not in c.actor1.name:
                                    print("contact")
                                    return True
                    return False
                loc_idx = self.np_random.randint(0, len(self.chamber_locs))
                if loc_idx < len(self.chamber_locs) - 1:
                    # selected internal chamber of maze and not final one,
                    if self.target_next_chamber:
                        next_chamber_pos = self._scale_world(self.chamber_locs[loc_idx + 1])
                        self.target.set_pose(sapien.Pose([next_chamber_pos[0], next_chamber_pos[1], 0]))
                pose_p = self.agent.pose.p
                pose_p[:2] = self._scale_world(self.chamber_locs[loc_idx])
                self.agent.set_pose(sapien.Pose(
                    pose_p,
                    self.agent.pose.q,
                ))
                self._scene.step()
                has_contact = therescontact()

    def get_action(self, action):
        a = np.clip(action, -1, 1)
        if self.agent_type == "point":
            mag = np.linalg.norm(a)
            if mag == 0: mag = 1e-8
            return np.array([a[0], a[1]]) / mag
        elif self.agent_type == "rect" or self.agent_type == "couch":
            # mag = np.linalg.norm(np.array([a[0], a[1]]))
            # if mag == 0: mag = 1e-8
            return np.array([a[0], a[1], a[2]])
    def check_success(self):
        dist_left = np.linalg.norm(self.target.pose.p[:2] - self.agent.pose.p[:2])
        done = False
        if dist_left < 5e-2:
            done = True
        return done
    def step(self, action):
        action = self.get_action(action)
    
        done = False
        min_dist = 5e-2
        if self.agent_type == "point":
            # the point teacher should be pretty accurate
            min_dist = 1e-2
        dist_left = np.linalg.norm(self.target.pose.p[:2] - self.agent.pose.p[:2])
        if dist_left < min_dist:
            done = True
        pose = self.agent.get_pose()
        vec, angle = quat2axangle(pose.q)
        angle = angle % (2*np.pi)
        self.agent_angle = angle * np.sign(vec[-1])
        pose_p_new = pose.p
        if self.world_size == 50:
            pose_p_new[2] = 0.008 # fix it onto the ground
        else:
            pose_p_new[2] = 0.01 # fix it onto the ground
        self.agent.set_pose(sapien.Pose(pose_p_new, pose.q))
        if self.agent_type == "rect" or self.agent_type == "couch":
            a = np.zeros(3)
            a[:2] = action[:2]
            self.agent.set_angular_velocity(np.array([0, 0, action[2]*self.agent_config["angle_speed"]]))
            self.agent.add_force_torque(a*self.agent_config["speed"], np.array([0, 0, 0]))
        elif self.agent_type == "point":
            q = axangle2quat([0, 0, 1], np.deg2rad(90))
            pose.set_q(q)
            pose.set_p(np.array([
                pose.p[0] + action[0]/100,
                pose.p[1] + action[1]/100,
                pose.p[2]
            ]))
            self.agent.set_pose(pose)
        
        for _ in range(self.repeat_actions):
            self._scene.step()
        
        obs = self._get_obs()
        reward = self._reward(obs)
        info = {}
        return obs, reward, done, info

    def _reward(self, obs):
        """
        compute the current reward signal
        """
        if self.reward_type == "sparse":
            reward = 0
        elif self.reward_type == "dense":
            if self.agent_type == "couch":
                # self.previous_chamber = -1
                within_chamber_dist = False
                for i in range(len(self.chamber_locs)):
                    c_loc = self._scale_world(self.chamber_locs[i])
                    dist_to_chamber = np.linalg.norm(c_loc - self.agent.pose.p[:2])
                    within_chamber_dist = False
                    if self.world_size == 50 and dist_to_chamber < 8e-2:
                        within_chamber_dist = True
                    elif self.world_size == 200 and dist_to_chamber < 8e-3:
                        within_chamber_dist = True
                    if within_chamber_dist:
                        self.previous_chamber = i
                        within_chamber_dist = True
                        self.reached_chambers = max(self.reached_chambers, i+1)
                # reward = self.reached_chambers
                reward = 0
                if self.previous_chamber != -1 and self.previous_chamber < len(self.chamber_locs) - 1:
                    prev_chamber = self.chamber_locs[self.previous_chamber]
                    next_chamber = self.chamber_locs[self.previous_chamber + 1]
                    prev_chamber_loc = self._scale_world(self.chamber_locs[self.previous_chamber])
                    next_chamber_loc = self._scale_world(self.chamber_locs[self.previous_chamber + 1])
                    dx = next_chamber_loc[0] - prev_chamber_loc[0]
                    dy = next_chamber_loc[1] - prev_chamber_loc[1]
                    open = dict(
                        down=self.world_map[prev_chamber[1] + 2][prev_chamber[0]] == 0,
                        right=self.world_map[prev_chamber[1]][prev_chamber[0] + 2] == 0,
                        up=self.world_map[prev_chamber[1] - 2][prev_chamber[0]] == 0,
                        left=self.world_map[prev_chamber[1]][prev_chamber[0] - 2] == 0,
                    )                    
                    angle_info = np.array([np.sin(self.agent_angle), np.cos(self.agent_angle)])
                    if dx > 0 and dy > 0:
                        if open['down'] and open['up']:
                            target = [0, 1]
                        else:
                            target = [1, 0]
                    elif dx > 0 and dy < 0:
                        if open['down'] and open['up']:
                            target = [0, 1]
                        else:
                            target = [-1, 0]
                    elif dx < 0 and dy > 0:
                        if open['down'] and open['up']:
                            target = [0, -1]
                        else:
                            target = [1, 0]
                    elif dx < 0 and dy < 0:
                        if open['down'] and open['up']:
                            target = [0, -1]
                        else:
                            target = [-1, 0]
                    angle_dist = np.linalg.norm(angle_info - np.array(target))
                    # reward = (1 - np.tanh(angle_dist*1))*5
                    reward = 0
                    # print(np.tanh(angle_dist), not within_chamber_dist)
                    if np.tanh(angle_dist) > 0.93 and not within_chamber_dist:
                        # penalize a lot for being at the wrong angle when outside of a chamber
                        reward = -1
                    else: 
                        reward = 0.2
            elif self.agent_type == "point":
                raise NotImplementedError("No dense 2 reward for pointmass agent")
        return reward

    def _get_actor_type(self, actor: sapien.Actor):
        if actor.name[:6] == "target":
            return "target"
        if actor.name[:6] == "ball_0":
            return "agent"
        if actor.name[:4] == "ball":
            return "ball"
        return "unknown"

    def _get_obs(self):
        """
        get observation. np array consisting of every ball's position that's still in play.
        """
        pose = self.agent.get_pose()
        vec, angle = quat2axangle(pose.q)
        angle = angle % (2*np.pi)
        self.agent_angle = angle * np.sign(vec[-1])
        agent_xy_raw = self.agent.pose.p[:2]
        angle_info = np.array([np.sin(self.agent_angle), np.cos(self.agent_angle)])
        if self.agent_type == "couch":
            # fix couch being incorrectly offset
            agent_xy_raw[0] += abs(self.agent_offsets[0]) * np.cos(self.agent_angle)
            agent_xy_raw[1] += abs(self.agent_offsets[1]) * np.sin(self.agent_angle)
        if self.agent_type == "point":
            obs = {
                "agent": self.agent.pose.p[:2],
                "target": self.target.pose.p[:2],
            }
        elif self.agent_type == "rect":
            obs = {
                "agent": np.hstack([self.agent.pose.p[:2], self.agent_angle]),
                "target": self.target.pose.p[:2],
            }
        elif self.agent_type == "couch":
            # generate local patch
            # agent_xy = self.agent.pose.p[:2]
            agent_xy = self._unscale_world(agent_xy_raw)
            if self.debug_map: print("### agent here: ",agent_xy, self.world_map[agent_xy[1]][agent_xy[0]])
            agent_info = np.hstack([agent_xy_raw, angle_info])
            patch = np.zeros((self.patch_size, self.patch_size))
            # patch_size must be odd
            for dy in range(0, self.patch_size):
                for dx in range(0, self.patch_size):
                    nx, ny = agent_xy[0] + dx - self.patch_size // 2, agent_xy[1] + dy - self.patch_size // 2
                    x_idx, y_idx = agent_xy[0] + dx,  agent_xy[1] + dy
                    patch[dy, dx] = self.world_map[ny][nx]
                    if self.debug_map:
                        if patch[dy, dx] == 1:
                            print("#",end="")
                        else:
                            print("o",end="")
                if self.debug_map: print()
            if self.debug_map: print()
            obs = {
                "agent": np.hstack([agent_info, patch.flatten()]),
                "target": self.target.pose.p[:2],
            }
        if self.obs_mode == "dict":
            return obs
        elif self.obs_mode == "dense":
            if self.exclude_target_state:
                return obs["agent"]
            else:
                return np.hstack(
                    [obs["agent"], obs["target"]]
                )

    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)
        else:
            if self.fixed_env:
                self.seed(self.seed_val)
        self.visuals = []
        self.reconfigure()
        self._scene.step()
        obs = self._get_obs()
        self.reached_chambers = 0
        self.previous_chamber = -1
        return obs
    def setup_camera(self):
        near, far = 0.1, 100
        width, height = 1024, 1024
        camera_mount_actor = self._scene.create_actor_builder().build_kinematic()
        self.camera = self._scene.add_mounted_camera(
            name="camera",
            actor=camera_mount_actor,
            pose=sapien.Pose(),  # relative to the mounted actor
            width=width,
            height=height,
            fovx=0,
            fovy=np.pi / 3,
            near=near,
            far=far,
        )
        q = [0.7073883, 0, 0.7068252, 0]
        camera_mount_actor.set_pose(sapien.Pose(p=[0, 0, 2.0], q=q))

        # uncomment for high quality render
        # near, far = 0.1, 100
        # width, height = 1920, 1280
        # camera_mount_actor = self._scene.create_actor_builder().build_kinematic()
        # self.camera = self._scene.add_mounted_camera(
        #     name="camera",
        #     actor=camera_mount_actor,
        #     pose=sapien.Pose(),  # relative to the mounted actor
        #     width=width,
        #     height=height,
        #     fovx=0,
        #     fovy=np.pi / 3,
        #     near=near,
        #     far=far,
        # )
        # q = [0.7073883, 0, 0.7068252, 0]
        # camera_mount_actor.set_pose(sapien.Pose(p=[0.4, 0, 1.5], q=q))

    def _setup_viewer(self):
        if self.viewer is None:
            print("OPEN VIEWER")
            self.viewer = Viewer(self._renderer)
        self._setup_lighting()
        self.viewer.set_scene(self._scene)
        self.viewer.set_camera_rpy(0, -np.pi/2, -np.pi/2)
        self.viewer.set_camera_xyz(0, 0, 1)
    def _get_state(self):
        walls = self.walls
        self._get_obs()
        agent_info = dict(q=self.agent.pose.q, p=self.agent.pose.p)
        return dict(
            walls=walls,
            agent_info=agent_info,
            target=self.target.pose.p[:2],
            world_map=self.world_map,
            path_locs=self.path_locs,
            world_size=self.world_size,
            chamber_locs=self.chamber_locs
        )
    def _set_state(self, state):
        if self.agent_type == "rect":
            agent = state["agent"]
            angle = state["agent_angle"]
            self.agent.set_pose(sapien.Pose(
                [agent[0], agent[1], 0.02],
                axangle2quat([0, 0, 1], np.deg2rad(angle))
            ))
            self.agent_angle = np.deg2rad(angle)
        elif self.agent_type == "point":
            agent = state["agent"]
            self.agent.set_pose(sapien.Pose(
                [agent[0], agent[0], 0.02],
                axangle2quat([0, 0, 1], np.deg2rad(90))
            ))
        elif self.agent_type == "couch":
            self.walls = state["walls"]
            target_xy = state["target"]
            self.target.set_pose(sapien.Pose([target_xy[0], target_xy[1], 0]))
            agent_p = state["agent_info"]["p"]
            agent_q = state["agent_info"]["q"]
            self.agent.set_pose(sapien.Pose(
                agent_p,
                agent_q
            ))
            self.path_locs = state["path_locs"]
            add_walls(self, self.walls)
            self.world_size = state["world_size"]
            self.world_map = state["world_map"]
            self.chamber_locs = state["chamber_locs"]
            if self.start_from_chamber:
                pose_p = self.agent.pose.p
                c_loc = self.chamber_locs[0]
                if self.random_init:
                    c_loc = self.chamber_locs[self.np_random.randint(0, len(self.chamber_locs))]
                pose_p[:2] = self._scale_world(c_loc)
                self.agent.set_pose(sapien.Pose(
                    pose_p,
                    self.agent.pose.q,
                ))

        target = state["target"]
        self.target.set_pose(sapien.Pose([target[0], target[1], 0]))