import gym
import numpy as np
import sapien.core as sapien
from gym import spaces
from sapien.utils.viewer import Viewer
from transforms3d.quaternions import axangle2quat

import tr2.envs.boxpusher.task_config as task_config
from tr2.envs.sapien_env import SapienEnv
from tr2.envs.world_building import add_ball, add_target

COLORS = {
    "GREEN": [0.1662, 0.5645, 0.2693],
    "RED": [0.9350, 0.0694, 0.0956],
    "BLUE_DARK": [0.05, 0.01, 0.99],
    "BLUE": [0.45, 0.52, 0.93],
}


class BoxPusherEnv(SapienEnv):
    def __init__(
        self,
        balls=1,
        ball_radius=0.05,
        controlled_ball_radius=0.05,
        target_radius=0.05,
        control_type="2D",
        magic_control=False,
        obs_mode="dict",
        reward_type="sparse",
        fixed_env=False,
        target_plans=None,  # order of which balls and targets to move around to
        target_locs="any",  # locations of potential targets
        disable_ball_removal=False,
        task='train',
        speed_factor=1,
        **kwargs,
    ):
        """
        control_type : str
            the type of control allowed. If '2D', action space is 2D, first value is direction (0, 1, 2, 3) for (+x, +y, -x, -y), 2nd value is magnitude
            If '2D-continuous', actions are of form [x, y] and the performed action is the larger of the two directions in magnitude
        magic_control : bool
            if True, controlled ball upon touching a second ball will be connected to that second ball, any other balls touched will not connect until second ball is
            pushed to target
        obs_mode : str
            if 'dict', obs space is a nice formatted dictionary
            if 'dense', obs space is a dense matrix of values
        reward_type : str
            if 'dense', give crafted dense reward
            if 'sparse', give success signal reward = number of balls that reached targets so far
        fixed_env : bool
            if True, then reseting this env resets to the same initial state it started with.
            if False, each reset changes the state unless given a new seed.

        target_locs : ndarray (N, 2) or str
            if "any", will randomly generate a single target on the map in square [-0.75, -0.75] x [0.75, 0.75]
        """
        self.num_balls = balls
        self.speed_factor = speed_factor
        self.controlled_ball_radius = controlled_ball_radius
        self.ball_radius = ball_radius
        self.target_radius = target_radius
        self.magic_control = magic_control
        self.obs_mode = obs_mode
        self.control_type = control_type
        self.reward_type = reward_type
        self.fixed_env = fixed_env
        self.disable_ball_removal = disable_ball_removal
        self.task = task
        self.silo_gripper_embodiment = True
        if self.task == "silo_obstacle_push":
            # WARNING, overriding 
            # self.ball_radius = 0.03
            if self.silo_gripper_embodiment:
                self.controlled_ball_radius = 0.025
            # self.target_radius = 0.03
            pass
        self.agent_target_path_actors =[]
        self.obstacle_actors = []


        self.balls: list[sapien.Actor] = []
        self.target_actors: list[sapien.Actor] = []
        self.balls_meta = []
        self.target = None
        self.target_locs = target_locs
        self.target_loc_generation_type = "corners"
        if self.target_locs == "any":
            self.target_locs = []
            self.target_loc_generation_type = "any"

        self.set_env_mode(obs_mode, control_type, reward_type)

        self.target_plans = []
        self.fixed_target_plans = target_plans

        super().__init__(control_freq=1, timestep=1e-2, **kwargs)
        self._build_world()

        self._shadow_id = 0
        self._shadow_opacity = 0.8

    def set_env_mode(
        self, obs_mode="dict", control_type="2D-continuous", reward_type="sparse"
    ):
        """
        set the environment observation, interaction, and reward modes
        """
        self.obs_mode = obs_mode
        self.control_type = control_type
        self.reward_type = reward_type
        if self.obs_mode == "dict":
            shared = {
                "agent_ball": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=[2],
                    dtype=np.float32,
                ),
                "target": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=[2],
                    dtype=np.float32,
                ),
                "target_ball": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=[2],
                    dtype=np.float32,
                ),
                "target_locs": spaces.Box(
                    low=-np.inf, high=np.inf, shape=[4, 2], dtype=np.float32
                ),
                "balls": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=[self.num_balls + 1, 2],
                    dtype=np.float32,
                ),
                "active_mask": spaces.Box(
                    low=0, high=1, shape=[self.num_balls + 1], dtype=np.int32
                ),
            }
            if self.magic_control:
                shared["controlled_ball_attached"] = spaces.Discrete(2)
            self.observation_space = spaces.Dict(
                shared
            )
        elif self.obs_mode == "dense":
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=[6],
            )
        if control_type == "2D":
            self.action_space = spaces.Box(
                low=np.array([0, -2e-2]),
                high=np.array([4, 2e-2]),
                shape=[2],
                dtype=np.float32,
            )
        elif control_type == "2D-continuous":
            self.action_space = spaces.Box(low=-1, high=1, shape=[2])
        else:
            raise ValueError(f"{control_type} is not supported")

    def _generate_target_plans(self):
        # generate target_plans if it doesn't exist
        self.target_plans = []
        if self.fixed_target_plans is not None:
            self.target_plans = self.fixed_target_plans
            return
        ball_ids = []
        target_ids = []
        for i in range(self.num_balls):
            ball_ids.append(i + 1)
        for i in range(self.num_balls):
            target_ids.append(
                self.np_random.choice(np.arange(0, len(self.target_locs)))
            )
        self.np_random.shuffle(ball_ids)
        for i in range(self.num_balls):
            self.target_plans.append(
                dict(ball_target=ball_ids[i], target=target_ids[i])
            )

    def _build_world(self):
        ground_mtl = self._scene.create_physical_material(1. ,1., 0.05)
        self._scene.add_ground(altitude=0, material=ground_mtl)

    def _add_actors(self):
        # add targets and boxes
        # ball_mtl = self._scene.create_physical_material(1., 1., 0.0)
        ball_mtl = self._scene.create_physical_material(0., 0., 0.0)
        dx_range = (0, 0)
        dy_range = (0, 0)
        for target_id, target in enumerate(self.target_locs):
            dx_range = (min(dx_range[0], target[0]), max(dx_range[1], target[1]))
            dy_range = (min(dy_range[0], target[0]), max(dy_range[1], target[1]))
            target_a = add_target(
                self._scene,
                sapien.Pose(
                    [target[0], target[1], 0.05], q=axangle2quat([0, 1, 0], np.deg2rad(90))
                ),
                radius=self.target_radius,
                physical_material=ball_mtl,
                target_id=target_id,
                color=COLORS["BLUE"],
            )
            self.target_actors.append(target_a)

        if self.task == 'obstacle' or self.task == 'long_maze' or self.task == 'train':
            ball_positions = self._generate_ball_positions(
                n=self.num_balls + 1,
                half_width=(1.5) / 2,
                half_height=(1.5) / 2,
            )
        elif self.task == 'silo_obstacle_push_demo' or self.task == 'silo_obstacle_push':
            assert (self.num_balls) == 1
            target_ball = np.array([0,0])
            length = np.sqrt(self.np_random.uniform(0, 0.02))
            angle = np.pi * self.np_random.uniform(0, 2)

            target_ball[0] = length * np.cos(angle)
            target_ball[1] = length * np.sin(angle)
            agent_ball = target_ball.copy()
            # from paper, robot arm is initialized right behind the block
            agent_ball = np.array([target_ball[0], target_ball[1] - 0.2])
            ball_positions = [agent_ball, target_ball]

        agent_xy = ball_positions[0]
        for ball_id, ball_pos in enumerate(ball_positions):
            half_size = self.ball_radius
            density = 1000.0
            c = COLORS["RED"]
            if ball_id == 0:
                half_size = self.controlled_ball_radius
                density = 100000.0
                if self.task == 'silo_obstacle_push':
                    density = 1000.0
                    if self.silo_gripper_embodiment:
                        density = 1000.0 * 4
                        half_size = [half_size, half_size, 0.05]
                c = [0.1, 0.1, 0.1]
            ball = add_ball(
                self._scene,
                ball_id=ball_id,
                material=ball_mtl,
                half_size=half_size,
                density=density,
                color=c,
            )
            if ball_id == 0:
                ball.set_damping(30, 3000)
            ball.set_pose(sapien.Pose([ball_pos[0], ball_pos[1], 0.05]))
                    

            self.balls.append(ball)
            self.balls_meta.append({"done": False})
            if ball_id != 0:
                ball.set_damping(linear=30., angular=3000.)

        self.controlled_ball_attached = None
        if self.task == 'obstacle':
            task_config.obstacle(self, n=2, half_size=half_size)
        elif self.task == 'long_maze':
            task_config.long_maze(self)
        elif self.task == 'silo_obstacle_push':
            task_config.silo_obstacle_push(self, n=3, half_size=half_size)
        elif self.task == 'silo_obstacle_push_demo':
            task_config.silo_obstacle_push(self, n=3, half_size=half_size, disable_collision=True)
    def _add_shadow(self, balls, colors):
        """
        draws shadowed balls for viewing purposes
        """
        ball_mtl = self._scene.create_physical_material(1., 1., 0.0)

        actors = []
        for ball_pos, c in zip(balls, colors):

            ball = add_ball(
                self._scene,
                ball_id=self._shadow_id,
                material=ball_mtl,
                half_size=self.ball_radius,
                density=1000,
                color=c,
                id=f"shadow_{self._shadow_id}",
            )
            for r in ball.get_visual_bodies():
                r.set_visibility(self._shadow_opacity)
            actors.append(ball)
            ball.set_pose(sapien.Pose([ball_pos[0], ball_pos[1], 0.05]))
            ball.set_damping(linear=72., angular=1.)
            self._shadow_id += 1
        return actors

    def _update_actors(self, actors, positions):
        """
        draws shadowed balls for viewing purposes
        """
        for a, p in zip(actors, positions):
            for r in a.get_visual_bodies():
                r.set_visibility(self._shadow_opacity)
            a.set_pose(sapien.Pose([p[0], p[1], 0.05]))

    def set_target(self, ball_target=None, target=None):
        """
        Set a goal in this env so it can give an appropriate reward, also disables other targets
        """

        if ball_target is None or target is None:
            balls_done = 0
            for i, meta in enumerate(self.balls_meta[1:]):
                if meta["done"]:
                    balls_done += 1
            self.ball_target = self.target_plans[balls_done]["ball_target"]
            self.target = self.target_plans[balls_done]["target"]
        else:
            self.ball_target = ball_target
            self.target = target
        #     ball_target = self.np_random.choice(choices)
        # if target is None:
        #     target = self.np_random.randint(0, len(self.target_actors))

        # color the other balls red
        # ball_mtl = self._scene.create_physical_material(1., 1., 0.0)
        ball_mtl = self._scene.create_physical_material(0., 0., 0.0)
        for i in range(1, len(self.balls)):

            if self.balls_meta[i]["done"]:
                continue
            density = 1000
            if i == self.ball_target:
                old_pose = self.balls[i].pose
                self._scene.remove_actor(self.balls[i])
                self.balls[i] = add_ball(
                    self._scene,
                    density=density,
                    material=ball_mtl,
                    half_size=self.ball_radius,
                    ball_id=i,
                    color=COLORS["GREEN"],
                )
                self.balls[i].set_pose(old_pose)
                self.balls[i].set_damping(linear=30., angular=3000)
            else:

                old_pose = self.balls[i].pose
                self._scene.remove_actor(self.balls[i])
                self.balls[i] = add_ball(
                    self._scene,
                    density=density,
                    material=ball_mtl,
                    half_size=self.ball_radius,
                    ball_id=i,
                    color=COLORS["RED"],
                )
                self.balls[i].set_pose(old_pose)
                self.balls[i].set_damping(linear=30., angular=3000.)

        for i in range(len(self.target_actors)):
            if i == self.target:
                old_pose = self.target_actors[i].pose
                self._scene.remove_actor(self.target_actors[i])
                self.target_actors[i] = add_target(
                    self._scene,
                    old_pose,
                    physical_material=ball_mtl,
                    target_id=i,
                    radius=self.target_radius,
                    color=COLORS["BLUE_DARK"],
                )
                self.target_actors[i].set_pose(old_pose)
            else:
                old_pose = self.target_actors[i].pose
                self._scene.remove_actor(self.target_actors[i])
                self.target_actors[i] = add_target(
                    self._scene,
                    old_pose,
                    physical_material=ball_mtl,
                    target_id=i,
                    radius=self.target_radius,
                    color=COLORS["BLUE"],
                )
                self.target_actors[i].set_pose(old_pose)

    def get_action(self, action):
        if self.control_type == "2D":
            m = np.clip(action[1], -1, 1)
            direction = np.floor(action[0]).astype(int)
            if direction == 0:
                a = [m, 0, 0]
            elif direction == 1:
                a = [0, m, 0]
            elif direction == 2:
                a = [-m, 0, 0]
            elif direction == 3:
                a = [0, -m, 0]
            return np.array(a)
        elif self.control_type == "2D-continuous":
            clipped_a = np.zeros(3)
            clipped_a[:2] = np.clip(action[:2], -1, 1)
            return clipped_a

    def step(self, action=np.array([0, 0.0]), allow_finish=True):
        """
        allow_finish = false means balls dont disappear
        """
        controlled_ball = 0
        q = axangle2quat([0, 1, 0], np.deg2rad(90))
        clipped_a = self.get_action(action) * 4
        if self.task == 'silo_obstacle_push':
            clipped_a = clipped_a * 2
        
        if self.magic_control:
            agent_delta = self.balls[controlled_ball].pose.p.copy()
            agent_pose = self.balls[controlled_ball].pose
            self.balls[controlled_ball].set_pose(sapien.Pose(p=agent_pose.p + clipped_a * 1e-2))
        else:
            force_scale = 10000.0
            if self.task == 'silo_obstacle_push':
                force_scale = 100.0
            self.balls[controlled_ball].add_force_at_point(clipped_a*force_scale*self.speed_factor, np.array([0,0,0]))
        if self.magic_control:
            if self.controlled_ball_attached is not None:
                agent_delta = agent_delta - self.balls[controlled_ball].pose.p
                # manually update pose here, so held ball follows controlled ball
                attached_ball = self.balls[self.controlled_ball_attached]
                new_pose = sapien.Pose(attached_ball.pose.p - agent_delta,q=q)
                attached_ball.set_pose(new_pose)

        # remove balls that hit a target and use magic control
        contacts = self._scene.get_contacts()
        for c in contacts:
            if self.magic_control:
                touched_ball = None
                if (
                    self._get_actor_type(c.actor0) == "ball"
                    and self._get_actor_type(c.actor1) == "agent"
                ):
                    touched_ball = c.actor0
                elif (
                    self._get_actor_type(c.actor1) == "ball"
                    and self._get_actor_type(c.actor0) == "agent"
                ):
                    touched_ball = c.actor1
                if touched_ball is not None:
                    ball_id = int(touched_ball.name.split("_")[1])
                    dist = np.linalg.norm(self.balls[ball_id].pose.p[:2] - self.balls[0].pose.p[:2])
                    if not self.balls_meta[ball_id]["done"] and dist < self.ball_radius * 3:
                        if self.controlled_ball_attached is None:
                            self.controlled_ball_attached = ball_id
            # if (
            #     self._get_actor_type(c.actor0) == "obstacle"
            #     and self._get_actor_type(c.actor1) == "agent"
            # ) or (
            #     self._get_actor_type(c.actor1) == "obstacle"
            #     and self._get_actor_type(c.actor0) == "agent"
            # ):
            #     for p in c.points:
            #         if np.linalg.norm(p.impulse) > 0:
            #             self.failed = True
        
        finished_box = False
        if allow_finish and not self.failed:
            for target_id, target in enumerate(self.target_actors):
                if self.target == target_id:
                    dist = np.linalg.norm(
                        target.pose.p[:2] - self.balls[self.ball_target].pose.p[:2]
                    )
                    if self.magic_control:
                        max_dist = 0.1
                    else:
                        max_dist = self.target_radius + self.ball_radius 
                    if dist < max_dist:
                        # self._scene.remove_actor(self.balls[self.ball_target])
                        self.balls_meta[self.ball_target]["done"] = True
                        if self.controlled_ball_attached is not None:
                            self.controlled_ball_attached = None
                        finished_box = True
        successes = 0
        for meta in self.balls_meta:
            if meta["done"]:
                successes += 1

        done = False
        if successes == len(self.balls) - 1 and not self.failed:
            done = True
        if done == False and finished_box:
            self.set_target()

        self._scene.step()

        if self.task == 'silo_obstacle_push':
            q = axangle2quat([0, 0, 1], np.deg2rad(90))
            pose = self.balls[0].get_pose()
            pose.set_q(q)
            self.balls[0].set_pose(pose)
            # in silo task, target box's rotation is not fixed.
        else:
            for idx, actor in enumerate(self.balls[:]):
                pose = actor.get_pose()
                pose.set_q(q)

                actor.set_pose(pose)
        
        obs = self._get_obs()

        reward = self._reward(obs)
        info = {}
        info["successes"] = successes
        info["failed"] = self.failed
        return obs, reward, done, info

    def _reward(self, obs):
        """
        compute the current reward signal
        """
        successes = 0
        reward = 0
        for meta in self.balls_meta:
            if meta["done"]:
                successes += 1
        if self.reward_type == "sparse":
            reward = successes
        elif self.reward_type == "test":
            xy = self.balls[0].pose.p[:2]
            return np.linalg.norm(xy)
        elif self.reward_type == "dense":
            assert len(self.balls) == 2  # only support 2 at the moment
            reward = 0
            if not self.balls_meta[self.ball_target]["done"]:
                obs = np.hstack(
                    [self.target_locs[self.target].flatten(), self.balls[0].pose.p[:2], self.balls[self.ball_target].pose.p[:2]]
                )
                reward = 0
                dist = np.linalg.norm(obs[0:2] - obs[4:6])
                control_dist = np.linalg.norm(obs[2:4] - obs[4:6]) / 2
                reward -= dist
                reward -= control_dist
                reward /= 10
        return reward

    def _get_actor_type(self, actor: sapien.Actor):
        if "obstacle" in actor.name:
            return "obstacle"
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
        positions = []
        mask = np.ones(len(self.balls))
        for i, (ball, ball_meta) in enumerate(zip(self.balls, self.balls_meta)):
            # if ball_meta["done"]:
                # mask[i] = 0
                # positions.append(np.zeros(2))
            # else:
            positions.append(ball.pose.p[:2])
        if self.obs_mode == "dict":
            obs = {
                "target_locs": self.target_locs,
                "balls": np.vstack(positions),
                "active_mask": mask,
                "agent_ball": positions[0],
                "target_ball": positions[self.ball_target],
                "target": self.target_locs[self.target],
            }
            if self.task == 'obstacle':
                obs["obstacles"] = np.array([x.pose.p[:2] for x in self.obstacle_actors])
            if self.magic_control:
                # add annotation of when we contact to make generating trajectory easier
                obs["controlled_ball_attached"] = self.controlled_ball_attached is not None
            return obs
        elif self.obs_mode == "dense":
            return np.hstack(
                [self.target_locs[self.target].flatten(), positions[0].flatten(), positions[self.ball_target].flatten()]
            )

    def clear_out_board(self):
        for ball_id, ball in enumerate(self.balls):
            # if not self.balls_meta[ball_id]["done"]:
            self._scene.remove_actor(ball)
        for target in self.target_actors:
            self._scene.remove_actor(target)
        for a in self.obstacle_actors:
            self._scene.remove_actor(a)
        self.target_actors = []
        self.balls = []
        self.balls_meta = []
        self.obstacle_actors = []

    def reset(self, seed=None):
        self.failed = False
        if seed is not None:
            self.seed(seed)
        else:
            if self.fixed_env:
                self.seed(self.seed_val)
        self.clear_out_board()
        if self.target_loc_generation_type == "any":
            if self.task == 'train':
                self.target_locs = self.np_random.rand(1, 2) * 1.5 - 1.5 / 2
            elif self.task == 'obstacle':
                self.target_locs = self.np_random.rand(1, 2) * 1.5 - 1.5 / 2
            elif self.task == 'long_maze':
                self.target_locs = self.np_random.rand(1, 2) * 10 - 5
            elif self.task == 'silo_obstacle_push' or self.task == 'silo_obstacle_push_demo':
                scale = 0.05 / 0.02
                self.target_locs = np.array([[0, 0.25]]) * scale + self.np_random.rand(1, 2) * 0.2 * scale - 0.1 * scale # from the paper directly, scaled up to our map size.
                # possible bug in paper? But force position of goal to be past obstacles and not inside it
                self.target_locs[0][1] = np.max([self.target_locs[0][1], 0.6])
        self._add_actors()
        self._scene.step()
        self._generate_target_plans()
        self.set_target()
        return self._get_obs()

    def _setup_viewer(self):
        self._setup_lighting()
        self.viewer = Viewer(self._renderer)

        self.viewer.set_scene(self._scene)
        # self.viewer.set_camera_xyz(0, 0, 20)
        # self.viewer.set_camera_rpy(-np.pi / 2, np.pi / 2, 0)  # birdeye
        self.viewer.set_camera_rpy(0, -np.pi/2, -np.pi/2)
        self.viewer.set_camera_xyz(0, 0, 1.5)
        # self.viewer.window.set_camera_parameters(near=0.1, far=100, fovy=np.pi)

    def _generate_ball_positions(
        self, half_width=1, half_height=2, n=10, ball_radius=0.05
    ):
        ball_positions = []
        half_width -= 0.2
        half_height -= 0.2
        ct = 0
        repeats = 0
        while True:
            repeats += 1
            x = self.np_random.rand() * half_width * 2 - half_width
            y = self.np_random.rand() * half_height * 2 - half_height
            # ensure no collisions with past balls
            pos = np.array([x, y])
            redo = False
            for old_pos in ball_positions:
                # else:
                if np.linalg.norm(old_pos - pos) < self.ball_radius * 2 + 1e-2:
                    redo = True
                    break
            for old_pos in self.target_locs:
                if self.task == 'obstacle':
                    if np.abs(old_pos[0] - pos[0]) < 4e-1 or np.abs(old_pos[1] - pos[1]) < 4e-1:
                        redo=True
                        break
                else:
                    if (
                        np.linalg.norm(old_pos - pos)
                        < self.ball_radius * 2 + self.target_radius + 1e-2
                    ):
                        redo = True
                        break
            if not redo:
                ball_positions.append(pos)
                ct += 1
            if ct >= n:
                break

        return ball_positions

    def _unpack_dense_state(self, state):
        targets_end = len(self.target_locs) * 2
        balls_end = len(self.balls) * 2
        return dict(
            target_locs=state[:targets_end].reshape(len(self.target_locs), 2),
            balls=state[targets_end : targets_end + balls_end].reshape(
                len(self.balls), 2
            ),
            active_mask=state[targets_end + balls_end :],
        )
    def _get_state(self):
        positions = []
        for i, (ball, ball_meta) in enumerate(zip(self.balls, self.balls_meta)):
            positions.append(ball.pose.p[:2])
        positions = np.hstack(positions)
        data = np.hstack(
            [self.target_locs[self.target], positions]
        )
        if self.task == 'obstacle':
            return dict(
                positions=data,
                obstacles=np.vstack([x.pose.p[:2] for x in self.obstacle_actors])
            )
        
        else:
            return data
        
    def _set_state(self, state):
        """
        for one ball only
        """
        if self.task == 'obstacle' :
            positions = state["positions"]
            for actor, actor_pos in zip(self.obstacle_actors, state["obstacles"]):
                actor.set_pose(
                    sapien.Pose([actor_pos[0], actor_pos[1], 0.05], actor.get_pose().q)
                )
        else:
            positions = state
        for actor, actor_pos in zip(self.balls, positions[2:].reshape(2, 2)):
            actor.set_pose(
                sapien.Pose([actor_pos[0], actor_pos[1], 0.05], actor.get_pose().q)
            )
        self.target_actors[0].set_pose(sapien.Pose([positions[0], positions[1], 0]))
        self.target_locs[0] = np.array(positions[:2])
        self.set_target(1, 0)
    def render(self, mode="human"):
        from mani_skill2.utils.sapien_utils import look_at

        # agent_xy = self.balls[0].pose.p[:2]
        look_at((0,0,1), (1,1, 0))
        return super().render(mode)
    def set_state(self, state):
        # state
        self.clear_out_board()
        self._add_actors()
        if "controlled_ball_attached" in state.keys():
            self.controlled_ball_attached = state["controlled_ball_attached"]
        if "target_locs" in state.keys():
            self.target_locs = state["target_locs"]
            assert len(self.target_locs) == len(self.target_actors)
            for actor, actor_pos in zip(self.target_actors, self.target_locs):
                actor.set_pose(
                    sapien.Pose([actor_pos[0], actor_pos[1], 0], actor.get_pose().q)
                )
        assert len(self.balls) == len(state["balls"])
        for actor, actor_pos in zip(self.balls, state["balls"]):
            actor.set_pose(
                sapien.Pose([actor_pos[0], actor_pos[1], 0], actor.get_pose().q)
            )

        assert len(self.balls_meta) == len(state["active_mask"])
        for i, m in enumerate(state["active_mask"]):
            self.balls_meta[i]["done"] = ~np.isclose(m, 1.0)
            if self.balls_meta[i]["done"]:
                # if done, remove
                self._scene.remove_actor(self.balls[i])
        self._scene.step()
        self.set_target()


gym.register(
    id="BoxPusher-v0",
    entry_point="skilltranslation.envs.boxpusher.env:BoxPusherEnv",
)
gym.register(
    id="BoxPusherFinite-v0",
    entry_point="skilltranslation.envs.boxpusher.env:BoxPusherEnv",
    max_episode_steps=100,
)
