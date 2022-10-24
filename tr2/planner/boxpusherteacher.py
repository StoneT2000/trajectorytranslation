import numpy as np
from gym.utils import seeding

from tr2.planner.base import HighLevelPlanner


class BoxPusherPlanner(HighLevelPlanner):
    def __init__(self, controlled_ball_radius=0.05, ball_radius=0.05, max_speed=1) -> None:
        super().__init__()
        self.ball_id = None
        self.target_id = None
        self.RADIUS = controlled_ball_radius + ball_radius
        self.max_speed = max_speed

class BoxPusherReacherPlanner(BoxPusherPlanner):
    """
    simply reach
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.counter = 0
        self.next_flip = 20 #self.np_random.randint(10, 40)
        self.direction = 0 #self.np_random.randint(0, 4)
        self.np_random = None
    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        # self.next_flip = self.np_random.randint(10, 40)
        # self.direction = self.np_random.randint(0, 4)
    def act(self, obs):
        if self.counter >= self.next_flip:
            new_direction =  self.np_random.randint(0, 2)
            if new_direction == 0:
                self.direction = (self.direction + 1) % 4
            else:
                self.direction = (self.direction + 4 - 1) % 4
            self.next_flip = self.counter+20# self.np_random.randint(10, 40) + self.counter
        self.counter += 1
        return np.array([self.direction, self.max_speed]), False


class BoxPusherTaskPlanner(BoxPusherPlanner):
    """
    Solve the actual task
    """
    def __init__(self, smart_mode=True, replan_threshold=3e-1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.smart_mode = smart_mode
        self.replan_threshold = replan_threshold
        self.v = 0
    def set_type(self, v):
        self.v = v
    def done(self, state, obs):
        return False
    def need_replan(self, state, obs, orig_trajectory_observations, env):
        if len(obs["observation"].shape) == 2:
            dense_obs = obs["observation"][-1, :]
        else:
            dense_obs = obs["observation"]
        if isinstance(state, dict):
            positions = state["positions"]
        else:
            positions = state
        agent_xy = positions[2:4]
        ball_xy = positions[4:6]
        dist_to_goal = np.linalg.norm(positions[:2] - ball_xy)
        teacher_agent = orig_trajectory_observations[:, :2]
        teacher_world = orig_trajectory_observations[:, 2:4]
        agent_diff = np.linalg.norm(
            teacher_agent - agent_xy, axis=1
        )
        world_diff = np.linalg.norm(
            teacher_world - ball_xy, axis=1
        )

        trajectory_diffs = (agent_diff / 10 + world_diff)
        # import pdb; pdb.set_trace()
        replan=False
        # print(trajectory_diffs.min())
        if trajectory_diffs.min() > self.replan_threshold and dist_to_goal > 2e-1:
            # print("strayed away, replanning")
            replan=True
        return replan
    def act(self, obs):
        # balls = obs["balls"]
        agent_state = obs["agent_ball"]
        ball_state = obs["target_ball"] # chosen by env
        # ball_state = balls[ball_id]
        target_state = obs["target"] # chosen by env
        obstacles = None
        if "obstacles" in obs:
            obstacles = obs["obstacles"]

        # agent_state = balls[0]
        dx = ball_state[0] - agent_state[0]
        dy = ball_state[1] - agent_state[1]
        ball_target_d = ball_state - target_state

        action = np.zeros(2)

        close_x_first = True

        if (
            "controlled_ball_attached" in obs
            and not obs["controlled_ball_attached"]
        ):
            if self.v < 2:
                close_x_first = True
            else:
                close_x_first = False
            if obstacles is not None:
                dx_obstacle = 999
                dy_obstacle = 999
                for obstacle in obstacles:
                    dx_obstacle = min(dx_obstacle, abs(ball_state[0] - obstacle[0]))
                    dy_obstacle = min(dy_obstacle, abs(ball_state[1] - obstacle[1]))
                if dx_obstacle < dy_obstacle:
                    close_x_first = False

            if close_x_first:
                if abs(dx) > 3e-2:
                    action[0] = np.sign(dx)
                elif abs(dy) > 3e-2:
                    action[1] = np.sign(dy)
            else:
                if abs(dy) > 3e-2:
                    action[1] = np.sign(dy)
                elif abs(dx) > 3e-2:
                    action[0] = np.sign(dx)
        else:
            # # in smart mode, will close x/y depending on orientation of block to make student easier
            # if self.smart_mode:
            #     close_x = True
            #     cx = abs(target_state[0] - ball_state[0]) - abs(target_state[0] - agent_state[0])
            #     cy = abs(target_state[1] - ball_state[1]) - abs(target_state[1] - agent_state[1])
            #     if cx < cy:
            #         close_x = True
            #     else:
            #         close_x = False
            #     if (close_x):
            #         if abs(ball_target_d[0]) > 3e-2:
            #             action[0] = np.sign(target_state[0] - ball_state[0])
            #         elif abs(ball_target_d[1]) > 3e-2:
            #             action[1] = np.sign(target_state[1] - ball_state[1])
            #     else:
            #         if abs(ball_target_d[1]) > 3e-2:
            #             action[1] = np.sign(target_state[1] - ball_state[1])
            #         elif abs(ball_target_d[0]) > 3e-2:
            #             action[0] = np.sign(target_state[0] - ball_state[0])
            # else:
            if (self.v > 0 and self.v < 3): close_x_first = True
            else: close_x_first = False
            if obstacles is not None:
                dx_obstacle = 999
                close_x_first=True
                dy_obstacle = 999
                for obstacle in obstacles:
                    dx_obstacle = min(dx_obstacle, abs(target_state[0] - obstacle[0]))
                    dy_obstacle = min(dy_obstacle, abs(target_state[1] - obstacle[1]))
                if dx_obstacle < dy_obstacle:
                    close_x_first = False
            if close_x_first:
                if abs(ball_target_d[0]) > 3e-2:
                    action[0] = np.sign(target_state[0] - ball_state[0])
                elif abs(ball_target_d[1]) > 3e-2:
                    action[1] = np.sign(target_state[1] - ball_state[1])
            else:
                if abs(ball_target_d[1]) > 3e-2:
                    action[1] = np.sign(target_state[1] - ball_state[1])
                elif abs(ball_target_d[0]) > 3e-2:
                    action[0] = np.sign(target_state[0] - ball_state[0])

            
        direction = 0
        if action[1] == 1:
            direction = 1
        elif action[0] == -1:
            direction = 2
        elif action[1] == -1:
            direction = 3
        dist = np.linalg.norm(target_state- ball_state)
        cutoff =  dist < 0.1
        return np.array([direction, self.max_speed]), cutoff


class BoxPusherOneDirection(BoxPusherPlanner):
    def __init__(self, controlled_ball_radius=0.05, ball_radius=0.05, max_speed=1) -> None:
        super().__init__(controlled_ball_radius, ball_radius, max_speed)
        self.v = 0
        self.counter = 0
        self.controlled_ball = False
    def set_type(self, v):
        self.v = v
    def done(self, state, obs):
        return False
    def act(self, obs):
        # balls = obs["balls"]
        agent_state = obs["agent_ball"]
        ball_state = obs["target_ball"] # chosen by env
        # ball_state = balls[ball_id]
        target_state = obs["target"] # chosen by env

        # agent_state = balls[0]
        dx = ball_state[0] - agent_state[0]
        dy = ball_state[1] - agent_state[1]
        ball_target_d = ball_state - target_state
        done=False
        action = np.zeros(2)
        if (
            "controlled_ball_attached" in obs
            and not obs["controlled_ball_attached"]
        ):
            if self.v < 2:
                if abs(dx) > 3e-2:
                    action[0] = np.sign(dx)
                elif abs(dy) > 3e-2:
                    action[1] = np.sign(dy)
            else:
                if abs(dy) > 3e-2:
                    action[1] = np.sign(dy)
                elif abs(dx) > 3e-2:
                    action[0] = np.sign(dx)
        else:
            if not self.controlled_ball:
                # done = True
                self.controlled_ball = True
            self.counter += 1
            action[0] = 1
            # if self.v > 0 and self.v < 3:
            #     if abs(ball_target_d[0]) > 3e-2:
            #         action[0] = np.sign(target_state[0] - ball_state[0])
            #     elif abs(ball_target_d[1]) > 3e-2:
            #         action[1] = np.sign(target_state[1] - ball_state[1])
            # else:
            #     if abs(ball_target_d[1]) > 3e-2:
            #         action[1] = np.sign(target_state[1] - ball_state[1])
            #     elif abs(ball_target_d[0]) > 3e-2:
            #         action[0] = np.sign(target_state[0] - ball_state[0])
        direction = 0
        if action[1] == 1:
            direction = 1
        elif action[0] == -1:
            direction = 2
        elif action[1] == -1:
            direction = 3

        return np.array([direction, self.max_speed]), done

class BoxPusherDrawingPlanner(BoxPusherPlanner):
    """
    Goes to target and tries to draw something meaningful
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.v = 0
        self.counter = 0
        self.controlled_ball = False
    def set_type(self, v):
        self.v = v
    def done(self, state, obs):
        return False
    def act(self, obs):
        # balls = obs["balls"]
        agent_state = obs["agent_ball"]
        ball_state = obs["target_ball"] # chosen by env
        # ball_state = balls[ball_id]
        target_state = obs["target"] # chosen by env

        # agent_state = balls[0]
        dx = ball_state[0] - agent_state[0]
        dy = ball_state[1] - agent_state[1]
        ball_target_d = ball_state - target_state
        done=False
        action = np.zeros(2)
        if (
            "controlled_ball_attached" in obs
            and not obs["controlled_ball_attached"]
        ):
            if self.v < 2:
                if abs(dx) > 3e-2:
                    action[0] = np.sign(dx)
                elif abs(dy) > 3e-2:
                    action[1] = np.sign(dy)
            else:
                if abs(dy) > 3e-2:
                    action[1] = np.sign(dy)
                elif abs(dx) > 3e-2:
                    action[0] = np.sign(dx)
        else:
            if not self.controlled_ball:
                # done = True
                self.controlled_ball = True
            self.counter += 1
            k = 24
            if self.counter < k:
                action[0] = 1
            elif self.counter < k*2:
                action[1] = 1
            elif self.counter < k*3:
                action[0] = -1
            else:
                action[1] = 1
            if self.counter > k*4:
                self.counter = 0
            # if self.v > 0 and self.v < 3:
            #     if abs(ball_target_d[0]) > 3e-2:
            #         action[0] = np.sign(target_state[0] - ball_state[0])
            #     elif abs(ball_target_d[1]) > 3e-2:
            #         action[1] = np.sign(target_state[1] - ball_state[1])
            # else:
            #     if abs(ball_target_d[1]) > 3e-2:
            #         action[1] = np.sign(target_state[1] - ball_state[1])
            #     elif abs(ball_target_d[0]) > 3e-2:
            #         action[0] = np.sign(target_state[0] - ball_state[0])
        direction = 0
        if action[1] == 1:
            direction = 1
        elif action[0] == -1:
            direction = 2
        elif action[1] == -1:
            direction = 3

        return np.array([direction, self.max_speed]), done
