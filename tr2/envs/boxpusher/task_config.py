import numpy as np
import sapien.core as sapien

from tr2.envs.world_building import add_ball, add_target

COLORS = {
    "GREEN": [0.1662, 0.5645, 0.2693],
    "RED": [0.9350, 0.0694, 0.0956],
    "BLUE_DARK": [0.05, 0.01, 0.99],
    "BLUE": [0.45, 0.52, 0.93],
}
def obstacle(self, n=2, half_size=None):
    half_size = [half_size, half_size, half_size]
    ball_mtl = self._scene.create_physical_material(0., 0., 0.0)
    self.obstacle_actors = []
    target_xy = self.target_actors[0].pose.p[:2]
    agent_xy = self.balls[0].pose.p[:2]
    ball_xy = self.balls[1].pose.p[:2]
    p = self.np_random.random() > 0.5
    for i in range(5):
        obs_xy = [target_xy[0], target_xy[1]]
        if p:
            obs_xy[1] = self.np_random.random() * (0.5+0.25) * (target_xy[1] - ball_xy[1]) + ball_xy[1]
            obs_xy[0] += self.np_random.random()*0.5-0.1
        else:
            obs_xy[0] = self.np_random.random() * (0.5+0.25) * (target_xy[0] - ball_xy[0]) + ball_xy[0]
            obs_xy[1] += self.np_random.random()*0.2-0.1
        # ball = add_ball(
        #     self._scene,
        #     ball_id=f"obstacle_{i}",
        #     material=ball_mtl,
        #     half_size=half_size,
        #     density=10000000,
        #     color=,
        # )
        builder: sapien.ActorBuilder = self._scene.create_actor_builder()
        
        builder.add_box_collision(
            half_size=half_size, material=ball_mtl, density=10000000
        )
        builder.add_box_visual(half_size=half_size, color=COLORS["RED"])
        ball = builder.build_kinematic(name=f"wall_{i}")
        ball.set_pose(sapien.Pose(p=[0, 0, 0.2], q=[1, 0, 0, 0]))
        self.obstacle_actors.append(ball)
        ball.set_pose(sapien.Pose([obs_xy[0], obs_xy[1], 0.05]))

def long_maze(self, half_size=None):
    ball_mtl = self._scene.create_physical_material(0., 0., 0.0)
    # add some obstacles randomly along path.
    # sample from [agent] - [target]
    self.obstacle_actors = []
    target_xy = self.target_actors[0].pose.p[:2]
    agent_xy = self.balls[0].pose.p[:2]
    ball_xy = self.balls[1].pose.p[:2]

    # generate long windy path??
    # take random walks towards target
    agent_target_path = []
    if self.np_random.random() > 0.5:
        p1 = ball_xy.copy()
        p1[0] = agent_xy[0]
        agent_target_path += [(agent_xy, p1)]
    else:
        p1 = ball_xy.copy()
        p1[1] = agent_xy[1]
        agent_target_path += [(agent_xy, p1)]
    agent_target_path += [(p1, ball_xy)]
    for t in self.agent_target_path_actors:
        self._scene.remove_actor(t)
    
    walk_size = 1
    dist_to_target = 99999999
    while dist_to_target > 1e-1:
        start = agent_target_path[-1][1]
        if dist_to_target < 2:
            end1 = target_xy.copy()
            end2 = target_xy.copy()
            if np.random.rand() > 0.5:
                end1[0] = target_xy[0]
            else:
                end1[1] = target_xy[1]
            agent_target_path += [(start, end1), (end1, target_xy)]
            break
        
        end = start.copy()
        if np.random.rand() > 0.5:
            end[1] += np.sign(target_xy[1]-start[1]) * walk_size
        else:
            end[0] += np.sign(target_xy[0]-start[0]) * walk_size
        dist_to_target = np.linalg.norm(end - target_xy)
        agent_target_path += [(start, end)]
    for i, (a,b) in enumerate(agent_target_path):
        color = COLORS["BLUE"].copy()
        color[1] = i / len(agent_target_path)
        t=add_target(self._scene, pose=sapien.Pose([b[0],b[1],0]),color=color, physical_material=ball_mtl,radius=self.target_radius)
        self.agent_target_path_actors.append(t)
    for i in range(5):
        obs_xy = [target_xy[0], target_xy[1]]
        # if self.np_random.random() > 0.5:
        obs_xy[1] = (self.np_random.random()*1) * (target_xy[1] - ball_xy[1]) + ball_xy[1]
        # else:
        obs_xy[0] = (self.np_random.random()*1) * (target_xy[0] - ball_xy[0]) + ball_xy[0]
        redo = False
        for a,b in agent_target_path:
            dist = np.linalg.norm(np.cross(b-a, a-obs_xy)) / np.linalg.norm(b - a)
            if dist < 0.2:
                redo=True
                break
        if redo: continue
        
        # print(obs_xy)
        # obs_xy = [ball_xy[0], ball_xy[1]]
        # if self.np_random.random() > 0.5:
        #     obs_xy[1] =  (self.np_random.random()*0.65+0.35)* (agent_xy[1] - ball_xy[1]) + ball_xy[1]
        # else:
        #     obs_xy[0] = (self.np_random.random()*0.65+0.35)* (agent_xy[0] - ball_xy[0]) + ball_xy[0]
        ball = add_ball(
            self._scene,
            ball_id=f"obstacle_{i}",
            material=ball_mtl,
            half_size=half_size,
            density=10000000,
            color=COLORS["RED"],
        )
        self.obstacle_actors.append(ball)
        ball.set_pose(sapien.Pose([obs_xy[0], obs_xy[1], 0.05]))


def silo_obstacle_push(self, n=3, half_size=None, disable_collision=False):
    half_size = [half_size, half_size, half_size]
    ball_mtl = self._scene.create_physical_material(0., 0., 0.0)
    self.obstacle_actors = []
    # from paper, original obstacle size divided by 2
    half_size = [0.05, 0.05, 0.05]
    for i in range(n):
        obs_xy = [0 - 0.4 + i * 0.4, 0]
        # from paper, obstacles are 0.16 in front of the target block
        scale = 0.05 / 0.02
        obs_xy[1] += 0.16 * scale
        # obs_xy = [target_xy[0], target_xy[1]]
        # if p:
        #     obs_xy[1] = self.np_random.random() * (0.5+0.25) * (target_xy[1] - ball_xy[1]) + ball_xy[1]
        #     obs_xy[0] += self.np_random.random()*0.2-0.1
        # else:
        #     obs_xy[0] = self.np_random.random() * (0.5+0.25) * (target_xy[0] - ball_xy[0]) + ball_xy[0]
        #     obs_xy[1] += self.np_random.random()*0.2-0.1
        # ball = add_ball(
        #     self._scene,
        #     ball_id=f"obstacle_{i}",
        #     material=ball_mtl,
        #     half_size=half_size,
        #     density=10000000,
        #     color=,
        # )
        builder: sapien.ActorBuilder = self._scene.create_actor_builder()
        if not disable_collision:
            builder.add_box_collision(
                half_size=half_size, material=ball_mtl, density=10000000
            )
        builder.add_box_visual(half_size=half_size, color=COLORS["RED"])
        ball = builder.build_kinematic(name=f"wall_{i}")
        ball.set_pose(sapien.Pose(p=[0, 0, 0.2], q=[1, 0, 0, 0]))
        self.obstacle_actors.append(ball)
        ball.set_pose(sapien.Pose([obs_xy[0], obs_xy[1], 0.05]))