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

        builder: sapien.ActorBuilder = self._scene.create_actor_builder()
        
        builder.add_box_collision(
            half_size=half_size, material=ball_mtl, density=10000000
        )
        builder.add_box_visual(half_size=half_size, color=COLORS["RED"])
        ball = builder.build_kinematic(name=f"wall_{i}")
        ball.set_pose(sapien.Pose(p=[0, 0, 0.2], q=[1, 0, 0, 0]))
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