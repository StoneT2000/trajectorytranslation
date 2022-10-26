import numpy as np
import sapien.core as sapien
from transforms3d.quaternions import axangle2quat, quat2axangle


def add_walls(self, walls):
    # pick wall, expand linearly along one direction, make longer wall
    half_sizes = np.array([0.025, 1 / self.world_size, 1 / self.world_size])
    long_wall_ids = 0
    simulated_walls = set(self.walls)
    def add_long_walls(start_wall):
        nonlocal long_wall_ids
        for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            nx, ny = start_wall[0] + dx, start_wall[1] + dy
            if (nx, ny) in simulated_walls and start_wall in simulated_walls:
                mins = [self.world_size*2, self.world_size*2]
                maxes = [-self.world_size*2, -self.world_size*2]
                for k in range(1, self.world_size):
                    nx2, ny2 = nx + dx*k, ny+dy*k
                    w_x, w_y = self._scale_world([nx2, ny2])
                    mins[0] = min(mins[0], w_x)
                    mins[1] = min(mins[1], w_y)
                    maxes[0] = max(maxes[0], w_x)
                    maxes[1] = max(maxes[1], w_y)
                    
                    if (nx2, ny2) not in simulated_walls:
                        break
                    simulated_walls.remove((nx2, ny2))
                for k in range(self.world_size):
                    nx2, ny2 = nx - dx*k, ny -dy*k
                    w_x, w_y = self._scale_world([nx2, ny2])
                    mins[0] = min(mins[0], w_x)
                    mins[1] = min(mins[1], w_y)
                    maxes[0] = max(maxes[0], w_x)
                    maxes[1] = max(maxes[1], w_y)
                    if (nx2, ny2) not in simulated_walls:
                        break
                    simulated_walls.remove((nx2, ny2))
                longwall_center = (np.array(mins) + np.array(maxes)) / 2
                pose = sapien.Pose(
                    [longwall_center[0], longwall_center[1], 0], q=axangle2quat([0, 0, 1], np.deg2rad(0))
                )
                long_sizes = np.array([(maxes[0] - mins[0])/2 - 1/self.world_size, (maxes[1] - mins[1])/2 - 1/self.world_size, 0.025])
                for i in range(2):
                    if maxes[i] - mins[i] == 0:
                        long_sizes[i] = 1/self.world_size
                builder = self._scene.create_actor_builder()
                ball_mtl = self._scene.create_physical_material(0.,0., 0.0)
                density = 1000.0
                builder.add_box_collision(half_size=long_sizes, material=ball_mtl, density=density)
                builder.add_box_visual(half_size=long_sizes, color=[0.2, 0.2, 1])
                longwall = builder.build_kinematic(name=f"worldwall_{long_wall_ids}")
                long_wall_ids += 1
                longwall.set_pose(pose)
                return
    for w in self.walls[:]:
        add_long_walls(w)
    wall_id = 0
    def add_wall(x, y, half_sizes, color=[0.2,0.2,1]):
        nonlocal wall_id
        wall_id += 1
        builder = self._scene.create_actor_builder()
        ball_mtl = self._scene.create_physical_material(0.,0., 0.0)
        density = 1000.0
        pose = sapien.Pose(
            [x, y, 0], q=axangle2quat([0, 1, 0], np.deg2rad(90))
        )
        builder.add_box_collision(
            # pose=pose,
            half_size=half_sizes, material=ball_mtl, density=density
        )
        builder.add_box_visual(half_size=half_sizes, color=color)
        wall = builder.build_kinematic(name=f"worldwall_box_{wall_id}")
        wall.set_pose(pose)
    for w in simulated_walls:
        w_x, w_y = self._scale_world(w)
        add_wall(w_x, w_y, half_sizes)