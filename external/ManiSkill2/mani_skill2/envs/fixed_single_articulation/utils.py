import json
from pathlib import Path

import sapien.core as sapien


def ignore_collision(articulation: sapien.Articulation):
    """ignore collision among all movable links for acceleration"""
    for joint, link in zip(articulation.get_joints(), articulation.get_links()):
        if joint.type in ["revolute", "prismatic"]:
            shapes = link.get_collision_shapes()
            for s in shapes:
                g0, g1, g2, g3 = s.get_collision_groups()
                s.set_collision_groups(g0, g1, g2 | 1 << 31, g3)
