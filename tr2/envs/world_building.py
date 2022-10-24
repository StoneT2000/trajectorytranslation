"""
Functions for adding objects to a sapien env
"""

import numpy as np
import sapien.core as sapien


def add_ball(
    scene: sapien.Scene,
    density=1000.0,
    ball_id=0,
    half_size=0.05,
    material=None,
    color=None,
    id=None,
    collision=True,
    sphere=False
) -> sapien.Actor:
    builder: sapien.ActorBuilder = scene.create_actor_builder()
    if type(half_size) != list:
        half_size = [half_size, half_size, half_size]
    if collision:
        
        builder.add_box_collision(
            half_size=half_size, material=material, density=density
        )
    if color is None:
        color = [ball_id / 10, 0.546, 0.166]
    # builder.add_box_visual(half_size=half_size, color=color)
    if sphere:
        builder.add_sphere_visual(radius=half_size[0], color=color)
    else:
        builder.add_box_visual(half_size=half_size, color=color)
    if id is None:
        ball = builder.build(name=f"ball_{ball_id}")
    else:
        ball = builder.build(name=id)
    ball.set_pose(sapien.Pose(p=[0, 0, 0.2], q=[1, 0, 0, 0]))
    return ball


def add_pole(
    scene: sapien.Scene,
    pose: sapien.Pose,
    radius=0.05,
    color=None,
    # density=1000.0,
    physical_material: sapien.PhysicalMaterial = None,
    pole_id=0,
) -> sapien.Actor:
    builder = scene.create_actor_builder()
    half_length = 0.5
    builder.add_capsule_collision(
        radius=radius, material=physical_material, half_length=half_length
    )
    if color is None:
        color = [0.2, 0.2, 1]
    builder.add_capsule_visual(radius=radius, color=color, half_length=half_length)
    sphere = builder.build_kinematic(name=f"pole_{pole_id}")
    sphere.set_pose(pose)
    return sphere


def add_target(
    scene: sapien.Scene,
    pose: sapien.Pose,
    radius=0.05,
    color=None,
    # density=1000.0,
    physical_material: sapien.PhysicalMaterial = None,
    target_id=0,
) -> sapien.Actor:
    builder = scene.create_actor_builder()
    half_length = 0.5
    # builder.add_capsule_collision(
    #     radius=radius, material=physical_material, half_length=half_length
    # )
    if color is None:
        color = [0.2, 0.2, 1]
    builder.add_sphere_visual(
        radius=radius,
        color=color,
    )
    sphere = builder.build_static(name=f"target_{target_id}")
    sphere.set_pose(pose)
    return sphere


def create_box(
    scene: sapien.Scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    is_kinematic=False,
    density=1000.0,
    physical_material: sapien.PhysicalMaterial = None,
    name="",
    collision=True,
) -> sapien.Actor:
    """Create a box.

    Args:
        scene: sapien.Scene to create a box.
        pose: 6D pose of the box.
        half_size: [3], half size along x, y, z axes.
        color: [3] or [4], rgb or rgba.
        is_kinematic: whether an object is kinematic (can not be affected by forces).
        density: float, the density of the box.
        physical_material: physical material of the actor.
        name: name of the actor.

    Returns:
        sapien.Actor
    """
    half_size = np.array(half_size)
    builder = scene.create_actor_builder()
    if collision:
        builder.add_box_collision(
            half_size=half_size, material=physical_material, density=density
        )  # Add collision shape
    builder.add_box_visual(half_size=half_size, color=color)  # Add visual shape
    if is_kinematic:
        box = builder.build_kinematic(name=name)
    else:
        box = builder.build(name=name)
    box.set_pose(pose)
    return box
