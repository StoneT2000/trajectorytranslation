import sys
from pathlib import Path

import cv2
import numpy as np
import open3d
import sklearn
import torch
import trimesh
from transforms3d.quaternions import quat2mat

VERTEX_COLORS = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
]
def chamfer_distance(source, target, metric="l2"):
    """
    computes the chamfer distance
    """
    source_neighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm="kd_tree", p=2).fit(
        source
    )
    distances = source_neighbors.kneighbors(target)[0]
    return np.mean(distances)

def get_corners():
    """Get 8 corners of a cuboid. (The order follows OrientedBoundingBox in open3d)
      (y)
      2 -------- 7
     /|         /|
    5 -------- 4 .
    | |        | |
    . 0 -------- 1 (x)
    |/         |/
    3 -------- 6
    (z)
    """
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    return corners - [0.5, 0.5, 0.5]


def get_edges(corners):
    assert len(corners) == 8
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(corners[i] == corners[j]) == 2:
                edges.append((i, j))
    assert len(edges) == 12
    return edges


def draw_projected_box3d(image, center, size, rotation, extrinsic, intrinsic, color=(0, 1, 0), thickness=1):
    """Draw a projected 3D bounding box on the image.

    Args:
        image (np.ndarray): [H, W, 3] array.
        center: [3]
        size: [3]
        rotation (np.ndarray): [3, 3]
        extrinsic (np.ndarray): [4, 4]
        intrinsic (np.ndarray): [3, 3]
        color: [3]
        thickness (int): thickness of lines
    Returns:
        np.ndarray: updated image.
    """
    if rotation.shape == (4,):
        rotation = quat2mat(rotation)
    corners = get_corners()  # [8, 3]
    edges = get_edges(corners)  # [12, 2]
    corners = corners * size
    corners_world = corners @ rotation.T + center
    corners_camera = corners_world @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    corners_image = corners_camera @ intrinsic.T
    uv = corners_image[:, 0:2] / corners_image[:, 2:]
    uv = uv.astype(int)

    for (i, j) in edges:
        cv2.line(
            image,
            (uv[i, 0], uv[i, 1]),
            (uv[j, 0], uv[j, 1]),
            tuple(color),
            thickness,
            cv2.LINE_AA,
        )
    return image
def transform_points(pts, mat):
    return pts @ mat[:3, :3].T + mat[:3, 3]
def get_point_from_image(intrinsic, depth):
    """
    Parameters
    ----------
    intrinsic : an intrinsic (3 x 3) numpy matrix
    depth: a depth image, (h x w) numpy matrix for height h and width w

    Uses the intrinsic camera matrix to return a set of 3D points from a depth image that represent the 3D points that are captured
    """
    z = depth
    v, u = np.indices(z.shape)
    uv1 = np.stack([u + 0.5, v + 0.5, np.ones_like(z)], axis=-1)
    points_viewer = uv1 @ np.linalg.inv(np.array(intrinsic)).T * z[..., None]  # [H, W, 3]
    return points_viewer

def get_segmented_point_clouds(meta, rgb, depth, seg):
    object_ids = meta["object_ids"]
    intrinsic = meta["cam_int"]
    all_points = get_point_from_image(intrinsic, depth)
    point_clouds = []
    colors = []
    for id in object_ids:
        # isolate the points
        points = all_points[seg == id]
        color = rgb[seg == id]
        point_clouds.append(points)
        colors.append(color)
    return point_clouds, colors
