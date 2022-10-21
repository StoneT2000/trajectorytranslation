from skilltranslation.utils.vision import get_point_from_image, get_segmented_point_clouds
import torch
import numpy as np
from pathlib import Path
import numpy as np
import cv2
from transforms3d.quaternions import mat2quat
import trimesh
import sys
import sklearn
import open3d as o3d

def gen_cuboid_pcd(s=8):
    pcd = []
    xs = np.linspace(0, 1.0, num=s)
    for x in xs:
        for y in xs:
            pcd.append([x, y, 0.0])
            pcd.append([x, y, 1.0])
            pcd.append([x, 0.0, y])
            pcd.append([x, 1.0, y])
            pcd.append([0.0, x, y])
            pcd.append([1.0, x, y])
    pcd = np.array(pcd)
    return pcd - [0.5, 0.5, 0.5]
CUBOID_PCD = gen_cuboid_pcd(12)
def ICP(source, target, threshold=0.02):
    """
    Parameters
    ----------
    source : (n x 3) numpy matrix
    target : (n x 3) numpy matrix
    threshold : float

    takes a source set of points and a target set of points and returns the estimated pose matrix using the ICP algorithm
    """

    trans_init = get_init_transform(source, target)

    points = o3d.utility.Vector3dVector(source)
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = points
    points = o3d.utility.Vector3dVector(target)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = points

    ## this normal estimation can be replaced with r method
    # target_pcd.estimate_normals()

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )
    return reg_p2p.transformation


def center_mean(points):
    return np.array([np.mean(points[:, 0]), np.mean(points[:, 1]), np.mean(points[:, 2])])

def transform_camera_to_world(points, extrinsic):
    A = (points - extrinsic[:3, 3]) @ extrinsic[:3, :3]
    return A
def get_init_transform(source, target):
    # atm, just a simple translation of center
    center_source = center_mean(source)
    center_target = center_mean(target)
    A = np.eye(4)
    A[:3, 3] = center_target - center_source
    return A
def predict_pose(rgb, depth, seg, meta, icp_config=dict(threshold=0.004)):
    pcd, colors = get_segmented_point_clouds(meta, rgb,depth, seg)
    target_pcd = gen_cuboid_pcd(s=12)*0.05
    pcd=np.array(pcd[0])
    pcd = transform_camera_to_world(pcd, meta['cam_ext'])
    pred_pose = ICP(target_pcd, pcd, **icp_config).copy()
    # return pred_pose
    p = pred_pose[:3, 3]
    q = mat2quat(pred_pose[:3, :3])
    return np.hstack([p, q])
from sklearn.cluster import KMeans
def predict_pose_noseg(rgb, depth, meta, min_clusters=1):
    # np.save("test.rgb.npy", rgb)
    pcd = get_point_from_image(meta['cam_int'],depth)
    pcd = transform_camera_to_world(pcd, meta['cam_ext'])
    nearby = np.linalg.norm(pcd[:,:,:2] - np.array([0,0]),axis=2) < 0.25
    obj_pts = (pcd[:,:,2] >1e-6) & (nearby)
    obj_pts = obj_pts & (rgb[:,:,0] > 120) & (rgb[:,:,1] < 100) 
    # sepearate floor and background, and uses red to separate gripper, leaving just the blocks 
    best_loss = 99999
    optimal_segmentation = None
    for clusters in range(min_clusters,10):
        kmeans = KMeans(n_clusters=clusters)
        kmeans.fit(pcd[obj_pts])
        y_km = kmeans.predict(pcd.reshape(-1,3)[:,:])
        y_km = y_km.reshape(512,512)+ 1
        y_km[~obj_pts] = 0
        # minimize scaled inertia
        loss = kmeans.inertia_ + clusters*0.5
        if loss < best_loss:
            best_loss = loss
            optimal_segmentation = y_km
        else:
            break
    max_id = np.max(optimal_segmentation)
    # np.save("test.seg.npy", optimal_segmentation)
    poses = {}
    for i in range(1, max_id + 1):
        seg = optimal_segmentation.copy()
        seg[seg != i] = 0
        seg[seg == i] = 1
        meta_cp = meta.copy()
        meta_cp["object_ids"] = [1]
        pose = predict_pose(rgb, depth, seg, meta=meta_cp)
        poses[i] = pose
    return poses


def predict_pose_real(pcd, rgb, sel=None, template_size=0.023, cluster_penalty=0.5):
    pcd2 = pcd.copy()
    # segment out the surround
    if sel is None:
        sel = (pcd2[:,0]>-0.05) & (pcd2[:,1]>-0.12) & (pcd2[:,1]<0.12)
    pcd2 = pcd2[sel]
    rgb2 = rgb[sel].copy()

    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(pcd2)
    pcl.colors= o3d.utility.Vector3dVector(rgb2)
    _, seg_pcd=pcl.segment_plane(distance_threshold=4e-3, ransac_n=10, num_iterations=500, seed=0)

    mask = np.zeros(len(pcd2), bool)
    mask[seg_pcd]=True

    pcd2s=pcd2[~mask].copy()
    rgb2s=rgb2[~mask].copy()
    mask.sum(), len(mask)
    min_cluster = 1
    max_cluster = 10
    optimal_segmentation = None
    best_loss = 99999999
    for clusters in range(min_cluster, max_cluster):
        kmeans = KMeans(n_clusters=clusters)
        kmeans.fit(pcd2s)
        y_km = kmeans.predict(pcd2s)
        loss = kmeans.inertia_ + clusters*cluster_penalty
        if loss < best_loss:
            best_loss = loss
            optimal_segmentation = y_km
        else:
            break
    seg_ids = np.unique(optimal_segmentation)
    poses = {}
    for seg_id in seg_ids:
        pcd2obj = pcd2s[optimal_segmentation == seg_id]
        # rgb2obj = rgb2s[y_km==seg_id]
        target_pcd = gen_cuboid_pcd(s=15)*template_size
        pred_pose = ICP(target_pcd, pcd2obj, **dict(threshold=0.01)).copy()
        # return pred_pose
        p = pred_pose[:3, 3]
        q = mat2quat(pred_pose[:3, :3])
        # pcd2obj -= p
        # pcd2obj = pcd2obj @ pred_pose[:3, :3]
        poses[seg_id] = pred_pose
    return poses
def view_pcd_bboxes(pcds, rgbs,zoom=0.12, poses=[], template_size=0.0235):
    pcls=[]
    for pcd,rgb in zip(pcds,rgbs):
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(pcd)
        if rgb is None:
            rgb = np.zeros_like(pcd)
            rgb += np.random.rand(3)
        pcl.colors = o3d.utility.Vector3dVector(rgb)
        pcls.append(pcl)
    bboxes=[]
    for k, p in poses.items():
        # gen
        bbox =o3d.geometry.OrientedBoundingBox(
            p[:3, 3], p[:3, :3], np.ones(3)*template_size
        )
        bbox.color = [0,1,0]
        bboxes.append(bbox)
        
    o3d.visualization.draw_geometries(pcls+bboxes,
                                  zoom=zoom,
                                  front=[.5, -1.2, .5],
                                  lookat=[0.45,.05,.15],
                                  up=[0,.0,1])

def estimate_single_block_pose_realsense():
    import pyrealsense2 as rs
    import numpy as np
    from skilltranslation.utils.vision import draw_projected_box3d, get_point_from_image
    import skilltranslation.utils.pose_est as pose_est
    import open3d as o3d
    from importlib import reload
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    # Skip 5 first frames to give the Auto-Exposure time to adjust
    for x in range(5):
        pipeline.wait_for_frames()
    pipeline = pipeline
    profile = profile

    for _ in range(5):
        pipeline.wait_for_frames()


    for i in range(1):
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame:
            continue

        pc = rs.pointcloud()
        pc.map_to(color_frame)
        cloud = pc.calculate(depth_frame)
        cloud.export_to_ply(f"point_cloud_{i}.ply", color_frame)
    # import open3d as o3d

    # TODO, make this T configurable. This is dependent on camera setup in real world
    T=np.array([[ 0.99612855,  0.05620104, -0.067597  ,  0.38972186],
     [ 0.08638289, -0.48314859,  0.87126657, -0.35334297],
     [ 0.01630669, -0.87373273, -0.48613291,  0.42647845],
     [ 0.        ,  0.        ,  0.        ,  1.        ]])
    Y_OFFSET = 0.17214171
    Y_OFFSET = 0.16814171
    X_OFFSET = -0.075
    Z_OFFSET = -0.1
    T[1,-1] -= Y_OFFSET
    T[0,-1] -= X_OFFSET
    T[2, -1] -= Z_OFFSET
    cloud = o3d.io.read_point_cloud("point_cloud_0.ply")
    points = np.asarray(cloud.points)
    points[:, 1:3] *= -1

    points = points @ T[:3, :3].T + T[:3, 3][None, :]
    cloud.points = o3d.utility.Vector3dVector(points)
    rgb=np.asarray(cloud.colors)

    # SEGMENTATION
    sel = (points[:,0]>.2) & (points[:, 0] < 0.8) & (points[:, 1] < 0.7) & (points[:, 2] > 0.01) & (points[:, 2] < 0.6)
    rgb2 = rgb[sel].copy()
    
    pcd2 = points[sel].copy()
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(pcd2)
    pcl.colors= o3d.utility.Vector3dVector(rgb2)
    _, seg_pcd=pcl.segment_plane(distance_threshold=4e-3, ransac_n=10, num_iterations=500, seed=0)
    # view_pcd_bboxes([pcd2], [rgb2], poses=dict())
    mask = np.zeros(len(pcd2), bool)
    mask[seg_pcd]=True

    pcd2s=pcd2[~mask].copy()
    rgb2s=rgb2[~mask].copy()
    mask.sum(), len(mask)
    
    import sklearn
    from sklearn.cluster import KMeans
    min_cluster = 1
    max_cluster = 20
    cluster_penalty=1.75
    optimal_segmentation = None
    best_loss = 99999999
    for clusters in range(min_cluster, max_cluster):
        kmeans = KMeans(n_clusters=clusters)
        kmeans.fit(pcd2s)
        y_km = kmeans.predict(pcd2s)
        loss = kmeans.inertia_ + clusters*cluster_penalty
        if loss < best_loss:
            # print(clusters, loss)
            best_loss = loss
            optimal_segmentation = y_km
        else:
            break
    seg_ids = np.unique(optimal_segmentation)
    
    
    poses = {}
    from transforms3d.quaternions import mat2quat
    template_size=0.05
    for seg_id in seg_ids:
        pcd2obj = pcd2s[optimal_segmentation == seg_id]
        if len(pcd2obj) < 500: continue
        # rgb2obj = rgb2s[y_km==seg_id]
        target_pcd = pose_est.gen_cuboid_pcd(s=15)*template_size
        pred_pose = pose_est.ICP(target_pcd, pcd2obj, **dict(threshold=0.01)).copy()

        # FIX OFFSET HERE!
        # pred_pose[:3, 3] += np.array([0.56, 0, 0])
        # return pred_pose
        p = pred_pose[:3, 3]
        q = mat2quat(pred_pose[:3, :3])
        # pcd2obj -= p
        # pcd2obj = pcd2obj @ pred_pose[:3, :3]
        poses[seg_id] = pred_pose
        
    sim_poses = dict()
    detected_block_pose = None
    closest_to_sensor_dist = 99999
    for k in poses:
        p = poses[k]
        # print(p)
        # print(p[2,-1])
        if p[2, -1] > 0.1: continue
        if p[1, -1] > 0.1: continue
        if p[1, -1] < closest_to_sensor_dist: 
            closest_to_sensor_dist = p[1, -1]
            detected_block_pose = p
        # p[:3, 3] -= np.array([0.56, 0, 0])
        sim_poses[k] = p
        # detected_block_pose = p
    # rgb[:,[2,0]] = rgb[:,[0, 2]]
    # view_pcd_bboxes([points], [rgb], poses=sim_poses, template_size=template_size)
    return detected_block_pose