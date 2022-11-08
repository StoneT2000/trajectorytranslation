import time
import yaml
import mplib
import numpy as np
import sapien.core as sapien
from sapien.core import Pose
import trimesh
from tqdm import tqdm
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from mani_skill2 import ASSET_DIR
from mani_skill2.envs.fixed_single_articulation.open_cabinet_drawer import (
    OpenCabinetDrawer,
)
from mani_skill2.agents.camera import get_texture_by_dltensor

OUTPUT_DIR = Path("/media/DATA/LINUX_DATA/mani_skill2022/data/cabinet_drawer_handle")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

NUM_POSE_PER_SCENE = 100

MAX_DEPTH = 2.0


def visualize_depth(depth):
    cmap = plt.get_cmap("rainbow")
    if depth.dtype == np.uint16:
        depth = depth.astype(float) / 1000.0
    if len(depth.shape) == 3:
        depth = depth[..., 0]
    depth = np.clip(depth / MAX_DEPTH, 0.0, 1.0)
    vis_depth = cmap(depth)
    vis_depth = (vis_depth[:, :, :3] * 255.0).astype(np.uint8)
    vis_depth = cv2.cvtColor(vis_depth, cv2.COLOR_RGB2BGR)
    return vis_depth


def gen_single_scene(config_name):
    np.set_printoptions(suppress=True, precision=4)
    env = OpenCabinetDrawer(
        articulation_config_path=ASSET_DIR
        / f"partnet_mobility_configs/fixed_cabinet_drawer_filtered/{config_name}.yml",
        is_action_normalized=False,
    )
    env.reset()
    env._init_open_extent_range = 1.0
    env._scene.set_ambient_light([0.5, 0.5, 0.5])
    env._scene.add_point_light([-0.3, -0.3, 2.5], [30, 30, 30])
    env._scene.add_point_light([2, -2, 2.5], [10, 10, 10])
    env._scene.add_point_light([-2, 2, 2.5], [10, 10, 10])
    env._scene.add_point_light([2, 2, 2.5], [10, 10, 10])
    env._scene.add_point_light([-2, -2, 2.5], [10, 10, 10])

    def get_handle_ids():
        visual_body_ids = []
        for link in env._articulation.get_links():
            for visual_body in link.get_visual_bodies():
                if "handle" not in visual_body.get_name():
                    continue
                visual_body_ids.append(visual_body.get_visual_id())
        return visual_body_ids

    handle_ids = get_handle_ids()

    def random_pose():
        pos_x = env.np_random.uniform(
            env._articulation_init_pos_min_x, env._articulation_init_pos_max_x
        )
        pos_y = env.np_random.uniform(
            env._articulation_init_pos_min_y, env._articulation_init_pos_max_y
        )
        rot_z = env.np_random.uniform(
            env._articulation_init_rot_min_z, env._articulation_init_rot_max_z
        )
        env._articulation.set_root_pose(
            Pose(
                [
                    pos_x,
                    pos_y,
                    -env._articulation_config.scale
                    * env._articulation_config.bbox_min[2],
                ],
                [np.sqrt(1 - rot_z ** 2), 0, 0, rot_z],
            )
        )

        [[lmin, lmax]] = env._target_joint.get_limits()
        init_open_extent = env.np_random.uniform(0, env._init_open_extent_range)
        qpos = np.zeros(env._articulation.dof)
        for i in range(env._articulation.dof):
            qpos[i] = env._articulation.get_active_joints()[i].get_limits()[0][0]
        qpos[env._articulation_config.target_joint_idx] = (
            lmin + (lmax - lmin) * init_open_extent
        )
        env._articulation.set_qpos(qpos)

    for pose_idx in tqdm(range(NUM_POSE_PER_SCENE)):
        pose_dir = OUTPUT_DIR / f"{config_name}_{pose_idx:04d}"
        pose_dir.mkdir(parents=True, exist_ok=True)
        random_pose()
        views = env._agent.get_images(depth=True)
        cv2.imwrite(str(pose_dir / "ir_L.png"), views["base_d415"]["ir_l"])
        cv2.imwrite(str(pose_dir / "ir_R.png"), views["base_d415"]["ir_r"])
        cv2.imwrite(str(pose_dir / "rgb.png"), views["base_d415"]["rgb"][..., ::-1])
        cv2.imwrite(
            str(pose_dir / "clean_depth.png"),
            (views["base_d415"]["clean_depth"][:, :, 0] * 1000.0).astype(np.uint16),
        )
        cv2.imwrite(
            str(pose_dir / "clean_depth_colored.png"),
            visualize_depth(views["base_d415"]["clean_depth"]),
        )
        cv2.imwrite(
            str(pose_dir / "stereo_depth.png"),
            (views["base_d415"]["stereo_depth"] * 1000.0).astype(np.uint16),
        )
        cv2.imwrite(
            str(pose_dir / "stereo_depth_colored.png"),
            visualize_depth(views["base_d415"]["stereo_depth"]),
        )
        appear_visual_ids = get_texture_by_dltensor(
            env._agent._sensors["base_d415"]._cam_rgb, "Segmentation", "uint32"
        )[..., 0]

        masks = []
        for handle_id in handle_ids:
            masks.append(appear_visual_ids == handle_id)
        mask = np.logical_or.reduce(masks)
        mask_image = (mask * 255).astype(np.uint8)
        cv2.imwrite(str(pose_dir / "handle_mask.png"), mask_image)

    env.close()


if __name__ == "__main__":
    config_dir = ASSET_DIR / "partnet_mobility_configs/fixed_cabinet_drawer_filtered"
    for s in sorted(config_dir.glob("*")):
        gen_single_scene(config_name=s.name[:-4])
