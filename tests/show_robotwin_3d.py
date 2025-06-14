import pickle
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

############################################
#  Basic helpers (minimal, no CLI)         #
############################################

def project_point(X_w: np.ndarray, K: np.ndarray, extr: np.ndarray) -> tuple[float, float]:
    """Transform *world* point → pixel coordinates.

    Parameters
    ----------
    X_w   : (3,)  XYZ in world frame [m]
    K     : (3,3) intrinsics (OpenCV)
    extr  : (3,4) [R|t] world→camera
    """
    X_c = extr @ np.append(X_w, 1.0)  # camera space
    if X_c[2] <= 0:
        raise ValueError("Point behind camera")
    uv_h = K @ X_c
    return float(uv_h[0] / uv_h[2]), float(uv_h[1] / uv_h[2])

############################################
#  Episode I/O                             #
############################################

def load_episode(pkl_path: str | Path):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    cam = d["observation"]["head_camera"]
    return (
        cam["rgb"],
        cam["intrinsic_cv"].astype(float),
        cam["extrinsic_cv"].astype(float),
        cam["cam2world_gl"].astype(float),
        cam["depth"].astype(np.float32)/1000.0,
        d["endpose"].astype(float),
    )

############################################
#  Main drawing routine                    #
############################################

def annotate_episode(
    pkl_path: str | Path,
    output: str | Path | None = None,
    show: bool = False,
):
    """Load episode, project end‑effectors, draw red *rings* on image."""

    img_rgb, K, extr, cam2world_gl, depth, endpose = load_episode(pkl_path)

    # 使用depth, rgb, K, cam2world_gl, endpose, 画出3d点云
    color_raw = o3d.geometry.Image(img_rgb)
    depth_raw = o3d.geometry.Image(depth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=1.0, depth_trunc=30, convert_rgb_to_intensity=False)
    # 使用intrinsic_cv, 画出3d点云
    inter = o3d.camera.PinholeCameraIntrinsic(img_rgb.shape[1], img_rgb.shape[0], K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    pcd = o3d.geometry.PointCloud().create_from_rgbd_image(rgbd_image, inter)
    # 把endpose转换到相机坐标系
    ep = endpose.reshape(2, 7)
    
    # 将endpose从世界坐标系转换到相机坐标系, 使用外参
    endpose_cam = extr @ np.vstack([ep[:, :3].T, np.ones(2)])
    endpose_cam = endpose_cam[:3].T  # 取前3行，转置回(2,3)形状
    
    # 创建endpose点云
    endpose_pcd = o3d.geometry.PointCloud()
    endpose_pcd.points = o3d.utility.Vector3dVector(endpose_cam)
    endpose_pcd.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0], [0, 1, 0]]))  # 红色和绿色点
    
    # 合并点云
    pcd += endpose_pcd
    
    # 可视化点云
    o3d.visualization.draw_geometries([pcd])
    
    if output:
        o3d.io.write_point_cloud(output, pcd)
        print(f"Saved → {output}")
    if show or not output:
        cv2.imshow("Annotation", img_rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

############################################
#  Quick usage                             #
############################################

if __name__ == "__main__":
    PKL = Path("/root/data/projects/embodiedai/RoboticsManipulation/RoboticsManipulation/data/dual_bottles_pick_hard_D435_pkl/episode0/100.pkl")
    annotate_episode(PKL, output="test.pcd", show=False)