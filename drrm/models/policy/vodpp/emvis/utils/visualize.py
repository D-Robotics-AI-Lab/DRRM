import os
from torchvision.utils import save_image
from tqdm import tqdm
import cv2
import torch
import numpy as np

def save_depth(depth, path):
    # depth: [B,H,W,C]
    depth_norm = ((depth-depth.min())/(depth.max()-depth.min())).permute(0,3,1,2)
    save_image(depth_norm, path)

def save_images(images, path):
    # images: [B,H,W,C]
    save_image(images, path)

def process_and_save(point_cloud, rgb_image, filename, original_height):
    """
    处理点云和RGB图像，拼接后存储为点云文件
    参数:
        point_cloud: [V, H, W, 3] 点云张量
        rgb_image: [V, 3, H, W] RGB图像张量
        filename: 保存的文件名
        original_height: 原始高度H
    """
    import open3d as o3d
    # 获取张量形状信息
    V, H, W, C = point_cloud.shape
    _, C_img, H_img, W_img = rgb_image.shape
    
    # 验证输入形状一致性
    assert H == H_img and W == W_img, "点云和图像的空间维度不匹配"
    assert C == 3 and C_img == 3, "通道数必须为3"

    # 1. 点云拼接: [V, H, W, 3] -> [H, V*W, 3]
    point_cloud_reshaped = point_cloud.permute(1, 0, 2, 3).reshape(H, V*W, 3)
    
    # 2. 图像拼接: [V, 3, H, W] -> [3, H, V*W]
    image_reshaped = rgb_image.permute(0, 2, 3, 1)  # [V, H, W, 3]
    image_reshaped = image_reshaped.permute(1, 0, 2, 3).reshape(H, V*W, 3)  # [H, V*W, 3]
    image_reshaped = image_reshaped.permute(2, 0, 1)  # [3, H, V*W]
    
    # 3. 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        point_cloud_reshaped.reshape(-1, 3).cpu().float().numpy()
    )
    pcd.colors = o3d.utility.Vector3dVector(
        image_reshaped.permute(1, 2, 0).reshape(-1, 3).cpu().float().numpy()
    )
    
    # 4. 保存点云文件并存储原始高度信息
    o3d.io.write_point_cloud(filename, pcd)
    with open(filename + ".meta", "w") as f:
        f.write(f"{original_height}")
    
    return point_cloud_reshaped, image_reshaped

def load_and_restore(filename):
    """
    从点云文件加载并还原原始张量
    参数:
        filename: 点云文件名
    返回:
        point_cloud_tensor: [H, V*W, 3] 点云张量
        rgb_tensor: [3, H, V*W] RGB张量
    """
    import open3d as o3d
    # 1. 加载点云和元数据
    pcd = o3d.io.read_point_cloud(filename)
    with open(filename + ".meta", "r") as f:
        original_height = int(f.readline().strip())
    
    # 2. 转换为numpy数组
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # 3. 计算第二维度大小
    total_points = points.shape[0]
    VW = total_points // original_height
    
    # 4. 还原点云张量: [H, V*W, 3]
    point_cloud_tensor = torch.tensor(points).reshape(original_height, VW, 3)
    
    # 5. 还原RGB张量: [3, H, V*W]
    colors_reshaped = torch.tensor(colors).reshape(original_height, VW, 3)
    rgb_tensor = colors_reshaped.permute(2, 0, 1)
    
    return point_cloud_tensor, rgb_tensor

def render_point_cloud_video(
    point_cloud,       # [S, H, W, 3] torch.Tensor
    rgb_images,        # [S, 3, H, W] torch.Tensor
    output_path='output.mp4',
    fps=10,
    camera_params=(1.0, 0.0, 0.0),  # (r, theta, phi)
):
    """
    渲染点云序列为视频
    
    参数:
        point_cloud: 时序点云数据 [S, H, W, 3]
        rgb_images: 对应的RGB图像 [S, 3, H, W] (值范围0-1)
        output_path: 输出视频路径
        fps: 视频帧率
        camera_params: 相机参数元组 (r, theta, phi)
    """
    from pytorch3d.structures import Pointclouds
    from pytorch3d.renderer import (
        look_at_rotation,
        FoVPerspectiveCameras,
        PointsRasterizationSettings,
        PointsRasterizer,
        PointsRenderer,
        AlphaCompositor
    )
    # 检查设备
    device = point_cloud.device
    S, H, W, _ = point_cloud.shape
    
    # 计算第一帧点云中心作为场景中心
    center = point_cloud[0].reshape(-1, 3).mean(dim=0, keepdim=True)  # [1, 3]
    
    # 将点云中心平移到原点
    point_cloud_centered = point_cloud - center
    
    # 提取相机参数
    r, theta, phi = camera_params
    

    def get_RT(r, theta, phi):
        # 将球坐标转换为笛卡尔坐标
        theta = torch.deg2rad(torch.tensor(theta))
        phi = torch.deg2rad(torch.tensor(phi))
        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)
        camera_position = torch.tensor([[x, y, z]], device=device)  # [1, 3]
        
        # 计算相机旋转矩阵（看向原点）
        R = look_at_rotation(camera_position, device=device)  # [1, 3, 3]
        T = -torch.bmm(R.transpose(1, 2), camera_position[:, :, None])[:, :, 0]  # [1, 3]
        return R, T
    
    R, T = get_RT(r, theta, phi)
    # 创建相机对象
    cameras = FoVPerspectiveCameras(
        R=R, 
        T=T,
        device=device,
        znear=0.01 * r,
        zfar=10 * r,
        fov=60  # 可调整视场角
    )
    
    # 点云渲染器配置
    raster_settings = PointsRasterizationSettings(
        image_size=(H, W),
        radius=0.008,  # 点半径（根据点云分布调整）
        points_per_pixel=10,  # 每个像素考虑的点数
        bin_size=None if H <= 256 else 0  # 大尺寸优化
    )
    
    # 创建渲染器
    rasterizer = PointsRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    )
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        output_path, 
        fourcc, 
        fps, 
        (W, H)
    )
    
    # 逐帧渲染点云
    for i in tqdm(range(S)):
        # R, T = get_RT(r, i*(36/S), phi)
        # cameras.R = R.to(device)
        # cameras.T = T.to(device)

        # 准备当前帧点云和颜色
        points = point_cloud_centered[i].reshape(1, -1, 3)  # [1, H*W, 3]
        colors = rgb_images[i].permute(1, 2, 0).reshape(1, -1, 3)  # [1, H*W, 3]
        
        # 创建点云对象
        pcl = Pointclouds(points=points, features=colors)
        
        # 渲染点云 (PyTorch3D输出: [1, H, W, 4] RGBA)
        rendered_image = renderer(pcl)[0]  # [H, W, 4]
        
        # 提取RGB通道并转换为OpenCV格式
        rgb = rendered_image[..., :3].cpu().numpy()  # [H, W, 3]
        bgr = (rgb * 255).astype(np.uint8)[..., ::-1]  # 转换为BGR
        
        # 写入视频帧
        video_writer.write(bgr)
    
    # 释放视频资源
    video_writer.release()
    print(f"视频已保存至: {output_path}")

def reproject_point_cloud(
    point_cloud,       # [V, H, W, 3] 相机1坐标系下的点云 (x, y, z)
    cam1_ext,          # [V, 3, 4] 相机1外参 (R|t)
    cam1_int,          # [V, 3, 3] 相机1内参
    cam2_ext,          # [V, 3, 4] 相机2外参
    cam2_int           # [V, 3, 3] 相机2内参
):
    V, H, W, _ = point_cloud.shape
    device = point_cloud.device
    
    # 1. 将点云从相机1的世界系->相机1的相机系
    # 扩展为齐次坐标 [V, H*W, 4]
    points_hom = torch.cat([point_cloud, torch.ones(V, H, W, 1, device=device)], dim=-1).reshape(V, H * W, 4)
    # 扩展外参矩阵为齐次形式[V, 4, 4]，第三列补0，最后一行补[0,0,0,1]
    extr_hom = torch.cat([cam1_ext, torch.tensor([0,0,0,1], device=cam1_ext.device).repeat(V,1,1)], dim=1)
    # 计算逆变换矩阵：从世界坐标系到相机坐标系
    extr_inv = torch.inverse(extr_hom)  # 形状[V, 4, 4]
    # 应用变换矩阵：cam_coords = T_world_to_cam * world_coords
    cam_points_hom = torch.einsum("vij,vkj->vki", extr_inv, points_hom)  # 形状[V, H*W, 4]
    # 转换回非齐次坐标并重塑为原始形状
    cam_points = cam_points_hom[..., :3] / cam_points_hom[..., 3:4]  # 形状[V, H*W, 3]

    point_cloud2 = []
    # 从外参矩阵提取旋转部分R和平移部分t
    for v in range(len(cam2_ext)):
        points = cam_points[v]
        points = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=1)
        point_cloud2.append(torch.mm(points, cam2_ext[v].T))
    point_cloud2 = torch.stack(point_cloud2).view(V, H, W, 3)
    
    # 恢复原始形状 [V, H, W, 3]
    return point_cloud2


if __name__ == "__main__":
    # S, H, W = 10, 256, 256
    # point_cloud = torch.rand(S, H, W, 3) * 10  # [S, H, W, 3]
    # rgb_images = torch.rand(S, 3, H, W)  # [S, 3, H, W]
    extri_trans = False
    for ep in os.listdir("./out/debug_img_518/vis"):
        base_path = os.path.join("./out/debug_img_518/vis", ep)
        pcd_path = os.path.join(base_path, "meta")
        point_cloud = []
        rgb_images = []
        files = [f for f in os.listdir(pcd_path) if '.pt' in f]
        files = sorted(files)
        extr, intr = None, None
        for file in files:
            if not '.pt' in file: continue
            path = os.path.join(pcd_path, file)
            tensors = torch.load(path)
            pcd, rgb = tensors['pcd'], tensors['rgb']
            if extr == None:
                extr, intr = tensors['extri'], tensors['intri']
            elif extri_trans:
                pcd = reproject_point_cloud(pcd, tensors['extri'], tensors['intri'], extr, intr)

            # x = pcd[...,0]
            # y = pcd[...,1]
            # z = pcd[...,2]
            # pcd[...,0] = (x - x.mean())
            # pcd[...,1] = (y - y.mean())
            # pcd[...,2] = (z - z.mean())+1
            point_cloud.append(pcd)
            rgb_images.append(rgb)
        
        # 极坐标系控制相机（原点在点云中心）
        point_cloud = torch.cat(point_cloud).to(dtype=torch.float32)
        m = torch.tensor([
            [1.,0.,0.],
            [0.,-1.,0.],
            [0.,0.,-1.]
        ]).to(device=point_cloud.device)
        point_cloud = point_cloud@m.T
        rgb_images = torch.cat(rgb_images).to(dtype=torch.float32)

        # 相机位置（极坐标）
        r = .8       # 径向距离
        theta = 20  # 天顶角
        phi = 0    # 90度（方位角）
        # 渲染视频
        render_point_cloud_video(
            point_cloud, 
            rgb_images,
            output_path = os.path.join(base_path, f"{ep}_{theta}_point_cloud_video.mp4"),
            fps = 10,
            camera_params = (r, theta, phi),
        )