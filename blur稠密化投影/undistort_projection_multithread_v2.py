#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
去畸变版投影：多线程CPU优化版 V2 - depth稠密化投影
统一变换逻辑：世界坐标系 → LiDAR坐标系 → 相机坐标系
输出深度图：近白远黑，无点区域纯黑色（.npy + .jpg）
深度图稠密化：形态学填充 → 引导滤波 → 最近邻插值 → 边缘保持平滑
"""

import json
import yaml
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import argparse
from scipy.spatial.transform import Rotation as R
from scipy import ndimage
import warnings
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import sys
import os
import re

# 添加父目录到路径以导入 common_utils（使用绝对路径）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common_utils

warnings.filterwarnings('ignore', category=UserWarning)

# 车端相机配置
VEHICLE_CAMERAS = {
    1: {"name": "FN", "desc": "前视窄角30°", "resolution": (3840, 2160)},
    2: {"name": "FW", "desc": "前视广角120°", "resolution": (3840, 2160)},
    3: {"name": "FL", "desc": "左前视120°", "resolution": (3840, 2160)},
    4: {"name": "FR", "desc": "右前视120°", "resolution": (3840, 2160)},
    5: {"name": "RL", "desc": "左后视60°", "resolution": (1920, 1080)},
    6: {"name": "RR", "desc": "右后视60°", "resolution": (1920, 1080)},
    7: {"name": "RN", "desc": "后视60°", "resolution": (1920, 1080)}
}

def quaternion_to_rotation_matrix(q):
    """四元数转旋转矩阵"""
    x, y, z, w = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
    ])
    return R


def find_gt_image(gt_images_folder, camera_name, timestamp_ms):
    """找到最接近的真值图片"""
    camera_folder = Path(gt_images_folder) / camera_name
    if not camera_folder.exists():
        return None

    target_timestamp_us = timestamp_ms * 1000

    jpg_files = list(camera_folder.glob("*.jpg"))
    closest_file = None
    min_diff = float('inf')

    for jpg_file in jpg_files:
        match = re.search(r'_(\d+)\.(\d+)\.jpg$', jpg_file.name)
        if match:
            seconds = int(match.group(1))
            microseconds = int(match.group(2))
            timestamp_us = seconds * 1000000 + microseconds

            diff = abs(timestamp_us - target_timestamp_us)
            if diff < min_diff:
                min_diff = diff
                closest_file = jpg_file

    return closest_file


class DepthDenseProjectorMultiThread:
    def __init__(self, roadside_calib_path, vehicle_calib_folder, gt_images_folder, transforms):
        """
        初始化投影器

        Args:
            roadside_calib_path: 路侧标定文件路径
            vehicle_calib_folder: 车端标定文件夹路径
            gt_images_folder: 真值图像文件夹
            transforms: world2lidar 变换矩阵列表
        """
        with open(roadside_calib_path, 'r') as f:
            self.roadside_calib = json.load(f)
        self.vehicle_calib_folder = Path(vehicle_calib_folder)
        self.gt_images_folder = Path(gt_images_folder)
        self.transforms = transforms
        self.camera_poses = {}
        self.camera_params = {}

        # 设置OpenCV线程数
        cv2.setNumThreads(0)  # 让每个线程独立使用OpenCV

    def get_world2lidar_transform(self, timestamp_ms):
        """获取 world2lidar 变换矩阵"""
        transform = common_utils.find_closest_transform(timestamp_ms, self.transforms)

        if transform is None:
            raise ValueError(f"未找到时间戳 {timestamp_ms} 对应的变换矩阵")

        rotation = np.array(transform['world2lidar']['rotation']).reshape((3, 1))
        translation = np.array(transform['world2lidar']['translation']).reshape((3, 1))

        return rotation, translation

    def load_camera_params(self, cam_id):
        """加载车端相机参数（使用固定标定路径）"""
        with open(self.vehicle_calib_folder / f"camera_{cam_id:02d}_intrinsics.yaml", 'r') as f:
            intrinsics = yaml.safe_load(f)
        K = np.array(intrinsics['K']).reshape(3, 3)
        D = np.array(intrinsics['D'])

        with open(self.vehicle_calib_folder / f"camera_{cam_id:02d}_extrinsics.yaml", 'r') as f:
            extrinsics = yaml.safe_load(f)

        transform = extrinsics['transform']
        q = [transform['rotation']['x'], transform['rotation']['y'],
             transform['rotation']['z'], transform['rotation']['w']]
        t = np.array([transform['translation']['x'],
                     transform['translation']['y'],
                     transform['translation']['z']])
        R_cam = quaternion_to_rotation_matrix(q)

        self.camera_poses[cam_id] = {
            'R': R_cam,
            't': t,
            'label': extrinsics.get('label', ''),
            'name': VEHICLE_CAMERAS[cam_id]['name']
        }

        self.camera_params[cam_id] = {
            'K': K,
            'D': D,
            'resolution': VEHICLE_CAMERAS[cam_id]["resolution"]
        }

        return K, D, R_cam, t

    def undistort_gt_image(self, gt_image_path, cam_id, output_path):
        """对真值图像进行去畸变"""
        if not gt_image_path or not gt_image_path.exists():
            return False

        img = cv2.imread(str(gt_image_path))
        if img is None:
            return False

        K = self.camera_params[cam_id]['K']
        D = self.camera_params[cam_id]['D']
        w, h = self.camera_params[cam_id]['resolution']

        if cam_id in [2, 3, 4] and np.max(np.abs(D)) > 1:
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D[:4], (w, h), np.eye(3), balance=0.0
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D[:4], np.eye(3), new_K, (w, h), cv2.CV_16SC2
            )
            undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
        else:
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0, (w, h))
            undistorted = cv2.undistort(img, K, D, None, new_K)

        cv2.imwrite(str(output_path), undistorted, [cv2.IMWRITE_JPEG_QUALITY, 100])
        return True

    def densify_depth_image(self, depth_raw):
        """
        深度图稠密化处理（4级策略）- 向量化优化版本

        级别1: 形态学填充（小空洞 < 5x5）- 使用卷积计算邻域平均
        级别2: 邻域平均填充（中等空洞）- 使用卷积计算邻域平均
        级别3: 最近邻插值（大空洞）- 使用距离变换一次性完成
        级别4: 轻度平滑 - 使用双边滤波

        Args:
            depth_raw: 原始深度图 (float32)

        Returns:
            depth_dense: 稠密化后的深度图 (float32)
        """
        depth_dense = depth_raw.copy()
        valid_mask = depth_raw > 0

        if np.sum(valid_mask) == 0:
            return depth_dense

        h, w = depth_raw.shape

        # ====== 级别1: 形态学填充（小空洞）使用向量化 ======
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        depth_binary = (depth_dense > 0).astype(np.uint8) * 255
        depth_binary_closed = cv2.morphologyEx(depth_binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 对新填充的区域，用3x3邻域平均填充（向量化版本）
        new_filled_mask = (depth_binary_closed > 0) & (depth_dense == 0)
        if np.any(new_filled_mask):
            # 创建一个权重矩阵，用于计算加权平均
            # 使用卷积计算邻域和与邻域计数
            kernel_3x3 = np.ones((3, 3), dtype=np.float32)

            # 计算邻域深度和
            depth_sum = cv2.filter2D(depth_dense, -1, kernel_3x3, borderType=cv2.BORDER_CONSTANT)

            # 计算邻域有效像素数（depth > 0 的像素）
            valid_binary = (depth_dense > 0).astype(np.float32)
            valid_count = cv2.filter2D(valid_binary, -1, kernel_3x3, borderType=cv2.BORDER_CONSTANT)

            # 计算平均值（避免除零）
            avg_depth = np.zeros_like(depth_dense)
            valid_avg_mask = valid_count > 0
            avg_depth[valid_avg_mask] = depth_sum[valid_avg_mask] / valid_count[valid_avg_mask]

            # 只填充需要填充的区域
            depth_dense[new_filled_mask] = avg_depth[new_filled_mask]

        # ====== 级别2: 5x5邻域平均填充（中等空洞）使用向量化 ======
        for iteration in range(2):
            depth_binary = (depth_dense > 0).astype(np.uint8) * 255
            depth_binary_dilated = cv2.dilate(depth_binary, np.ones((5, 5), np.uint8), iterations=1)
            new_filled_mask = (depth_binary_dilated > 0) & (depth_dense == 0)

            if np.any(new_filled_mask):
                # 使用5x5卷积核计算邻域平均（向量化版本）
                kernel_5x5 = np.ones((5, 5), dtype=np.float32)

                # 计算邻域深度和
                depth_sum = cv2.filter2D(depth_dense, -1, kernel_5x5, borderType=cv2.BORDER_CONSTANT)

                # 计算邻域有效像素数
                valid_binary = (depth_dense > 0).astype(np.float32)
                valid_count = cv2.filter2D(valid_binary, -1, kernel_5x5, borderType=cv2.BORDER_CONSTANT)

                # 计算平均值
                avg_depth = np.zeros_like(depth_dense)
                valid_avg_mask = valid_count > 0
                avg_depth[valid_avg_mask] = depth_sum[valid_avg_mask] / valid_count[valid_avg_mask]

                # 只填充需要填充的区域
                depth_dense[new_filled_mask] = avg_depth[new_filled_mask]

        # ====== 级别3: 最近邻插值（大空洞，距离<20像素）使用距离变换 ======
        invalid_mask = depth_dense == 0
        if np.any(invalid_mask):
            valid_mask_level3 = depth_dense > 0
            if np.any(valid_mask_level3):
                # 使用距离变换一次性找到所有无效像素的最近有效像素
                # distance_transform_edt 返回到最近有效像素的距离和索引
                distances, indices = ndimage.distance_transform_edt(
                    invalid_mask,
                    return_distances=True,
                    return_indices=True
                )

                # 只填充距离 < 20 像素的区域
                fill_mask = invalid_mask & (distances < 20)

                if np.any(fill_mask):
                    # indices[0] 是行索引，indices[1] 是列索引
                    nearest_y = indices[0][fill_mask]
                    nearest_x = indices[1][fill_mask]

                    # 从最近的有效像素复制深度值
                    depth_dense[fill_mask] = depth_dense[nearest_y, nearest_x]

        # ====== 级别4: 轻度双边滤波平滑 ======
        valid_mask_final = depth_dense > 0
        if np.sum(valid_mask_final) > 100:
            # 只对有效区域进行轻度平滑
            depth_temp = depth_dense.copy()
            depth_smoothed = cv2.bilateralFilter(
                depth_temp.astype(np.float32), d=5, sigmaColor=10, sigmaSpace=5
            )
            depth_dense[valid_mask_final] = depth_smoothed[valid_mask_final]

        # 保证无点区域仍为纯黑色
        depth_dense[depth_dense <= 0] = 0

        return depth_dense

    def project_to_camera_depth(self, points, colors, rotate_world2lidar,
                                trans_world2lidar, cam_id):
        """
        投影到去畸变的相机平面并生成深度图

        深度图要求：近白远黑，无点区域纯黑色
        变换流程：世界坐标系 → LiDAR坐标系 → 相机坐标系 → 图像坐标系
        """
        cam_info = VEHICLE_CAMERAS[cam_id]
        img_w, img_h = cam_info["resolution"]

        K = self.camera_params[cam_id]['K']
        D = self.camera_params[cam_id]['D']
        R_cam2lidar = self.camera_poses[cam_id]['R']
        t_cam2lidar = self.camera_poses[cam_id]['t']

        # 步骤1: 世界坐标系 → LiDAR坐标系
        points_lidar = common_utils.transform_points_to_lidar(
            points[:, :3],
            {'world2lidar': {
                'rotation': rotate_world2lidar.flatten().tolist(),
                'translation': trans_world2lidar.flatten().tolist()
            }}
        )

        # 步骤2: LiDAR坐标系 → 相机坐标系
        R_lidar2cam = R_cam2lidar.T
        t_lidar2cam = -R_cam2lidar.T @ t_cam2lidar

        points_cam = (R_lidar2cam @ points_lidar.T).T + t_lidar2cam

        # 过滤背后的点
        valid = points_cam[:, 2] > 0.1
        if not np.any(valid):
            depth_raw = np.zeros((img_h, img_w), dtype=np.float32)
            depth_vis = np.zeros((img_h, img_w), dtype=np.uint8)
            return depth_raw, depth_vis, 0

        points_valid = points_cam[valid]
        depths = points_valid[:, 2]  # z值即为深度

        # 步骤3: 相机坐标系 → 图像坐标系（去畸变投影）
        if cam_id in [2, 3, 4] and np.max(np.abs(D)) > 1:
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D[:4], (img_w, img_h), np.eye(3), balance=0.0
            )
        else:
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (img_w, img_h), 0, (img_w, img_h))

        uv_homogeneous = (new_K @ points_valid.T).T
        z_proj = uv_homogeneous[:, 2]
        uv = (uv_homogeneous[:, :2] / z_proj[:, np.newaxis]).astype(int)

        # 过滤有效投影点
        valid_proj = (uv[:, 0] >= 0) & (uv[:, 0] < img_w) & \
                    (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
        uv_valid = uv[valid_proj]
        depths_valid = depths[valid_proj]

        # 创建深度图（原始值）
        depth_raw = np.zeros((img_h, img_w), dtype=np.float32)

        if len(uv_valid) > 0:
            # 使用深度缓冲，保留最近的深度值
            for (u, v), depth in zip(uv_valid, depths_valid):
                if depth_raw[v, u] == 0 or depth < depth_raw[v, u]:
                    depth_raw[v, u] = depth

        # 进行稠密化处理
        depth_dense = self.densify_depth_image(depth_raw)

        # 创建可视化深度图：近白远黑
        depth_vis = np.zeros((img_h, img_w), dtype=np.uint8)
        valid_mask = depth_dense > 0

        if np.any(valid_mask):
            min_depth = np.min(depth_dense[valid_mask])
            max_depth = np.max(depth_dense[valid_mask])

            if max_depth > min_depth:
                # 近白远黑：depth越小，值越大（白色）
                depth_normalized = (max_depth - depth_dense[valid_mask]) / (max_depth - min_depth)
                depth_vis[valid_mask] = (depth_normalized * 255).astype(np.uint8)
            else:
                # 所有深度相同，设为中灰色
                depth_vis[valid_mask] = 128

        return depth_dense, depth_vis, len(uv_valid)

    def process_single_camera(self, cam_id, points, colors, rotate_world2lidar,
                             trans_world2lidar, timestamp_ms, depth_dir, gt_dir,
                             compare_dir, overlay_dir):
        """处理单个相机（用于多线程）"""
        cam_name = VEHICLE_CAMERAS[cam_id]['name']

        results = {'cam_name': cam_name, 'depth_vis': None, 'gt_img': None, 'count': 0}

        # 处理GT图像
        gt_image_path = find_gt_image(self.gt_images_folder, cam_name, timestamp_ms)
        if gt_image_path:
            gt_output = gt_dir / f"{cam_name}.jpg"
            if self.undistort_gt_image(gt_image_path, cam_id, gt_output):
                results['gt_img'] = cv2.imread(str(gt_output))

        # 投影生成深度图
        depth_raw, depth_vis, count = self.project_to_camera_depth(
            points, colors, rotate_world2lidar, trans_world2lidar, cam_id
        )

        # 保存深度图（.npy原始值 + .jpg可视化）
        depth_npy_output = depth_dir / f"{cam_name}.npy"
        depth_jpg_output = depth_dir / f"{cam_name}.jpg"
        np.save(str(depth_npy_output), depth_raw)
        cv2.imwrite(str(depth_jpg_output), depth_vis, [cv2.IMWRITE_JPEG_QUALITY, 100])

        results['depth_vis'] = depth_vis
        results['count'] = count

        # 生成compare图（GT和深度图左右对比）
        if results['gt_img'] is not None:
            gt_img = results['gt_img']
            # 将深度图转换为3通道用于对比
            depth_vis_color = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
            compare_img = np.hstack([gt_img, depth_vis_color])
            compare_output = compare_dir / f"{cam_name}.jpg"
            cv2.imwrite(str(compare_output), compare_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        # 生成overlay图（深度图叠加到GT上，使用伪彩色）
        if results['gt_img'] is not None:
            gt_img = results['gt_img']
            # 将深度图应用伪彩色
            depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            # 叠加：有深度的地方用彩色深度图
            overlay_img = gt_img.copy()
            mask = depth_vis > 0
            overlay_img[mask] = depth_colored[mask]
            overlay_output = overlay_dir / f"{cam_name}.jpg"
            cv2.imwrite(str(overlay_output), overlay_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        return results

    def process_single_frame(self, pcd_path, output_dir, timestamp_ms, num_threads=7):
        """
        处理单帧数据（多线程）

        Args:
            pcd_path: PCD文件路径
            output_dir: 输出目录
            timestamp_ms: 时间戳（毫秒）
            num_threads: 线程数
        """
        output_dir = Path(output_dir)
        depth_dir = output_dir / "depth"  # 深度图目录
        gt_dir = output_dir / "gt"
        compare_dir = output_dir / "compare"
        overlay_dir = output_dir / "overlay"

        depth_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        compare_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

        # 1. 加载点云
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else np.ones((len(points), 3)) * 0.5

        # 2. 获取 world2lidar 变换
        try:
            rotate_world2lidar, trans_world2lidar = self.get_world2lidar_transform(timestamp_ms)
        except ValueError as e:
            print(f"❌ {e}")
            return False

        # 3. 加载相机参数
        for cam_id in range(1, 8):
            if cam_id not in self.camera_params:
                self.load_camera_params(cam_id)

        # 4. 多线程处理每个相机
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for cam_id in range(1, 8):
                future = executor.submit(
                    self.process_single_camera,
                    cam_id, points, colors, rotate_world2lidar,
                    trans_world2lidar, timestamp_ms, depth_dir, gt_dir,
                    compare_dir, overlay_dir
                )
                futures.append(future)

            # 收集结果
            for future in futures:
                result = future.result()

        return True


def main():
    parser = argparse.ArgumentParser(description="多线程优化去畸变版投影 V2 - Depth稠密化投影")
    parser.add_argument("--roadside-calib", type=str, required=True)
    parser.add_argument("--vehicle-calib", type=str, required=True)
    parser.add_argument("--gt-images", type=str, required=True)
    parser.add_argument("--pcd", type=str, required=True)
    parser.add_argument("--transform-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--timestamp", type=int, required=True)
    parser.add_argument("--num-threads", type=int, default=7, help="每帧使用的线程数")

    args = parser.parse_args()

    transforms = common_utils.load_world2lidar_transforms(args.transform_json)

    projector = DepthDenseProjectorMultiThread(
        args.roadside_calib, args.vehicle_calib, args.gt_images, transforms
    )
    projector.process_single_frame(
        args.pcd, args.output_dir, args.timestamp, args.num_threads
    )

if __name__ == "__main__":
    main()
