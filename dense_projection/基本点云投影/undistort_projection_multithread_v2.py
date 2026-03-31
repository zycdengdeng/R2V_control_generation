#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dense Projection - 去畸变版投影：多线程CPU优化版
支持动静结合的点云投影
统一变换逻辑：世界坐标系 → LiDAR坐标系 → 相机坐标系
"""

import json
import yaml
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import argparse
from scipy.spatial.transform import Rotation as R
import warnings
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import sys
import os

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common_utils

warnings.filterwarnings('ignore', category=UserWarning)

# 车端相机配置
VEHICLE_CAMERAS = {
    1: {"name": "FN", "desc": "前视窄角30", "resolution": (3840, 2160)},
    2: {"name": "FW", "desc": "前视广角120", "resolution": (3840, 2160)},
    3: {"name": "FL", "desc": "左前视120", "resolution": (3840, 2160)},
    4: {"name": "FR", "desc": "右前视120", "resolution": (3840, 2160)},
    5: {"name": "RL", "desc": "左后视60", "resolution": (1920, 1080)},
    6: {"name": "RR", "desc": "右后视60", "resolution": (1920, 1080)},
    7: {"name": "RN", "desc": "后视60", "resolution": (1920, 1080)}
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
    import re

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


class UndistortProjectorMultiThread:
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
        cv2.setNumThreads(0)

    def get_world2lidar_transform(self, timestamp_ms):
        """
        获取 world2lidar 变换矩阵
        """
        transform = common_utils.find_closest_transform(timestamp_ms, self.transforms)

        if transform is None:
            raise ValueError(f"未找到时间戳 {timestamp_ms} 对应的变换矩阵")

        rotation = np.array(transform['world2lidar']['rotation']).reshape((3, 1))
        translation = np.array(transform['world2lidar']['translation']).reshape((3, 1))

        return rotation, translation

    def load_camera_params(self, cam_id):
        """加载车端相机参数"""
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

    def project_to_camera_undistorted(self, points, colors, rotate_world2lidar,
                                     trans_world2lidar, cam_id):
        """
        投影到去畸变的相机平面
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
            return np.zeros((img_h, img_w, 3), dtype=np.uint8), 0

        points_valid = points_cam[valid]
        colors_valid = colors[valid] if colors is not None else None

        # 步骤3: 相机坐标系 → 图像坐标系
        if cam_id in [2, 3, 4] and np.max(np.abs(D)) > 1:
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D[:4], (img_w, img_h), np.eye(3), balance=0.0
            )
        else:
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (img_w, img_h), 0, (img_w, img_h))

        uv_homogeneous = (new_K @ points_valid.T).T
        z_proj = uv_homogeneous[:, 2]
        uv = (uv_homogeneous[:, :2] / z_proj[:, np.newaxis]).astype(int)

        valid_proj = (uv[:, 0] >= 0) & (uv[:, 0] < img_w) & \
                    (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
        uv_valid = uv[valid_proj]

        img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        if len(uv_valid) > 0:
            proj_colors = (colors_valid[valid_proj] * 255).astype(np.uint8)
            for (u, v), color in zip(uv_valid, proj_colors):
                cv2.circle(img, (u, v), 2,
                         (int(color[2]), int(color[1]), int(color[0])), -1)

        return img, len(uv_valid)

    def process_single_camera(self, cam_id, points, colors, rotate_world2lidar,
                             trans_world2lidar, timestamp_ms, proj_dir, gt_dir,
                             compare_dir, overlay_dir):
        """处理单个相机"""
        cam_name = VEHICLE_CAMERAS[cam_id]['name']

        results = {'cam_name': cam_name, 'proj_img': None, 'gt_img': None, 'count': 0}

        # 处理GT图像
        gt_image_path = find_gt_image(self.gt_images_folder, cam_name, timestamp_ms)
        if gt_image_path:
            gt_output = gt_dir / f"{cam_name}.jpg"
            if self.undistort_gt_image(gt_image_path, cam_id, gt_output):
                results['gt_img'] = cv2.imread(str(gt_output))

        # 投影点云
        proj_img, count = self.project_to_camera_undistorted(
            points, colors, rotate_world2lidar, trans_world2lidar, cam_id
        )
        proj_output = proj_dir / f"{cam_name}.jpg"
        cv2.imwrite(str(proj_output), proj_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        results['proj_img'] = proj_img
        results['count'] = count

        # 生成compare图
        if results['gt_img'] is not None and results['proj_img'] is not None:
            gt_img = results['gt_img']
            compare_img = np.hstack([gt_img, proj_img])
            compare_output = compare_dir / f"{cam_name}.jpg"
            cv2.imwrite(str(compare_output), compare_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        # 生成overlay图
        if results['gt_img'] is not None and results['proj_img'] is not None:
            gt_img = results['gt_img']
            overlay_img = gt_img.copy()
            mask = np.any(proj_img > 10, axis=2)
            overlay_img[mask] = proj_img[mask]
            overlay_output = overlay_dir / f"{cam_name}.jpg"
            cv2.imwrite(str(overlay_output), overlay_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        return results

    def create_combined_view(self, images, output_path):
        """创建组合视图"""
        combined = np.zeros((720*3, 1280*3, 3), dtype=np.uint8)

        layout = {
            'FL': (0, 0),
            'FN': (0, 1),
            'FR': (0, 2),
            'FW': (1, 1),
            'RL': (2, 0),
            'RN': (2, 1),
            'RR': (2, 2)
        }

        for cam_name, img in images.items():
            if cam_name in layout:
                row, col = layout[cam_name]
                img_resized = cv2.resize(img, (1280, 720))
                combined[row*720:(row+1)*720, col*1280:(col+1)*1280] = img_resized

        cv2.imwrite(str(output_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 100])

    def process_single_frame(self, pcd_path, output_dir, timestamp_ms, num_threads=7):
        """
        处理单帧数据（从文件读取）

        Args:
            pcd_path: PCD文件路径
            output_dir: 输出目录
            timestamp_ms: 时间戳（毫秒）
            num_threads: 线程数
        """
        # 加载点云
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else np.ones((len(points), 3)) * 0.5

        return self.process_single_frame_with_data(
            points, colors, output_dir, timestamp_ms, num_threads
        )

    def process_single_frame_with_data(self, points, colors, output_dir, timestamp_ms, num_threads=7):
        """
        处理单帧数据（直接传入点云数据）

        用于 Dense Projection 的动静结合点云

        Args:
            points: 点云坐标 (N, 3)
            colors: 点云颜色 (N, 3)，范围 [0, 1]
            output_dir: 输出目录
            timestamp_ms: 时间戳（毫秒）
            num_threads: 线程数
        """
        output_dir = Path(output_dir)
        proj_dir = output_dir / "proj"
        gt_dir = output_dir / "gt"
        compare_dir = output_dir / "compare"
        overlay_dir = output_dir / "overlay"

        proj_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        compare_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

        # 确保 colors 不是 None
        if colors is None:
            colors = np.ones((len(points), 3)) * 0.5

        # 获取 world2lidar 变换
        try:
            rotate_world2lidar, trans_world2lidar = self.get_world2lidar_transform(timestamp_ms)
        except ValueError as e:
            print(f"  {e}")
            return False

        # 加载相机参数
        for cam_id in range(1, 8):
            if cam_id not in self.camera_params:
                self.load_camera_params(cam_id)

        # 多线程处理每个相机
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for cam_id in range(1, 8):
                future = executor.submit(
                    self.process_single_camera,
                    cam_id, points, colors, rotate_world2lidar,
                    trans_world2lidar, timestamp_ms, proj_dir, gt_dir,
                    compare_dir, overlay_dir
                )
                futures.append(future)

            for future in futures:
                result = future.result()

        return True


def main():
    parser = argparse.ArgumentParser(description="Dense Projection - 多线程投影")
    parser.add_argument("--roadside-calib", type=str, required=True)
    parser.add_argument("--vehicle-calib", type=str, required=True)
    parser.add_argument("--gt-images", type=str, required=True)
    parser.add_argument("--pcd", type=str, required=True)
    parser.add_argument("--transform-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--timestamp", type=int, required=True)
    parser.add_argument("--num-threads", type=int, default=7)

    args = parser.parse_args()

    transforms = common_utils.load_world2lidar_transforms(args.transform_json)

    projector = UndistortProjectorMultiThread(
        args.roadside_calib, args.vehicle_calib, args.gt_images, transforms
    )
    projector.process_single_frame(
        args.pcd, args.output_dir, args.timestamp, args.num_threads
    )


if __name__ == "__main__":
    main()
