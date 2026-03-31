#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDMap投影 V2 - 3D bbox → 2D bbox 投影
统一变换逻辑：世界坐标系 → LiDAR坐标系 → 相机坐标系
"""

import json
import yaml
import numpy as np
import cv2
from pathlib import Path
import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor
import sys
import os

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

# 颜色配置（20个类别 + unknown）
LABEL_COLORS = {
    "Car": [255, 0, 0],
    "Suv": [255, 69, 0],
    "Non_motor_rider": [255, 215, 0],
    "Bollards": [128, 128, 128],
    "Pedestrian": [0, 255, 0],
    "Crash_bucket": [192, 192, 192],
    "Tricycle": [135, 206, 250],
    "Truck": [255, 140, 0],
    "Motorcycle": [255, 0, 255],
    "Bus": [0, 0, 255],
    "Motor_rider": [255, 105, 180],
    "Pedestrian_else": [144, 238, 144],
    "Bicycle": [0, 255, 255],
    "Vehicle_else": [160, 82, 45],
    "Vehicle_door": [255, 192, 203],
    "Other_rider": [221, 160, 221],
    "Huge_vehicle": [139, 0, 0],
    "Unknown": [128, 128, 128],
    "Animal_small": [255, 228, 196],
    "Cone": [255, 255, 0],
    # 其他未知类别
    "unknown": [255, 0, 128]  # 紫红色，用于未定义的类别
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


def get_3d_bbox_corners(center, size, yaw):
    """计算3D bbox的8个角点（世界坐标系）"""
    x, y, z = center
    l, w, h = size

    # 局部坐标系的8个角点
    corners = np.array([
        [-l/2, -w/2, -h/2], [l/2, -w/2, -h/2],
        [l/2, w/2, -h/2], [-l/2, w/2, -h/2],
        [-l/2, -w/2, h/2], [l/2, -w/2, h/2],
        [l/2, w/2, h/2], [-l/2, w/2, h/2]
    ])

    # 旋转矩阵（绕z轴）
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    R = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ])

    # 旋转 + 平移
    corners_rotated = (R @ corners.T).T + center
    return corners_rotated


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


class HDMapProjectorMultiThread:
    def __init__(self, roadside_calib_path, vehicle_calib_folder, gt_images_folder, transforms):
        """
        初始化HDMap投影器

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

        Args:
            timestamp_ms: 时间戳（毫秒）

        Returns:
            rotation_vector: 旋转向量（罗德里格斯）
            translation: 平移向量
        """
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

        # 对于鱼眼相机（FL, FR, FW），使用特殊的去畸变方法
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

    def project_bbox_to_camera(self, bbox_corners_world, rotate_world2lidar,
                               trans_world2lidar, cam_id):
        """
        将3D bbox投影到相机并计算2D bbox

        Args:
            bbox_corners_world: 3D bbox的8个角点（世界坐标系）
            rotate_world2lidar: world2lidar旋转向量
            trans_world2lidar: world2lidar平移向量
            cam_id: 相机ID

        Returns:
            bbox_2d: [x1, y1, x2, y2] or None
            corners_2d: 投影后的角点坐标列表
        """
        cam_info = VEHICLE_CAMERAS[cam_id]
        img_w, img_h = cam_info["resolution"]

        K = self.camera_params[cam_id]['K']
        D = self.camera_params[cam_id]['D']
        R_cam2lidar = self.camera_poses[cam_id]['R']
        t_cam2lidar = self.camera_poses[cam_id]['t']

        # 步骤1: 世界坐标系 → LiDAR坐标系
        points_lidar = common_utils.transform_points_to_lidar(
            bbox_corners_world,
            {'world2lidar': {
                'rotation': rotate_world2lidar.flatten().tolist(),
                'translation': trans_world2lidar.flatten().tolist()
            }}
        )

        # 步骤2: LiDAR坐标系 → 相机坐标系
        R_lidar2cam = R_cam2lidar.T
        t_lidar2cam = -R_cam2lidar.T @ t_cam2lidar

        points_cam = (R_lidar2cam @ points_lidar.T).T + t_lidar2cam

        # 过滤相机前方的点
        valid_mask = points_cam[:, 2] > 0.1
        if not valid_mask.any():
            return None, None

        points_valid = points_cam[valid_mask]

        # 步骤3: 相机坐标系 → 图像坐标系（去畸变投影）
        if cam_id in [2, 3, 4] and np.max(np.abs(D)) > 1:
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D[:4], (img_w, img_h), np.eye(3), balance=0.0
            )
        else:
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (img_w, img_h), 0, (img_w, img_h))

        # 线性投影
        uv_homogeneous = (new_K @ points_valid.T).T
        z_proj = uv_homogeneous[:, 2]
        uv = uv_homogeneous[:, :2] / z_proj[:, np.newaxis]

        # 计算2D bbox
        x1, y1 = uv.min(axis=0)
        x2, y2 = uv.max(axis=0)

        # 检查是否在图像内
        if x2 < 0 or y2 < 0 or x1 > img_w or y1 > img_h:
            return None, None

        # 裁剪到图像范围
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)

        bbox_2d = [float(x1), float(y1), float(x2), float(y2)]
        corners_2d = uv.tolist()

        return bbox_2d, corners_2d

    def process_single_camera(self, cam_id, objects_data, rotate_world2lidar,
                             trans_world2lidar, timestamp_ms, gt_dir,
                             overlay_dir, bbox_on_gt_dir):
        """处理单个相机（用于多线程）"""
        cam_name = VEHICLE_CAMERAS[cam_id]['name']
        img_w, img_h = VEHICLE_CAMERAS[cam_id]['resolution']

        results = {'cam_name': cam_name, 'gt_img': None, 'bboxes': []}

        # 处理GT图像
        gt_image_path = find_gt_image(self.gt_images_folder, cam_name, timestamp_ms)
        if gt_image_path:
            gt_output = gt_dir / f"{cam_name}.jpg"
            if self.undistort_gt_image(gt_image_path, cam_id, gt_output):
                results['gt_img'] = cv2.imread(str(gt_output))
        else:
            # 找不到GT图像时打印警告
            print(f"    ⚠️  相机 {cam_name}: 未找到时间戳 {timestamp_ms} 的GT图像")

        # 投影所有物体的bbox
        for obj_data in objects_data:
            bbox_corners = obj_data['bbox_corners']
            bbox_2d, corners_2d = self.project_bbox_to_camera(
                bbox_corners, rotate_world2lidar, trans_world2lidar, cam_id
            )

            if bbox_2d is not None:
                results['bboxes'].append({
                    'id': obj_data['id'],
                    'label': obj_data['label'],
                    'color': obj_data['color'],
                    'bbox_2d': bbox_2d,
                    'corners_2d': corners_2d,
                    'bbox_3d': obj_data['bbox_3d']
                })

        # 生成纯bbox图（黑色背景 + 彩色bbox框）
        # 总是生成overlay：有bbox就绘制，没bbox就是纯黑色
        cam_info = VEHICLE_CAMERAS[cam_id]
        img_w, img_h = cam_info["resolution"]
        bbox_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)  # 黑色背景

        if results['bboxes']:
            # 如果有bbox，绘制在黑色背景上
            for bbox_info in results['bboxes']:
                x1, y1, x2, y2 = [int(v) for v in bbox_info['bbox_2d']]
                color = tuple(int(c) for c in bbox_info['color'])
                cv2.rectangle(bbox_img, (x1, y1), (x2, y2), color, 2)

        # 总是保存overlay（有bbox就是黑色+bbox，没bbox就是纯黑色）
        overlay_output = overlay_dir / f"{cam_name}.jpg"
        cv2.imwrite(str(overlay_output), bbox_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        # 生成bbox投影在GT上的图（用于检查框的准确性）
        # 总是生成bbox_on_gt：有GT就输出，有bbox就叠加，没bbox就是纯GT
        if results['gt_img'] is not None:
            bbox_on_gt_img = results['gt_img'].copy()

            if results['bboxes']:
                # 如果有bbox，绘制在GT图像上
                for bbox_info in results['bboxes']:
                    x1, y1, x2, y2 = [int(v) for v in bbox_info['bbox_2d']]
                    color = tuple(int(c) for c in bbox_info['color'])
                    cv2.rectangle(bbox_on_gt_img, (x1, y1), (x2, y2), color, 2)

            # 总是保存bbox_on_gt（有bbox就是GT+bbox，没bbox就是纯GT）
            bbox_on_gt_output = bbox_on_gt_dir / f"{cam_name}.jpg"
            cv2.imwrite(str(bbox_on_gt_output), bbox_on_gt_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        return results

    def process_single_frame(self, annotation_path, output_dir, timestamp_ms,
                           ego_vehicle_id=45, num_threads=7):
        """
        处理单帧数据（多线程）

        Args:
            annotation_path: 标注文件路径
            output_dir: 输出目录
            timestamp_ms: 时间戳（毫秒）
            ego_vehicle_id: 自车ID（将被排除）
            num_threads: 线程数
        """
        output_dir = Path(output_dir)
        gt_dir = output_dir / "gt"
        overlay_dir = output_dir / "overlay"
        bbox_on_gt_dir = output_dir / "bbox_on_gt"

        gt_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        bbox_on_gt_dir.mkdir(parents=True, exist_ok=True)

        # 1. 加载标注
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

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

        # 4. 准备所有物体数据（排除自车）
        objects_data = []
        for obj in annotation.get('object', []):
            if obj['id'] == ego_vehicle_id:
                continue

            # 计算3D bbox的8个角点
            center = [obj['x'], obj['y'], obj['z']]
            size = [obj['length'], obj['width'], obj['height']]
            yaw = obj['yaw']
            bbox_corners = get_3d_bbox_corners(center, size, yaw)

            objects_data.append({
                'id': obj['id'],
                'label': obj['label'],
                'color': LABEL_COLORS.get(obj['label'], LABEL_COLORS['unknown']),  # 未知类别使用紫红色
                'bbox_corners': bbox_corners,
                'bbox_3d': {
                    'x': obj['x'], 'y': obj['y'], 'z': obj['z'],
                    'length': obj['length'], 'width': obj['width'],
                    'height': obj['height'], 'yaw': obj['yaw']
                }
            })

        # 5. 多线程处理每个相机
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for cam_id in range(1, 8):
                future = executor.submit(
                    self.process_single_camera,
                    cam_id, objects_data, rotate_world2lidar,
                    trans_world2lidar, timestamp_ms, gt_dir,
                    overlay_dir, bbox_on_gt_dir
                )
                futures.append(future)

            # 收集结果
            for future in futures:
                result = future.result()

        return True


def main():
    parser = argparse.ArgumentParser(description="HDMap投影 V2 - 3D bbox → 2D bbox")
    parser.add_argument("--roadside-calib", type=str, required=True)
    parser.add_argument("--vehicle-calib", type=str, required=True)
    parser.add_argument("--gt-images", type=str, required=True)
    parser.add_argument("--annotation", type=str, required=True)
    parser.add_argument("--transform-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--timestamp", type=int, required=True)
    parser.add_argument("--ego-vehicle-id", type=int, default=45)
    parser.add_argument("--num-threads", type=int, default=7)

    args = parser.parse_args()

    # 加载变换矩阵
    transforms = common_utils.load_world2lidar_transforms(args.transform_json)

    projector = HDMapProjectorMultiThread(
        args.roadside_calib, args.vehicle_calib, args.gt_images, transforms
    )
    projector.process_single_frame(
        args.annotation, args.output_dir, args.timestamp,
        args.ego_vehicle_id, args.num_threads
    )

if __name__ == "__main__":
    main()
