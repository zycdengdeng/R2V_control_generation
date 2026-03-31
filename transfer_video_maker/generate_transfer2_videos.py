#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 Transfer2 训练数据格式的视频

从投影输出结果（如 depth投影、HDMap投影等）生成符合 transfer2 格式的视频数据集
"""

import json
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

# 相机名称映射：我的相机名 → Transfer2 相机名
CAMERA_NAME_MAPPING = {
    'FN': 'ftheta_camera_front_tele_30fov',      # 前视窄角30°
    'FW': 'ftheta_camera_front_wide_120fov',     # 前视广角120°
    'FL': 'ftheta_camera_cross_left_120fov',     # 左前视120°
    'FR': 'ftheta_camera_cross_right_120fov',    # 右前视120°
    'RL': 'ftheta_camera_rear_left_70fov',       # 左后视60° (transfer叫70fov)
    'RR': 'ftheta_camera_rear_right_70fov',      # 右后视60° (transfer叫70fov)
    'RN': 'ftheta_camera_rear_tele_30fov'        # 后视60°
}

# 相机列表（按顺序）
CAMERA_NAMES = ['FN', 'FW', 'FL', 'FR', 'RL', 'RR', 'RN']


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='生成 Transfer2 格式视频数据集')

    parser.add_argument('--project-type', type=str, required=True,
                        choices=['depth', 'depth_dense', 'hdmap', 'blur', 'blur_dense', 'basic'],
                        help='项目类型（如 depth, depth_dense, hdmap 等）')

    parser.add_argument('--project-root', type=str,
                        default=str(Path(__file__).resolve().parent.parent),
                        help='项目根目录')

    parser.add_argument('--scenes', type=str, nargs='+', required=True,
                        help='场景ID列表（如 002 004 006）')

    parser.add_argument('--frames-per-seg', type=int, required=True,
                        help='每个seg的帧数（如 21）')

    parser.add_argument('--num-segs', type=int, required=True,
                        help='每个场景生成的seg数量（如 4）')

    parser.add_argument('--fps', type=int, default=10,
                        help='视频帧率（默认 10 FPS）')

    parser.add_argument('--output-dir', type=str, required=True,
                        help='输出目录（将创建 Transfer2 格式的目录结构）')

    parser.add_argument('--control-subdir', type=str, required=True,
                        help='控制输入图像子目录名（如 depth, compare, proj）')

    parser.add_argument('--control-input-type', type=str, required=True,
                        help='控制输入类型（如 depth, hdmap_bbox）')

    parser.add_argument('--caption-template', type=str, default='',
                        help='Caption 模板（如 "A depth map from camera {camera}"）')

    return parser.parse_args()


def get_sorted_timestamp_folders(scene_dir):
    """
    获取场景目录下所有时间戳文件夹，按时间戳排序

    Args:
        scene_dir: 场景目录路径

    Returns:
        sorted_folders: 排序后的时间戳文件夹列表
    """
    scene_path = Path(scene_dir)

    if not scene_path.exists():
        print(f"警告: 场景目录不存在: {scene_dir}")
        return []

    # 查找所有数字文件夹
    timestamp_folders = []
    for folder in scene_path.iterdir():
        if folder.is_dir() and folder.name.isdigit():
            timestamp_folders.append(folder)

    # 按时间戳（文件夹名）排序
    timestamp_folders.sort(key=lambda x: int(x.name))

    return timestamp_folders


def select_middle_frames(timestamp_folders, total_frames):
    """
    从时间戳文件夹列表中选取中间的 N 帧

    Args:
        timestamp_folders: 所有时间戳文件夹列表
        total_frames: 需要选取的总帧数

    Returns:
        selected_folders: 选中的时间戳文件夹列表
    """
    total_available = len(timestamp_folders)

    if total_available < total_frames:
        print(f"警告: 可用帧数 {total_available} 少于需求 {total_frames}，将使用所有可用帧")
        return timestamp_folders

    # 计算中间位置
    start_idx = (total_available - total_frames) // 2
    end_idx = start_idx + total_frames

    selected = timestamp_folders[start_idx:end_idx]

    print(f"  可用帧数: {total_available}")
    print(f"  选取范围: [{start_idx}, {end_idx}) (中间 {total_frames} 帧)")
    print(f"  时间戳范围: {selected[0].name} ~ {selected[-1].name}")

    return selected


def create_video_from_images(image_paths, output_path, fps, target_resolution=(1280, 720)):
    """
    从图像列表创建视频（统一分辨率为1280×720）

    Args:
        image_paths: 图像路径列表
        output_path: 输出视频路径
        fps: 帧率
        target_resolution: 目标分辨率 (width, height)，默认1280×720
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 初始化视频写入器（使用目标分辨率）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        target_resolution
    )

    # 写入帧
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"警告: 无法读取图像 {img_path}")
            continue

        # 缩放到目标分辨率（1280×720）
        if img.shape[1] != target_resolution[0] or img.shape[0] != target_resolution[1]:
            img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR)

        video_writer.write(img)

    video_writer.release()


def create_caption_json(scene_id, seg_id, camera_name, caption_template):
    """
    创建 caption JSON 文件内容

    Args:
        scene_id: 场景ID
        seg_id: Seg ID
        camera_name: 相机名称（Transfer2格式）
        caption_template: Caption 模板

    Returns:
        caption_dict: Caption 字典
    """
    if caption_template:
        caption = caption_template.format(
            scene=scene_id,
            seg=seg_id,
            camera=camera_name
        )
    else:
        caption = f"Scene {scene_id} segment {seg_id} from {camera_name}"

    return {
        "scene_id": scene_id,
        "segment_id": f"seg{seg_id:02d}",
        "camera": camera_name,
        "caption": caption
    }


def process_single_scene(args, scene_id):
    """
    处理单个场景

    Args:
        args: 命令行参数
        scene_id: 场景ID
    """
    print(f"\n{'='*60}")
    print(f"处理场景: {scene_id}")
    print(f"{'='*60}")

    # 构建场景目录路径
    project_dir_map = {
        'depth': 'depth投影',
        'depth_dense': 'depth稠密化投影',
        'hdmap': 'HDMap投影',
        'blur': 'blur投影',
        'blur_dense': 'blur稠密化投影',
        'basic': '基本点云投影'
    }

    project_dir = project_dir_map.get(args.project_type, args.project_type)
    # 路径结构: project_root / project_dir / scene_id / scene_id / timestamp
    # 例如: blur投影/002/002/1742877430866/gt
    scene_dir = Path(args.project_root) / project_dir / scene_id / scene_id

    print(f"场景目录: {scene_dir}")

    # 获取所有时间戳文件夹
    timestamp_folders = get_sorted_timestamp_folders(scene_dir)

    if not timestamp_folders:
        print(f"跳过场景 {scene_id}：未找到时间戳文件夹")
        return False

    # 计算总帧数
    total_frames = args.frames_per_seg * args.num_segs
    print(f"需要帧数: {args.frames_per_seg} × {args.num_segs} = {total_frames}")

    # 选取中间帧
    selected_folders = select_middle_frames(timestamp_folders, total_frames)

    if len(selected_folders) < total_frames:
        total_frames = len(selected_folders)
        print(f"实际使用帧数: {total_frames}")

    # 处理每个相机
    for cam_name in CAMERA_NAMES:
        transfer_cam_name = CAMERA_NAME_MAPPING[cam_name]
        print(f"\n处理相机: {cam_name} → {transfer_cam_name}")

        # 检查GT和控制输入的第一帧是否存在
        first_gt_path = selected_folders[0] / 'gt' / f"{cam_name}.jpg"
        first_control_path = selected_folders[0] / args.control_subdir / f"{cam_name}.jpg"

        if not first_gt_path.exists():
            print(f"  跳过：未找到GT图像 {first_gt_path}")
            continue

        if not first_control_path.exists():
            print(f"  跳过：未找到控制输入图像 {first_control_path}")
            continue

        first_img = cv2.imread(str(first_gt_path))
        if first_img is None:
            print(f"  跳过：无法读取GT图像 {first_gt_path}")
            continue

        print(f"  输出分辨率: 1280×720 (统一分辨率)")

        # 生成每个seg
        for seg_idx in range(args.num_segs):
            seg_id = seg_idx + 1

            # 计算当前seg的帧范围
            start_frame = seg_idx * args.frames_per_seg
            end_frame = min(start_frame + args.frames_per_seg, total_frames)

            seg_folders = selected_folders[start_frame:end_frame]

            print(f"  生成 seg{seg_id:02d}: 帧 [{start_frame}, {end_frame})")

            # 收集GT图像路径
            gt_image_paths = []
            for folder in seg_folders:
                img_path = folder / 'gt' / f"{cam_name}.jpg"
                if img_path.exists():
                    gt_image_paths.append(img_path)
                else:
                    print(f"    警告: 缺失GT图像 {img_path}")

            # 收集控制输入图像路径
            control_image_paths = []
            for folder in seg_folders:
                img_path = folder / args.control_subdir / f"{cam_name}.jpg"
                if img_path.exists():
                    control_image_paths.append(img_path)
                else:
                    print(f"    警告: 缺失控制输入图像 {img_path}")

            if not gt_image_paths:
                print(f"    跳过 seg{seg_id:02d}：无有效GT图像")
                continue

            if not control_image_paths:
                print(f"    跳过 seg{seg_id:02d}：无有效控制输入图像")
                continue

            video_filename = f"{scene_id}_seg{seg_id:02d}.mp4"

            # 创建GT视频（videos/），统一分辨率1280×720
            video_output_path = Path(args.output_dir) / 'videos' / transfer_cam_name / video_filename
            create_video_from_images(gt_image_paths, video_output_path, args.fps)
            print(f"    ✓ GT视频: {video_output_path}")

            # 创建控制输入视频（control_input_xxx/），统一分辨率1280×720
            control_output_path = Path(args.output_dir) / f'control_input_{args.control_input_type}' / transfer_cam_name / video_filename
            create_video_from_images(control_image_paths, control_output_path, args.fps)
            print(f"    ✓ 控制视频: {control_output_path}")

            # 创建 caption JSON
            caption_output_path = Path(args.output_dir) / 'captions' / transfer_cam_name / f"{scene_id}_seg{seg_id:02d}.json"
            caption_output_path.parent.mkdir(parents=True, exist_ok=True)

            caption_data = create_caption_json(scene_id, seg_id, transfer_cam_name, args.caption_template)

            with open(caption_output_path, 'w', encoding='utf-8') as f:
                json.dump(caption_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 场景 {scene_id} 处理完成")
    return True


def main():
    """主函数"""
    args = parse_args()

    print("="*60)
    print("Transfer2 视频数据集生成器")
    print("="*60)
    print(f"项目类型: {args.project_type}")
    print(f"场景列表: {', '.join(args.scenes)}")
    print(f"每个seg帧数: {args.frames_per_seg}")
    print(f"seg数量: {args.num_segs}")
    print(f"总帧数/场景: {args.frames_per_seg * args.num_segs}")
    print(f"帧率: {args.fps} FPS")
    print(f"输出目录: {args.output_dir}")
    print(f"控制输入子目录: {args.control_subdir}")
    print(f"控制输入类型: {args.control_input_type}")

    # 创建输出目录
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 处理每个场景
    success_count = 0
    for scene_id in args.scenes:
        if process_single_scene(args, scene_id):
            success_count += 1

    print(f"\n{'='*60}")
    print(f"处理完成: {success_count}/{len(args.scenes)} 个场景成功")
    print(f"{'='*60}")

    # 显示输出目录结构
    print(f"\n输出目录结构:")
    print(f"  {args.output_dir}/")
    print(f"    ├── videos/                                 (GT去畸变视频)")
    for cam_name in CAMERA_NAMES[:2]:
        print(f"    │   ├── {CAMERA_NAME_MAPPING[cam_name]}/")
    print(f"    │   └── ... (全部7个相机)")
    print(f"    ├── control_input_{args.control_input_type}/  (控制输入视频)")
    for cam_name in CAMERA_NAMES[:2]:
        print(f"    │   ├── {CAMERA_NAME_MAPPING[cam_name]}/")
    print(f"    │   └── ... (全部7个相机)")
    print(f"    └── captions/                               (caption JSON)")
    for cam_name in CAMERA_NAMES[:2]:
        print(f"        ├── {CAMERA_NAME_MAPPING[cam_name]}/")
    print(f"        └── ... (全部7个相机)")


if __name__ == "__main__":
    main()
