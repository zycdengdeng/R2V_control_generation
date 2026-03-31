#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dense Projection Video Maker - 交互式视频生成

从投影输出结果生成视频，支持按方向和clip选择
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import os

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DIRECTION_CLIPS, DIRECTION_CHOICES, get_clip_id

# 相机名称
CAMERA_NAMES = ['FN', 'FW', 'FL', 'FR', 'RL', 'RR', 'RN']

# 投影类型配置
PROJECT_TYPES = {
    '1': {'name': '基本点云投影', 'dir': '基本点云投影', 'subdirs': ['proj', 'gt', 'compare', 'overlay']},
    '2': {'name': 'depth投影', 'dir': 'depth投影', 'subdirs': ['depth', 'gt', 'compare', 'overlay']},
    '3': {'name': 'depth稠密化投影', 'dir': 'depth稠密化投影', 'subdirs': ['depth', 'gt', 'compare', 'overlay']},
    '4': {'name': 'blur投影', 'dir': 'blur投影', 'subdirs': ['proj', 'gt', 'compare', 'overlay']},
    '5': {'name': 'blur稠密化投影', 'dir': 'blur稠密化投影', 'subdirs': ['proj', 'gt', 'compare', 'overlay']},
}


def get_sorted_timestamp_folders(scene_dir):
    """获取场景目录下所有时间戳文件夹，按时间戳排序"""
    scene_path = Path(scene_dir)

    if not scene_path.exists():
        return []

    timestamp_folders = []
    for folder in scene_path.iterdir():
        if folder.is_dir() and folder.name.isdigit():
            timestamp_folders.append(folder)

    timestamp_folders.sort(key=lambda x: int(x.name))
    return timestamp_folders


def create_video_from_images(image_paths, output_path, fps):
    """从图像列表创建视频"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not image_paths:
        return False

    first_img = cv2.imread(str(image_paths[0]))
    if first_img is None:
        return False

    resolution = (first_img.shape[1], first_img.shape[0])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, resolution)

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is not None:
            if img.shape[1] != resolution[0] or img.shape[0] != resolution[1]:
                img = cv2.resize(img, resolution, interpolation=cv2.INTER_LINEAR)
            video_writer.write(img)

    video_writer.release()
    return True


def process_clip(clip_name, project_type_info, fps, output_root, proj_root):
    """处理单个clip生成视频"""
    clip_id = get_clip_id(clip_name)
    project_dir = project_type_info['dir']
    subdirs = project_type_info['subdirs']

    # 投影输出目录
    clip_dir = Path(proj_root) / project_dir / clip_id

    print(f"\n{'='*60}")
    print(f"处理 Clip: {clip_id}")
    print(f"目录: {clip_dir}")
    print(f"{'='*60}")

    if not clip_dir.exists():
        print(f"  跳过: 目录不存在 {clip_dir}")
        return False

    # 获取时间戳文件夹
    timestamp_folders = get_sorted_timestamp_folders(clip_dir)
    if not timestamp_folders:
        print(f"  跳过: 未找到时间戳文件夹")
        return False

    print(f"  找到 {len(timestamp_folders)} 帧")

    # 输出目录
    output_dir = Path(output_root) / project_dir / clip_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # 处理每个子目录
    for subdir in subdirs:
        first_subdir = timestamp_folders[0] / subdir
        if not first_subdir.exists():
            print(f"  {subdir}: 子目录不存在，跳过")
            continue

        print(f"  处理 {subdir}...")

        # 处理每个相机
        for cam_name in CAMERA_NAMES:
            image_paths = []
            for folder in timestamp_folders:
                for ext in ['.jpg', '.png']:
                    img_path = folder / subdir / f"{cam_name}{ext}"
                    if img_path.exists():
                        image_paths.append(img_path)
                        break

            if not image_paths:
                continue

            video_filename = f"{clip_id}_{subdir}_{cam_name}.mp4"
            video_output = output_dir / subdir / video_filename

            if create_video_from_images(image_paths, video_output, fps):
                print(f"    {cam_name}: {len(image_paths)} 帧")

    return True


def interactive_input():
    """交互式输入"""
    print("\n" + "="*60)
    print("  Dense Projection Video Maker")
    print("="*60)

    # 1. 选择投影类型
    print("\n请选择投影类型:")
    for key, info in PROJECT_TYPES.items():
        print(f"  {key}) {info['name']}")

    project_choice = input("\n请选择 (1-5): ").strip()
    if project_choice not in PROJECT_TYPES:
        print("无效选择")
        return None

    project_type_info = PROJECT_TYPES[project_choice]
    print(f"  选择: {project_type_info['name']}")

    # 2. 选择方向
    print("\n请选择方向:")
    for key, direction in DIRECTION_CHOICES.items():
        info = DIRECTION_CLIPS[direction]
        print(f"  {key}) {info['name']} - {len(info['clips'])} 个clips")

    direction_choice = input("\n请选择 (a/b/c/d): ").strip().lower()
    if direction_choice not in DIRECTION_CHOICES:
        print("无效选择")
        return None

    direction = DIRECTION_CHOICES[direction_choice]
    clips = DIRECTION_CLIPS[direction]['clips']

    if not clips:
        print(f"  {direction} 方向没有clips")
        return None

    print(f"  选择: {DIRECTION_CLIPS[direction]['name']}")

    # 3. 选择clips
    print(f"\n可用clips ({len(clips)}个):")
    for i, clip in enumerate(clips, 1):
        clip_id = get_clip_id(clip)
        print(f"  {i}. {clip_id}")

    print("\n处理方式:")
    print("  1. 处理全部")
    print("  2. 处理范围 (如: 1-5)")
    print("  3. 处理指定 (如: 1,3,5)")

    mode = input("\n请选择: ").strip()

    selected_clips = []
    if mode == '1':
        selected_clips = clips
    elif mode.startswith('2') or '-' in mode:
        try:
            range_str = mode.replace('2', '').strip() or input("输入范围 (如 1-5): ").strip()
            start, end = map(int, range_str.split('-'))
            selected_clips = clips[start-1:end]
        except:
            print("无效范围")
            return None
    elif mode.startswith('3') or ',' in mode:
        try:
            indices_str = mode.replace('3', '').strip() or input("输入索引 (如 1,3,5): ").strip()
            indices = [int(x.strip()) for x in indices_str.split(',')]
            selected_clips = [clips[i-1] for i in indices]
        except:
            print("无效索引")
            return None
    else:
        try:
            idx = int(mode)
            selected_clips = [clips[idx-1]]
        except:
            print("无效选择")
            return None

    # 4. 投影输出根目录
    default_proj_root = "/mnt/zihanw/proj_utils_pro/dense_projection"
    proj_root_input = input(f"\n投影输出根目录 (默认 {default_proj_root}): ").strip()
    proj_root = proj_root_input if proj_root_input else default_proj_root

    # 5. 帧率
    fps_input = input("\n帧率 (默认10): ").strip()
    fps = int(fps_input) if fps_input else 10

    # 6. 视频输出目录
    default_output = f"/mnt/zihanw/dense_projection_videos"
    output_input = input(f"\n视频输出目录 (默认 {default_output}): ").strip()
    output_root = output_input if output_input else default_output

    return {
        'project_type_info': project_type_info,
        'clips': selected_clips,
        'proj_root': proj_root,
        'fps': fps,
        'output_root': output_root
    }


def main():
    config = interactive_input()
    if not config:
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  处理计划:")
    print(f"   投影类型: {config['project_type_info']['name']}")
    print(f"   Clip数量: {len(config['clips'])}")
    print(f"   投影输出目录: {config['proj_root']}")
    print(f"   帧率: {config['fps']} FPS")
    print(f"   视频输出目录: {config['output_root']}")
    print(f"{'='*60}")

    confirm = input("\n开始处理? (y/n): ").strip().lower()
    if confirm != 'y':
        print("取消")
        sys.exit(0)

    # 处理每个clip
    success_count = 0
    for clip_name in config['clips']:
        if process_clip(clip_name, config['project_type_info'], config['fps'], config['output_root'], config['proj_root']):
            success_count += 1

    print(f"\n{'='*60}")
    print(f"  完成! {success_count}/{len(config['clips'])} 个clips")
    print(f"  视频保存到: {config['output_root']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(1)
