#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dense Projection 配置文件
定义 clips 分组、静态地图路径等配置
"""

import os
from pathlib import Path

# ==================== 路径配置 ====================
# 动态点云根目录
DYNAMIC_PCD_ROOT = "/mnt/zongrx/PointCloudMerge_batch/runs"

# 静态地图根目录
STATIC_MAP_ROOT = "/mnt/zihanw/merge_dyn_stat/static_maps"

# Transform JSON 根目录
TRANSFORM_JSON_ROOT = "/mnt/zihanw/proj_utils_pro/transform_json"

# 合并后PCD输出目录
MERGED_PCD_OUTPUT = "/mnt/zihanw/merge_dyn_stat/merge_pcd"

# 原始数据集路径（用于车端GT图像和标定）
ORIGINAL_DATASET_ROOT = "/mnt/car_road_data_fix"
SUPPORT_INFO_DIR = os.path.join(ORIGINAL_DATASET_ROOT, "support_info")
VEHICLE_CALIB_DIR = os.path.join(SUPPORT_INFO_DIR, "NoEER705_v3", "camera")
ROADSIDE_CALIB_FILE = os.path.join(SUPPORT_INFO_DIR, "calib.json")
CARID_JSON_FILE = os.path.join(SUPPORT_INFO_DIR, "carid.json")


# ==================== Clips 分组配置 ====================
# 按方向分组的 clips
# 每个方向对应一个静态地图

DIRECTION_CLIPS = {
    'NS': {  # 南北向
        'name': '南北向 (NS)',
        'clips': [
            '051_car0402_road0402_t29-20251103-083512',
            '057_car0402_road0402_t35-20251103-083512',
            '059_car0402_road0402_t37-20251104-074154',
            '061_car0402_road0402_t39-20251104-205043',
            '065_car0402_road0402_t43-20251104-205043',
            '067_car0402_road0402_t45-20251104-205043',
            '069_car0402_road0402_t47-20251104-205043',
            '071_car0402_road0402_t49-20251107-180504',
            '073_car0402_road0402_t51-20251107-180504',
            '075_car0402_road0402_t53-20251107-180504',
            '077_car0402_road0402_t55-20251107-180504',
        ],
        'static_map': 'static_map_world.pcd',  # 静态地图文件名
    },
    'EW': {  # 东西向（预留）
        'name': '东西向 (EW)',
        'clips': [],
        'static_map': 'static_map_world.pcd',
    },
    'SN': {  # 北南向（预留）
        'name': '北南向 (SN)',
        'clips': [],
        'static_map': 'static_map_world.pcd',
    },
    'WE': {  # 西东向（预留）
        'name': '西东向 (WE)',
        'clips': [],
        'static_map': 'static_map_world.pcd',
    },
}

# 方向选择映射
DIRECTION_CHOICES = {
    'a': 'NS',
    'b': 'EW',
    'c': 'SN',
    'd': 'WE',
}


def get_clip_id(clip_name):
    """从 clip 名称提取 clip ID (如 051)"""
    return clip_name.split('_')[0]


def get_dynamic_pcd_dir(clip_name):
    """获取动态点云目录路径"""
    return os.path.join(DYNAMIC_PCD_ROOT, clip_name, 'dyn_2scene_fix')


def get_static_map_path(direction):
    """获取静态地图路径"""
    config = DIRECTION_CLIPS.get(direction)
    if not config:
        return None
    return os.path.join(STATIC_MAP_ROOT, direction, config['static_map'])


def get_transform_json_path(clip_name):
    """获取 transform JSON 路径"""
    clip_id = get_clip_id(clip_name)
    return os.path.join(TRANSFORM_JSON_ROOT, clip_id, 'world2lidar_transforms.json')


def get_original_scene_path(clip_name):
    """
    获取原始数据集中对应的场景路径
    用于读取车端GT图像

    clip_name 格式: 051_car0402_road0402_t29-20251103-083512
    对应场景ID: 051
    """
    clip_id = get_clip_id(clip_name)
    # 查找匹配的场景目录
    import glob
    pattern = os.path.join(ORIGINAL_DATASET_ROOT, f"{clip_id}_*")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    return None


def get_vehicle_images_path(clip_name):
    """获取车端GT图像路径"""
    scene_path = get_original_scene_path(clip_name)
    if scene_path:
        return os.path.join(scene_path, 'car', 'images')
    return None


def list_available_clips(direction):
    """列出某个方向下的所有可用 clips"""
    config = DIRECTION_CLIPS.get(direction)
    if not config:
        return []
    return config['clips']


def validate_direction(direction):
    """验证方向配置是否有效"""
    config = DIRECTION_CLIPS.get(direction)
    if not config:
        print(f"无效的方向: {direction}")
        return False

    if not config['clips']:
        print(f"方向 {direction} 暂无可用 clips")
        return False

    static_map_path = get_static_map_path(direction)
    if not os.path.exists(static_map_path):
        print(f"静态地图不存在: {static_map_path}")
        return False

    return True


def interactive_select_direction():
    """交互式选择方向"""
    print("\n" + "=" * 60)
    print("请选择处理方向:")
    print("=" * 60)

    for key, direction in DIRECTION_CHOICES.items():
        config = DIRECTION_CLIPS[direction]
        clip_count = len(config['clips'])
        status = f"({clip_count} clips)" if clip_count > 0 else "(暂无数据)"
        print(f"  {key}) {config['name']} {status}")

    print()
    choice = input("请输入选项 [a/b/c/d]: ").strip().lower()

    if choice not in DIRECTION_CHOICES:
        print(f"无效选项: {choice}")
        return None

    direction = DIRECTION_CHOICES[choice]
    config = DIRECTION_CLIPS[direction]

    if not config['clips']:
        print(f"\n{config['name']} 暂无可用 clips，请选择其他方向")
        return None

    print(f"\n已选择: {config['name']}")
    print(f"包含 {len(config['clips'])} 个 clips:")
    for clip in config['clips']:
        print(f"  - {clip}")

    return direction


if __name__ == "__main__":
    # 测试配置
    print("=" * 60)
    print("Dense Projection 配置测试")
    print("=" * 60)

    direction = interactive_select_direction()
    if direction:
        print(f"\n静态地图路径: {get_static_map_path(direction)}")

        clips = list_available_clips(direction)
        if clips:
            clip = clips[0]
            print(f"\n示例 clip: {clip}")
            print(f"  动态PCD目录: {get_dynamic_pcd_dir(clip)}")
            print(f"  Transform JSON: {get_transform_json_path(clip)}")
            print(f"  原始场景路径: {get_original_scene_path(clip)}")
            print(f"  车端图像路径: {get_vehicle_images_path(clip)}")
