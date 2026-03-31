#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dense Projection 统一工具模块
基于原 proj_utils_pro/common_utils.py 修改
支持动静结合的点云投影
"""

import os
import json
import glob
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# 导入配置
from config import (
    DIRECTION_CLIPS, DIRECTION_CHOICES,
    ORIGINAL_DATASET_ROOT, SUPPORT_INFO_DIR,
    VEHICLE_CALIB_DIR, ROADSIDE_CALIB_FILE, CARID_JSON_FILE,
    TRANSFORM_JSON_ROOT,
    get_clip_id, get_dynamic_pcd_dir, get_static_map_path,
    get_transform_json_path, get_original_scene_path, get_vehicle_images_path,
    list_available_clips, interactive_select_direction
)


# ==================== 固定路径配置 ====================
# 使用 config.py 中的配置
DATASET_ROOT = ORIGINAL_DATASET_ROOT


# ==================== 场景路径管理 ====================
def find_scene_path(scene_id: str) -> Optional[str]:
    """
    根据场景ID查找完整场景路径

    Args:
        scene_id: 场景ID（如 "051"）

    Returns:
        完整场景路径，如果找不到返回 None
    """
    pattern = os.path.join(DATASET_ROOT, f"{scene_id}_*")
    matches = glob.glob(pattern)

    if not matches:
        print(f"  未找到场景 {scene_id}")
        return None

    if len(matches) > 1:
        print(f"  场景 {scene_id} 找到多个匹配目录:")
        for m in matches:
            print(f"   - {m}")
        print(f"使用第一个: {matches[0]}")

    return matches[0]


def get_scene_paths(clip_name: str) -> Dict[str, str]:
    """
    获取 clip 的所有相关路径

    Args:
        clip_name: clip 名称（如 "051_car0402_road0402_t29-20251103-083512"）

    Returns:
        包含各类路径的字典
    """
    clip_id = get_clip_id(clip_name)
    scene_root = find_scene_path(clip_id)

    if not scene_root:
        return {}

    paths = {
        'root': scene_root,
        'clip_name': clip_name,
        'clip_id': clip_id,
        'scene_name': os.path.basename(scene_root),
        'dynamic_pcd': get_dynamic_pcd_dir(clip_name),
        'transform_json': get_transform_json_path(clip_name),
        'roadside_images': os.path.join(scene_root, 'road', 'cameras'),
        'roadside_labels': os.path.join(scene_root, 'road_labels', 'interpolation_labels'),
        'vehicle_images': os.path.join(scene_root, 'car', 'images'),
        'vehicle_calib': VEHICLE_CALIB_DIR,
        'roadside_calib': ROADSIDE_CALIB_FILE
    }

    return paths


def validate_clip_paths(paths: Dict[str, str], direction: str) -> bool:
    """
    验证 clip 路径是否存在

    Args:
        paths: get_scene_paths() 返回的路径字典
        direction: 方向（用于获取静态地图）

    Returns:
        是否所有必需路径都存在
    """
    required = {
        'dynamic_pcd': paths.get('dynamic_pcd'),
        'transform_json': paths.get('transform_json'),
        'vehicle_calib': paths.get('vehicle_calib'),
        'roadside_calib': paths.get('roadside_calib'),
    }

    # 检查静态地图
    static_map = get_static_map_path(direction)

    missing = []

    for key, path in required.items():
        if not path or not os.path.exists(path):
            missing.append(f"{key}: {path}")

    if not static_map or not os.path.exists(static_map):
        missing.append(f"static_map: {static_map}")

    if missing:
        print(f"  缺失路径:")
        for m in missing:
            print(f"   - {m}")
        return False

    return True


# ==================== 变换矩阵管理 ====================
def load_world2lidar_transforms(json_path: str, show_range: bool = False) -> List[Dict]:
    """
    加载 world2lidar 变换矩阵

    Args:
        json_path: 变换JSON文件路径
        show_range: 是否显示时间戳范围

    Returns:
        变换矩阵列表
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"变换JSON文件不存在: {json_path}")

    with open(json_path, 'r') as f:
        transforms = json.load(f)

    # 时间戳单位转换
    if transforms and 'timestamp' in transforms[0]:
        first_ts = transforms[0]['timestamp']

        if first_ts < 1e12:
            print(f"   检测到时间戳为秒格式，转换为毫秒...")
            for t in transforms:
                t['timestamp'] = t['timestamp'] * 1000
        else:
            print(f"   检测到时间戳为毫秒格式")

    if show_range and transforms:
        timestamps = [t['timestamp'] for t in transforms]
        print(f"  加载了 {len(transforms)} 个world2lidar变换矩阵")
        print(f"   时间戳范围: {min(timestamps):.0f} ~ {max(timestamps):.0f} (毫秒)")
        print(f"   时间跨度: {(max(timestamps) - min(timestamps)) / 1000:.1f} 秒")
    else:
        print(f"  加载了 {len(transforms)} 个world2lidar变换矩阵")

    return transforms


def find_closest_transform(timestamp_ms: float, transforms: List[Dict],
                          tolerance_ms: float = 5000.0, verbose: bool = False) -> Optional[Dict]:
    """
    根据时间戳查找最接近的变换矩阵
    """
    min_diff = float('inf')
    closest = None
    closest_ts = None

    for t in transforms:
        diff = abs(t['timestamp'] - timestamp_ms)
        if diff < min_diff:
            min_diff = diff
            closest = t
            closest_ts = t['timestamp']

    if verbose:
        print(f"   时间戳匹配: 目标={timestamp_ms}, 最近={closest_ts}, 差异={min_diff:.1f}ms")

    if min_diff > tolerance_ms:
        if verbose:
            print(f"   差异 {min_diff:.1f}ms 超出容差 {tolerance_ms:.1f}ms")
        return None

    return closest


def transform_points_to_lidar(points_world: np.ndarray, transform: Dict) -> np.ndarray:
    """
    将世界坐标系点云变换到LiDAR坐标系
    """
    import cv2

    rotation = np.array(transform['world2lidar']['rotation'])
    translation = np.array(transform['world2lidar']['translation'])

    R, _ = cv2.Rodrigues(rotation)
    points_lidar = (R @ points_world.T).T + translation

    return points_lidar


# ==================== 批次选择逻辑 ====================
def get_batch_files(file_list: List[str], batch_mode: str) -> List[str]:
    """
    根据批次模式选择文件
    """
    total = len(file_list)

    if batch_mode == "all":
        return file_list

    if batch_mode.isdigit():
        n = int(batch_mode)
        return file_list[:min(n, total)]

    if batch_mode == "middle_90":
        if total < 90:
            return file_list
        start = (total - 90) // 2
        return file_list[start:start+90]

    if batch_mode.startswith("middle_"):
        try:
            n = int(batch_mode.split("_")[1])
            if total < n:
                return file_list
            start = (total - n) // 2
            return file_list[start:start+n]
        except (IndexError, ValueError):
            return file_list

    if batch_mode.startswith("range_"):
        try:
            parts = batch_mode.split("_")
            q = int(parts[1])
            p = int(parts[2])

            start_idx = q - 1
            end_idx = min(start_idx + p, total)

            if start_idx >= total:
                return []

            return file_list[start_idx:end_idx]
        except (IndexError, ValueError):
            return file_list

    return file_list


def print_batch_info(selected_files: List[str], batch_mode: str, original_total: int = None):
    """打印批次选择信息"""
    if original_total is None:
        original_total = len(selected_files)

    selected_count = len(selected_files)
    print(f"\n  批次选择: {batch_mode}")
    print(f"   原始文件数: {original_total}")
    print(f"   选择文件数: {selected_count}")


# ==================== 输出路径管理 ====================
def get_unified_output_paths(output_root: str, clip_name: str, project_type: str) -> Dict[str, str]:
    """
    生成统一的输出路径结构
    """
    clip_id = get_clip_id(clip_name)
    clip_output = os.path.join(output_root, clip_id)

    paths = {
        'root': clip_output,
        'proj': os.path.join(clip_output, 'proj'),
        'gt': os.path.join(clip_output, 'gt'),
        'compare': os.path.join(clip_output, 'compare'),
        'overlay': os.path.join(clip_output, 'overlay'),
    }

    if project_type in ['depth', 'depth_dense']:
        paths['depth'] = os.path.join(clip_output, 'depth')

    if project_type in ['blur_dense']:
        paths['projections'] = os.path.join(clip_output, 'projections')

    if project_type in ['hdmap', 'json']:
        paths['annotations'] = os.path.join(clip_output, 'annotations')
        paths['videos'] = os.path.join(clip_output, 'videos')

    return paths


def create_output_dirs(paths: Dict[str, str]):
    """创建输出目录"""
    for key, path in paths.items():
        if key == 'root':
            continue
        os.makedirs(path, exist_ok=True)


# ==================== 批量模式配置管理 ====================
TEMP_DIR = Path(__file__).resolve().parent / "temp"
BATCH_CONFIG_FILE = TEMP_DIR / "dense_projection_batch_config.json"


def save_batch_config(config: Dict):
    """保存批量模式配置"""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    with open(BATCH_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

    if BATCH_CONFIG_FILE.exists():
        print(f"  批量配置已保存: {BATCH_CONFIG_FILE}")


def load_batch_config() -> Optional[Dict]:
    """加载批量模式配置"""
    if BATCH_CONFIG_FILE.exists():
        try:
            with open(BATCH_CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"  配置文件读取失败: {e}")
    return None


def clear_batch_config():
    """清除批量模式配置文件"""
    if BATCH_CONFIG_FILE.exists():
        BATCH_CONFIG_FILE.unlink()
        print(f"  已清除批量配置: {BATCH_CONFIG_FILE}")


# ==================== 交互式输入 ====================
def interactive_input(batch_mode_enabled: bool = False) -> Optional[Dict]:
    """
    统一的交互式输入流程

    Returns:
        配置字典，包含 direction, clips, batch_mode 等
    """
    # 批量模式：从配置文件加载
    if batch_mode_enabled:
        print(f"\n{'='*60}")
        print("  批量模式已启用")
        print(f"{'='*60}")

        config = load_batch_config()
        if config:
            print("\n  使用已保存的配置:")
            print(f"   方向: {config.get('direction')}")
            print(f"   Clips: {len(config.get('clips', []))} 个")
            print(f"   批次: {config.get('batch_mode')}")
            return config
        else:
            print("\n  未找到批量配置文件，切换到交互式输入\n")

    print("\n" + "="*60)
    print("  Dense Projection - 动静结合点云投影系统")
    print("="*60)

    # 步骤1: 选择方向
    direction = interactive_select_direction()
    if not direction:
        return None

    # 获取该方向的所有clips
    available_clips = list_available_clips(direction)

    # 步骤2: 选择要处理的clips
    print(f"\n  步骤 2/3: 选择要处理的 Clips")
    print("   选项:")
    print("     - all     : 处理所有 clips")
    print("     - 1,3,5   : 处理指定索引的 clips (逗号分隔)")
    print("     - 1-5     : 处理索引范围内的 clips")

    print("\n   可用 clips:")
    for i, clip in enumerate(available_clips):
        clip_id = get_clip_id(clip)
        print(f"     {i+1}. [{clip_id}] {clip}")

    clip_selection = input("\n   请输入选择 [all]: ").strip() or "all"

    if clip_selection == "all":
        selected_clips = available_clips
    elif '-' in clip_selection:
        # 范围选择
        try:
            start, end = map(int, clip_selection.split('-'))
            selected_clips = available_clips[start-1:end]
        except:
            print("   无效的范围格式")
            return None
    elif ',' in clip_selection:
        # 索引选择
        try:
            indices = [int(x.strip()) for x in clip_selection.split(',')]
            selected_clips = [available_clips[i-1] for i in indices if 0 < i <= len(available_clips)]
        except:
            print("   无效的索引格式")
            return None
    else:
        # 单个索引
        try:
            idx = int(clip_selection)
            selected_clips = [available_clips[idx-1]]
        except:
            print("   无效的选择")
            return None

    print(f"\n   已选择 {len(selected_clips)} 个 clips:")
    for clip in selected_clips:
        print(f"     - {clip}")

    # 步骤3: 选择批次模式
    print("\n  步骤 3/3: 选择批次模式")
    print("   选项:")
    print("     - all          : 处理所有帧（默认）")
    print("     - N            : 处理前N帧（例如：10）")
    print("     - middle_90    : 处理中间90帧")
    print("     - middle_N     : 处理中间N帧（例如：middle_50）")
    print("     - range_Q_P    : 从第Q帧开始处理P帧（例如：range_10_50）")

    batch_mode = input("   请输入批次模式 [all]: ").strip() or "all"

    print(f"\n   批次模式: {batch_mode}")

    # 构建配置
    config = {
        'direction': direction,
        'clips': selected_clips,
        'batch_mode': batch_mode,
        'static_map': get_static_map_path(direction),
    }

    print("\n" + "="*60)
    print("  配置完成")
    print("="*60 + "\n")

    save_batch_config(config)

    return config


def get_parallel_config(batch_mode_enabled: bool = False) -> Dict:
    """
    获取并行配置
    """
    import multiprocessing as mp

    if batch_mode_enabled:
        config = load_batch_config()
        if config and 'num_processes' in config:
            return {
                'num_processes': config['num_processes'],
                'threads_per_frame': config.get('threads_per_frame', 7)
            }

    print("\n  多核并行设置:")
    total_cores = mp.cpu_count()
    print(f"   检测到 {total_cores} 个CPU核心")

    if total_cores >= 128:
        default_processes = 16
    elif total_cores >= 64:
        default_processes = 8
    elif total_cores >= 32:
        default_processes = 4
    else:
        default_processes = 2

    default_threads = 7

    print(f"   推荐配置: {default_processes}进程 x {default_threads}线程")

    num_processes = input(f"   并行进程数 [默认{default_processes}]: ").strip()
    num_processes = int(num_processes) if num_processes else default_processes

    threads_per_frame = input(f"   每帧线程数 [默认{default_threads}]: ").strip()
    threads_per_frame = int(threads_per_frame) if threads_per_frame else default_threads

    print(f"\n   配置: {num_processes}进程 x {threads_per_frame}线程")

    # 更新配置
    config = load_batch_config()
    if config:
        config['num_processes'] = num_processes
        config['threads_per_frame'] = threads_per_frame
        save_batch_config(config)

    return {
        'num_processes': num_processes,
        'threads_per_frame': threads_per_frame
    }


def get_ego_vehicle_id(clip_names: List[str], batch_mode_enabled: bool = False,
                       default_id: int = 45) -> Dict[str, int]:
    """
    获取自车ID配置
    """
    if batch_mode_enabled:
        config = load_batch_config()
        if config and 'ego_vehicle_mapping' in config:
            return config['ego_vehicle_mapping']

    print(f"\n  自车ID配置:")
    print(f"   选项:")
    print(f"     - auto    : 自动从 carid.json 加载")
    print(f"     - <数字>  : 所有 clips 使用相同ID")

    mode = input(f"   请选择模式 [auto]: ").strip() or "auto"

    ego_mapping = {}

    if mode == "auto":
        carid_mapping = load_carid_mapping()
        for clip_name in clip_names:
            clip_id = get_clip_id(clip_name)
            if clip_id in carid_mapping:
                ego_mapping[clip_name] = carid_mapping[clip_id]
            else:
                ego_mapping[clip_name] = default_id
    else:
        try:
            ego_id = int(mode)
            for clip_name in clip_names:
                ego_mapping[clip_name] = ego_id
        except ValueError:
            for clip_name in clip_names:
                ego_mapping[clip_name] = default_id

    # 保存配置
    config = load_batch_config()
    if config:
        config['ego_vehicle_mapping'] = ego_mapping
        save_batch_config(config)

    return ego_mapping


def load_carid_mapping(carid_json_path: Optional[str] = None) -> Dict[str, int]:
    """
    从 carid.json 加载场景ID到自车ID映射
    """
    if carid_json_path is None:
        carid_json_path = CARID_JSON_FILE

    if not os.path.exists(carid_json_path):
        return {}

    try:
        with open(carid_json_path, 'r') as f:
            data = json.load(f)

        mapping = {}
        for item in data.get('results', []):
            clip_name = item.get('clip_name', '')
            nearest_carid = item.get('nearest_carid')

            if clip_name and nearest_carid is not None:
                scene_id = clip_name.split('_')[0]
                mapping[scene_id] = nearest_carid

        return mapping

    except Exception as e:
        print(f"  加载carid.json失败: {e}")
        return {}


# ==================== 工具函数 ====================
def extract_timestamp_from_filename(filename: str) -> Optional[float]:
    """
    从文件名提取时间戳

    支持格式：
        - {track_id}_{timestamp}_2scene.pcd
        - {timestamp}.pcd
    """
    import re
    basename = os.path.basename(filename)

    # 尝试匹配 {track_id}_{timestamp}_2scene.pcd
    match = re.search(r'_(\d+)_2scene\.pcd$', basename)
    if match:
        return float(match.group(1))

    # 尝试匹配一般格式
    match = re.search(r'(\d+)(?:\.\d+)?\.', basename)
    if match:
        return float(match.group(1))

    return None


def sort_files_by_timestamp(files: List[str]) -> List[str]:
    """按时间戳排序文件列表"""
    files_with_ts = []
    for f in files:
        ts = extract_timestamp_from_filename(f)
        if ts is not None:
            files_with_ts.append((ts, f))

    files_with_ts.sort(key=lambda x: x[0])
    return [f for _, f in files_with_ts]


if __name__ == "__main__":
    print("="*60)
    print("测试 Dense Projection common_utils.py")
    print("="*60)

    config = interactive_input()
    if config:
        print("\n配置内容:")
        for key, value in config.items():
            if isinstance(value, list):
                print(f"  {key}: {len(value)} 项")
            else:
                print(f"  {key}: {value}")
