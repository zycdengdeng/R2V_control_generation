#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动静点云合并工具
用于将动态点云和静态地图合并
"""

import os
import numpy as np
from collections import defaultdict
from pathlib import Path


def read_static_map_rgb_only(filepath):
    """
    读取静态地图，只保留xyz和rgb

    静态地图格式: x y z h s v rgb (7字段, ascii)
    输出格式: x y z rgb (4字段)
    """
    print(f"  读取静态地图: {filepath}")
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('DATA'):
                break

        points = []
        for line in f:
            vals = line.strip().split()
            if len(vals) >= 7:
                x, y, z = float(vals[0]), float(vals[1]), float(vals[2])
                rgb = int(vals[6])  # rgb是第7个字段
                # 将uint32 rgb转换为float32的bit pattern（pcd标准格式）
                rgb_float = np.array([rgb], dtype=np.uint32).view(np.float32)[0]
                points.append([x, y, z, rgb_float])

        arr = np.array(points, dtype=np.float32)
        print(f"    点数: {len(arr)}")
        return arr


def read_dynamic_pcd(filepath):
    """
    读取动态点云 (binary, 4字段: x y z rgb)
    """
    with open(filepath, 'rb') as f:
        num_points = 0
        while True:
            line = f.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith('POINTS'):
                num_points = int(line.split()[1])
            if line.startswith('DATA'):
                break

        dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<f4')])
        data = np.frombuffer(f.read(), dtype=dt, count=num_points)
        return np.column_stack([data['x'], data['y'], data['z'], data['rgb']])


def group_dynamic_pcd_by_timestamp(dyn_dir):
    """
    按时间戳分组动态点云文件

    文件名格式: {track_id}_{timestamp}_2scene.pcd

    Returns:
        dict: {timestamp: [file_paths]}
    """
    pcd_files = [f for f in os.listdir(dyn_dir) if f.endswith('.pcd')]
    timestamp_groups = defaultdict(list)

    for f in pcd_files:
        parts = f.split('_')
        if len(parts) >= 2:
            timestamp = parts[1]
            timestamp_groups[timestamp].append(os.path.join(dyn_dir, f))

    return timestamp_groups


def merge_pcd_for_timestamp(static_pts, dynamic_paths):
    """
    合并单个时间戳的点云

    Args:
        static_pts: 静态地图点云 (N, 4) - x y z rgb
        dynamic_paths: 动态点云文件路径列表

    Returns:
        merged_pts: 合并后的点云 (M, 4)
    """
    all_dyn = []
    for path in dynamic_paths:
        pts = read_dynamic_pcd(path)
        all_dyn.append(pts)

    if all_dyn:
        dyn_pts = np.vstack(all_dyn)
        merged = np.vstack([static_pts, dyn_pts])
    else:
        merged = static_pts

    return merged


def get_merged_pcd_as_o3d(static_pts, dynamic_paths):
    """
    获取合并后的点云（Open3D格式）

    用于直接传递给投影模块

    Returns:
        points: (N, 3) xyz坐标
        colors: (N, 3) RGB颜色 [0, 1]
    """
    import open3d as o3d

    merged = merge_pcd_for_timestamp(static_pts, dynamic_paths)

    # 提取xyz
    points = merged[:, :3]

    # 提取rgb并转换为颜色
    # rgb是packed float32，需要转换回uint32再解包
    rgb_packed = merged[:, 3].view(np.uint32)

    # 解包RGB
    r = ((rgb_packed >> 16) & 0xFF) / 255.0
    g = ((rgb_packed >> 8) & 0xFF) / 255.0
    b = (rgb_packed & 0xFF) / 255.0

    colors = np.column_stack([r, g, b])

    return points, colors


class MergedPCDLoader:
    """
    合并点云加载器

    缓存静态地图，按需加载动态点云并合并
    """

    def __init__(self, static_map_path, dynamic_pcd_dir):
        """
        初始化加载器

        Args:
            static_map_path: 静态地图路径
            dynamic_pcd_dir: 动态点云目录
        """
        self.static_map_path = static_map_path
        self.dynamic_pcd_dir = dynamic_pcd_dir

        # 加载静态地图（只加载一次）
        print("加载静态地图...")
        self.static_pts = read_static_map_rgb_only(static_map_path)

        # 按时间戳分组动态点云
        print("索引动态点云...")
        self.timestamp_groups = group_dynamic_pcd_by_timestamp(dynamic_pcd_dir)

        print(f"  找到 {len(self.timestamp_groups)} 个时间戳")

    def get_timestamps(self):
        """获取所有可用的时间戳（已排序）"""
        return sorted(self.timestamp_groups.keys())

    def get_merged_pcd(self, timestamp):
        """
        获取指定时间戳的合并点云

        Args:
            timestamp: 时间戳字符串

        Returns:
            points: (N, 3) xyz坐标
            colors: (N, 3) RGB颜色 [0, 1]
        """
        if timestamp not in self.timestamp_groups:
            raise ValueError(f"时间戳 {timestamp} 不存在")

        dynamic_paths = self.timestamp_groups[timestamp]
        return get_merged_pcd_as_o3d(self.static_pts, dynamic_paths)

    def get_track_count(self, timestamp):
        """获取指定时间戳的track数量"""
        if timestamp not in self.timestamp_groups:
            return 0
        return len(self.timestamp_groups[timestamp])


def write_merged_pcd(filepath, points, colors=None):
    """
    写入合并后的点云 (binary格式)

    Args:
        filepath: 输出路径
        points: (N, 3) xyz坐标
        colors: (N, 3) RGB颜色 [0, 1]，可选
    """
    num_points = len(points)

    if colors is not None:
        # 将RGB打包为float32
        r = (colors[:, 0] * 255).astype(np.uint32)
        g = (colors[:, 1] * 255).astype(np.uint32)
        b = (colors[:, 2] * 255).astype(np.uint32)
        rgb_packed = (r << 16) | (g << 8) | b
        rgb_float = rgb_packed.view(np.float32)

        data = np.column_stack([points, rgb_float]).astype(np.float32)
        fields = "x y z rgb"
        field_count = 4
    else:
        data = points.astype(np.float32)
        fields = "x y z"
        field_count = 3

    data = np.ascontiguousarray(data)

    with open(filepath, 'wb') as f:
        header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS {fields}
SIZE {' '.join(['4'] * field_count)}
TYPE {' '.join(['F'] * field_count)}
COUNT {' '.join(['1'] * field_count)}
WIDTH {num_points}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {num_points}
DATA binary
"""
        f.write(header.encode('utf-8'))
        data.tofile(f)


if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from config import get_static_map_path, get_dynamic_pcd_dir, list_available_clips

    print("=" * 60)
    print("测试动静合并")
    print("=" * 60)

    # 测试NS方向的第一个clip
    direction = 'NS'
    clips = list_available_clips(direction)

    if clips:
        clip = clips[0]
        static_map_path = get_static_map_path(direction)
        dynamic_pcd_dir = get_dynamic_pcd_dir(clip)

        print(f"\nClip: {clip}")
        print(f"静态地图: {static_map_path}")
        print(f"动态PCD目录: {dynamic_pcd_dir}")

        if os.path.exists(static_map_path) and os.path.exists(dynamic_pcd_dir):
            loader = MergedPCDLoader(static_map_path, dynamic_pcd_dir)

            timestamps = loader.get_timestamps()
            print(f"\n可用时间戳数: {len(timestamps)}")

            if timestamps:
                ts = timestamps[0]
                points, colors = loader.get_merged_pcd(ts)
                print(f"\n时间戳 {ts}:")
                print(f"  点数: {len(points)}")
                print(f"  Track数: {loader.get_track_count(ts)}")
                print(f"  X范围: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
        else:
            print("\n路径不存在，跳过测试")
    else:
        print("无可用clips")
