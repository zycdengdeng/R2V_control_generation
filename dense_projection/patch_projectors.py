#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为所有投影模块添加 process_single_frame_with_data 方法
"""

import os
import re
from pathlib import Path

# 需要修改的模块
MODULES = [
    'blur投影',
    'blur稠密化投影',
    'depth投影',
    'depth稠密化投影',
    'HDMap投影',
]

# 要添加的方法
PATCH_METHOD = '''
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
        # 确保 colors 不是 None
        if colors is None:
            colors = np.ones((len(points), 3)) * 0.5

        # 调用原始方法，但传入已加载的点云数据
        return self._process_single_frame_internal(points, colors, output_dir, timestamp_ms, num_threads)
'''


def patch_projector_file(filepath):
    """为投影器文件添加 process_single_frame_with_data 方法"""
    with open(filepath, 'r') as f:
        content = f.read()

    # 检查是否已经有这个方法
    if 'process_single_frame_with_data' in content:
        print(f"  已存在: {filepath}")
        return

    # 找到 process_single_frame 方法
    pattern = r'(    def process_single_frame\(self, pcd_path.*?)(        return True\n)'
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        print(f"  找不到 process_single_frame 方法: {filepath}")
        return

    # 获取原始方法内容
    original_method = match.group(0)

    # 修改原始方法名为 _process_single_frame_internal
    # 并移除文件读取部分
    internal_method = original_method.replace(
        'def process_single_frame(self, pcd_path',
        'def _process_single_frame_internal(self, points, colors'
    )

    # 移除 pcd 读取代码
    internal_method = re.sub(
        r'        # \d+\. 加载点云.*?colors = np\..*?\n',
        '',
        internal_method,
        flags=re.DOTALL
    )

    # 创建新的 process_single_frame 方法（从文件读取）
    new_process_single_frame = '''    def process_single_frame(self, pcd_path, output_dir, timestamp_ms, num_threads=7):
        """
        处理单帧数据（从文件读取）
        """
        import open3d as o3d

        # 加载点云
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else np.ones((len(points), 3)) * 0.5

        return self._process_single_frame_internal(points, colors, output_dir, timestamp_ms, num_threads)

''' + PATCH_METHOD + '\n'

    # 替换原始方法
    new_content = content.replace(original_method, internal_method + '\n' + new_process_single_frame)

    with open(filepath, 'w') as f:
        f.write(new_content)

    print(f"  已修改: {filepath}")


def main():
    base_dir = Path(__file__).parent

    print("正在修改投影器文件...")

    for module in MODULES:
        projector_file = base_dir / module / 'undistort_projection_multithread_v2.py'
        if projector_file.exists():
            patch_projector_file(projector_file)
        else:
            print(f"  不存在: {projector_file}")

    print("\n完成!")


if __name__ == '__main__':
    main()
