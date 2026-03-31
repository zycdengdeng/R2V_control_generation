#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dense Projection - Blur稠密化投影批量处理
使用路侧相机着色+稠密化，支持动静结合的点云投影
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common_utils
from config import get_clip_id, get_static_map_path
from merge_utils import MergedPCDLoader

PROJECTOR_SCRIPT = Path(__file__).resolve().parent / "undistort_projection_multithread_v2.py"


def run_single_projection(args):
    """运行单个投影任务"""
    points, colors, timestamp_ms, output_dir, roadside_calib, roadside_images, \
    vehicle_calib, gt_images_folder, transform_json, threads_per_frame = args

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("blur_projector_v2", PROJECTOR_SCRIPT)
        projector_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(projector_module)

        if not hasattr(run_single_projection, 'transforms_cache'):
            run_single_projection.transforms_cache = {}

        if transform_json not in run_single_projection.transforms_cache:
            run_single_projection.transforms_cache[transform_json] = \
                common_utils.load_world2lidar_transforms(transform_json, show_range=False)

        transforms = run_single_projection.transforms_cache[transform_json]

        projector = projector_module.BlurDenseProjectorMultiThread(
            roadside_calib, roadside_images, vehicle_calib, gt_images_folder, transforms
        )

        success = projector.process_single_frame_with_data(
            points, colors, output_dir, timestamp_ms, threads_per_frame
        )

        return success, "成功" if success else "处理失败", timestamp_ms

    except Exception as e:
        import traceback
        return False, str(e)[:200], timestamp_ms


def process_single_clip(clip_name, config, num_processes, threads_per_frame, project_root):
    """处理单个clip"""
    clip_id = get_clip_id(clip_name)
    direction = config['direction']

    print(f"\n{'='*80}")
    print(f"开始处理 Clip: {clip_name}")
    print(f"{'='*80}")

    scene_paths = common_utils.get_scene_paths(clip_name)
    if not scene_paths:
        print(f"  场景路径获取失败，跳过")
        return

    if not common_utils.validate_clip_paths(scene_paths, direction):
        print(f"  路径验证失败，跳过")
        return

    static_map_path = get_static_map_path(direction)
    dynamic_pcd_dir = scene_paths['dynamic_pcd']
    transform_json = scene_paths['transform_json']

    print(f"\n  路径信息:")
    print(f"   静态地图: {static_map_path}")
    print(f"   动态PCD: {dynamic_pcd_dir}")

    print(f"\n  初始化点云加载器...")
    try:
        loader = MergedPCDLoader(static_map_path, dynamic_pcd_dir)
    except Exception as e:
        print(f"  点云加载器初始化失败: {e}")
        return

    timestamps = loader.get_timestamps()
    if not timestamps:
        print(f"  没有找到可用的时间戳")
        return

    selected_timestamps = common_utils.get_batch_files(timestamps, config['batch_mode'])
    common_utils.print_batch_info(selected_timestamps, config['batch_mode'], len(timestamps))

    if not selected_timestamps:
        return

    transforms = common_utils.load_world2lidar_transforms(transform_json, show_range=True)

    output_root = Path(project_root) / clip_id
    output_paths = common_utils.get_unified_output_paths(str(output_root), clip_name, 'blur_dense')
    common_utils.create_output_dirs(output_paths)

    print(f"\n  准备任务...")
    tasks = []
    output_root_path = Path(output_paths['root'])

    for ts in tqdm(selected_timestamps, desc="准备数据", unit="帧"):
        try:
            points, colors = loader.get_merged_pcd(ts)
            timestamp_ms = int(ts)
            output_frame_dir = output_root_path / str(timestamp_ms)

            tasks.append((
                points, colors, timestamp_ms, str(output_frame_dir),
                scene_paths['roadside_calib'],
                scene_paths['roadside_images'],
                scene_paths['vehicle_calib'],
                scene_paths.get('vehicle_images', scene_paths['roadside_images']),
                transform_json, threads_per_frame
            ))
        except Exception as e:
            print(f"   跳过时间戳 {ts}: {e}")

    if not tasks:
        return

    print(f"\n  开始处理 ({num_processes}进程 x {threads_per_frame}线程)...")
    success_count = 0
    failed_list = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {executor.submit(run_single_projection, task): task for task in tasks}

        with tqdm(total=len(tasks), desc=f"Clip {clip_id}", unit="帧") as pbar:
            for future in as_completed(futures):
                task = futures[future]
                timestamp_ms = task[2]

                try:
                    success, message, _ = future.result()
                    if success:
                        success_count += 1
                    else:
                        failed_list.append((timestamp_ms, message))
                except Exception as e:
                    failed_list.append((timestamp_ms, str(e)))

                pbar.update(1)

    elapsed_time = time.time() - start_time
    print(f"\n成功: {success_count}/{len(tasks)}, 耗时: {elapsed_time/60:.1f} 分钟")


def main():
    print("\n" + "="*80)
    print("  Dense Projection - Blur稠密化投影 (路侧着色+稠密化)")
    print("="*80)

    if not PROJECTOR_SCRIPT.exists():
        print(f"\n  找不到核心投影脚本: {PROJECTOR_SCRIPT}")
        sys.exit(1)

    batch_mode = os.environ.get('PROJECTION_BATCH_MODE', 'false') == 'true'
    config = common_utils.interactive_input(batch_mode_enabled=batch_mode)
    if not config:
        sys.exit(1)

    parallel_config = common_utils.get_parallel_config(batch_mode_enabled=batch_mode)
    output_root = Path(__file__).resolve().parent

    confirm = input("\n开始处理? (y/n): ").strip().lower()
    if confirm != 'y':
        sys.exit(0)

    for clip_name in config['clips']:
        process_single_clip(
            clip_name, config,
            parallel_config['num_processes'],
            parallel_config['threads_per_frame'],
            output_root
        )


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
