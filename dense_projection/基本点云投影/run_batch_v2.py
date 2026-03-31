#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dense Projection - 基本点云投影批量处理
支持动静结合的点云投影
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common_utils
from config import get_clip_id, get_transform_json_path, get_static_map_path
from merge_utils import MergedPCDLoader

# 核心投影脚本路径
PROJECTOR_SCRIPT = Path(__file__).resolve().parent / "undistort_projection_multithread_v2.py"


def run_single_projection(args):
    """运行单个投影任务"""
    points, colors, timestamp_ms, output_dir, roadside_calib, vehicle_calib, \
    gt_images_folder, transform_json, threads_per_frame = args

    try:
        # 动态导入核心模块
        import importlib.util
        spec = importlib.util.spec_from_file_location("projector_v2", PROJECTOR_SCRIPT)
        projector_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(projector_module)

        # 加载变换矩阵
        if not hasattr(run_single_projection, 'transforms_cache'):
            run_single_projection.transforms_cache = {}

        if transform_json not in run_single_projection.transforms_cache:
            run_single_projection.transforms_cache[transform_json] = \
                common_utils.load_world2lidar_transforms(transform_json, show_range=False)

        transforms = run_single_projection.transforms_cache[transform_json]

        # 创建投影器
        projector = projector_module.UndistortProjectorMultiThread(
            roadside_calib, vehicle_calib, gt_images_folder, transforms
        )

        # 处理单帧（传入已合并的点云数据）
        success = projector.process_single_frame_with_data(
            points, colors, output_dir, timestamp_ms, threads_per_frame
        )

        return success, "成功" if success else "处理失败", timestamp_ms

    except Exception as e:
        import traceback
        error_msg = str(e) + "\n" + traceback.format_exc()
        return False, error_msg[:200], timestamp_ms


def process_single_clip(clip_name, config, num_processes, threads_per_frame, project_root):
    """处理单个clip"""
    clip_id = get_clip_id(clip_name)
    direction = config['direction']

    print(f"\n{'='*80}")
    print(f"开始处理 Clip: {clip_name}")
    print(f"{'='*80}")

    # 获取路径
    scene_paths = common_utils.get_scene_paths(clip_name)
    if not scene_paths:
        print(f"  场景路径获取失败，跳过")
        return

    # 验证路径
    if not common_utils.validate_clip_paths(scene_paths, direction):
        print(f"  路径验证失败，跳过")
        return

    # 获取静态地图和动态点云目录
    static_map_path = get_static_map_path(direction)
    dynamic_pcd_dir = scene_paths['dynamic_pcd']
    transform_json = scene_paths['transform_json']

    print(f"\n  路径信息:")
    print(f"   静态地图: {static_map_path}")
    print(f"   动态PCD: {dynamic_pcd_dir}")
    print(f"   Transform JSON: {transform_json}")

    # 创建点云加载器
    print(f"\n  初始化点云加载器...")
    try:
        loader = MergedPCDLoader(static_map_path, dynamic_pcd_dir)
    except Exception as e:
        print(f"  点云加载器初始化失败: {e}")
        return

    # 获取所有时间戳
    timestamps = loader.get_timestamps()
    print(f"   可用时间戳数: {len(timestamps)}")

    if not timestamps:
        print(f"  没有找到可用的时间戳")
        return

    # 批次选择
    selected_timestamps = common_utils.get_batch_files(timestamps, config['batch_mode'])
    common_utils.print_batch_info(selected_timestamps, config['batch_mode'], len(timestamps))

    if not selected_timestamps:
        print(f"  没有选择任何时间戳")
        return

    # 加载并显示transform时间戳范围
    print(f"\n  加载变换矩阵...")
    transforms = common_utils.load_world2lidar_transforms(transform_json, show_range=True)

    # 创建输出目录
    output_root = Path(project_root) / clip_id
    output_paths = common_utils.get_unified_output_paths(str(output_root), clip_name, 'basic')
    common_utils.create_output_dirs(output_paths)

    print(f"\n  输出目录: {output_paths['root']}")

    # 准备任务列表
    print(f"\n  准备任务...")
    tasks = []
    output_root_path = Path(output_paths['root'])

    for ts in tqdm(selected_timestamps, desc="准备数据", unit="帧"):
        try:
            # 获取合并后的点云
            points, colors = loader.get_merged_pcd(ts)

            timestamp_ms = int(ts)
            output_frame_dir = output_root_path / str(timestamp_ms)

            tasks.append((
                points,
                colors,
                timestamp_ms,
                str(output_frame_dir),
                scene_paths['roadside_calib'],
                scene_paths['vehicle_calib'],
                scene_paths.get('vehicle_images', scene_paths['roadside_images']),
                transform_json,
                threads_per_frame
            ))
        except Exception as e:
            print(f"   跳过时间戳 {ts}: {e}")

    if not tasks:
        print(f"  没有有效的任务")
        return

    # 多进程处理
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
                        tqdm.write(f"  {timestamp_ms}")
                    else:
                        failed_list.append((timestamp_ms, message))
                        tqdm.write(f"  {timestamp_ms} - {message[:50]}")
                except Exception as e:
                    failed_list.append((timestamp_ms, str(e)))
                    tqdm.write(f"  {timestamp_ms} - 异常: {str(e)[:50]}")

                pbar.update(1)
                elapsed = time.time() - start_time
                if elapsed > 0:
                    fps = pbar.n / elapsed
                    pbar.set_postfix(速度=f"{fps:.2f}帧/秒", 成功率=f"{success_count/pbar.n*100:.1f}%")

    elapsed_time = time.time() - start_time

    # 结果统计
    print(f"\n{'='*80}")
    print(f"Clip {clip_id} 处理完成")
    print(f"{'='*80}")
    print(f"成功: {success_count}/{len(tasks)} ({success_count/len(tasks)*100:.1f}%)")
    print(f"耗时: {elapsed_time/60:.1f} 分钟")
    print(f"速度: {len(tasks)/elapsed_time:.2f} 帧/秒")

    if failed_list:
        failed_file = output_root_path / "failed_list.txt"
        with open(failed_file, 'w') as f:
            f.write(f"失败文件列表 ({len(failed_list)} 个)\n")
            f.write("="*50 + "\n\n")
            for timestamp, error in failed_list:
                f.write(f"时间戳: {timestamp}\n")
                f.write(f"错误: {error}\n")
                f.write("-"*30 + "\n")
        print(f"失败详情: {failed_file}")


def main():
    print("\n" + "="*80)
    print("  Dense Projection - 基本点云投影 (动静结合)")
    print("="*80)

    if not PROJECTOR_SCRIPT.exists():
        print(f"\n  找不到核心投影脚本: {PROJECTOR_SCRIPT}")
        sys.exit(1)

    # 交互式输入
    batch_mode = os.environ.get('PROJECTION_BATCH_MODE', 'false') == 'true'
    config = common_utils.interactive_input(batch_mode_enabled=batch_mode)
    if not config:
        print("  配置输入失败")
        sys.exit(1)

    # 并行配置
    parallel_config = common_utils.get_parallel_config(batch_mode_enabled=batch_mode)
    num_processes = parallel_config['num_processes']
    threads_per_frame = parallel_config['threads_per_frame']

    # 输出根目录
    output_root = Path(__file__).resolve().parent

    # 确认
    print(f"\n{'='*80}")
    print(f"  处理计划:")
    print(f"   方向: {config['direction']}")
    print(f"   Clip数量: {len(config['clips'])}")
    print(f"   批次模式: {config['batch_mode']}")
    print(f"   并行配置: {num_processes}进程 x {threads_per_frame}线程")
    print(f"   输出目录: {output_root}/{{clip_id}}/")
    print(f"{'='*80}")

    confirm = input("\n开始处理? (y/n): ").strip().lower()
    if confirm != 'y':
        print("  取消处理")
        sys.exit(0)

    # 处理每个clip
    overall_start = time.time()

    for clip_name in config['clips']:
        process_single_clip(
            clip_name, config, num_processes, threads_per_frame, output_root
        )

    overall_elapsed = time.time() - overall_start

    # 总结
    print(f"\n{'='*80}")
    print(f"  所有 Clips 处理完成!")
    print(f"{'='*80}")
    print(f"Clip数量: {len(config['clips'])}")
    print(f"总耗时: {overall_elapsed/60:.1f} 分钟")
    print(f"输出目录: {output_root}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  用户中断处理")
        sys.exit(1)
    except Exception as e:
        print(f"\n  发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
