#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量depth稠密化投影处理 V2 - 统一交互版
支持多场景、统一批次选择、固定标定路径
输出深度图：近白远黑+稠密化，.npy + .jpg
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加父目录到路径（使用绝对路径）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common_utils

# 核心投影脚本路径（绝对路径）
PROJECTOR_SCRIPT = Path(__file__).resolve().parent / "undistort_projection_multithread_v2.py"


def run_single_projection(args):
    """运行单个投影任务"""
    pcd_path, timestamp_ms, output_dir, roadside_calib, vehicle_calib, \
    gt_images_folder, transform_json, threads_per_frame = args

    try:
        # 动态导入核心模块
        import importlib.util
        spec = importlib.util.spec_from_file_location("depth_dense_projector_v2", PROJECTOR_SCRIPT)
        projector_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(projector_module)

        # 加载变换矩阵（每个进程加载一次）
        if not hasattr(run_single_projection, 'transforms_cache'):
            run_single_projection.transforms_cache = {}

        if transform_json not in run_single_projection.transforms_cache:
            run_single_projection.transforms_cache[transform_json] = \
                common_utils.load_world2lidar_transforms(transform_json, show_range=False)

        transforms = run_single_projection.transforms_cache[transform_json]

        # 创建投影器
        projector = projector_module.DepthDenseProjectorMultiThread(
            roadside_calib, vehicle_calib, gt_images_folder, transforms
        )

        # 处理单帧
        success = projector.process_single_frame(
            pcd_path, output_dir, timestamp_ms, threads_per_frame
        )

        return success, "成功" if success else "处理失败", timestamp_ms

    except Exception as e:
        error_msg = str(e)
        return False, error_msg[:100], timestamp_ms


def get_scene_transform_json(config, scene_id):
    """获取场景对应的transform JSON路径"""
    transform_json = config['transform_json']

    # 如果是字典，根据scene_id获取对应路径
    if isinstance(transform_json, dict):
        return transform_json.get(scene_id)
    # 如果是字符串，直接返回（所有场景共用）
    return transform_json


def process_single_scene(scene_id, config, num_processes, threads_per_frame, project_root):
    """处理单个场景"""
    print(f"\n{'='*80}")
    print(f"开始处理场景: {scene_id}")
    print(f"{'='*80}")

    # 获取当前场景的transform JSON路径
    scene_transform_json = get_scene_transform_json(config, scene_id)
    if not scene_transform_json:
        print(f"❌ 场景 {scene_id} 缺少transform JSON路径")
        return

    # 为当前场景创建独立的输出目录
    output_root = Path(project_root) / scene_id
    print(f"📂 输出目录: {output_root}")
    print(f"📄 Transform JSON: {scene_transform_json}")

    # 获取场景路径
    scene_paths = common_utils.get_scene_paths(scene_id)
    if not scene_paths or not common_utils.validate_scene_paths(scene_paths):
        print(f"❌ 场景 {scene_id} 路径验证失败，跳过")
        return

    # 获取PCD文件列表
    pcd_folder = Path(scene_paths['pcd'])
    pcd_files = sorted(pcd_folder.glob("*.pcd"))

    if not pcd_files:
        print(f"❌ 场景 {scene_id} 没有找到PCD文件")
        return

    # 按时间戳排序
    pcd_files = common_utils.sort_files_by_timestamp(pcd_files)

    print(f"\n📁 场景信息:")
    print(f"   名称: {scene_paths['scene_name']}")
    print(f"   PCD文件数: {len(pcd_files)}")

    # 批次选择
    selected_files = common_utils.get_batch_files(pcd_files, config['batch_mode'])
    common_utils.print_batch_info(selected_files, config['batch_mode'], len(pcd_files))

    if not selected_files:
        print(f"❌ 没有选择任何文件")
        return

    # 诊断：检查时间戳范围
    print(f"\n🔍 时间戳诊断:")
    pcd_timestamps = [common_utils.extract_timestamp_from_filename(f) for f in selected_files]
    pcd_timestamps = [t for t in pcd_timestamps if t is not None]
    if pcd_timestamps:
        print(f"   PCD时间戳范围: {min(pcd_timestamps):.0f} ~ {max(pcd_timestamps):.0f}")
        print(f"   PCD时间跨度: {(max(pcd_timestamps) - min(pcd_timestamps)) / 1000:.1f} 秒")

    # 加载并显示transform时间戳范围
    transforms = common_utils.load_world2lidar_transforms(scene_transform_json, show_range=True)

    # 创建输出目录
    output_paths = common_utils.get_unified_output_paths(output_root, scene_id, 'depth_dense')
    common_utils.create_output_dirs(output_paths)

    print(f"\n📂 输出目录: {output_paths['root']}")

    # 准备任务列表
    tasks = []
    output_root_path = Path(output_paths['root'])
    for pcd_file in selected_files:
        timestamp_ms = common_utils.extract_timestamp_from_filename(pcd_file)
        if timestamp_ms is None:
            continue

        output_frame_dir = output_root_path / str(int(timestamp_ms))

        tasks.append((
            str(pcd_file),
            int(timestamp_ms),
            str(output_frame_dir),
            scene_paths['roadside_calib'],
            scene_paths['vehicle_calib'],
            scene_paths.get('vehicle_images', scene_paths['roadside_images']),  # 优先使用车端GT图像
            scene_transform_json,
            threads_per_frame
        ))

    # 多进程处理
    print(f"\n🚀 开始处理 ({num_processes}进程 × {threads_per_frame}线程)...")
    success_count = 0
    failed_list = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {executor.submit(run_single_projection, task): task for task in tasks}

        with tqdm(total=len(tasks), desc=f"场景{scene_id}", unit="帧") as pbar:
            for future in as_completed(futures):
                task = futures[future]
                timestamp_ms = task[1]

                try:
                    success, message, _ = future.result()

                    if success:
                        success_count += 1
                        tqdm.write(f"✓ {timestamp_ms}")
                    else:
                        failed_list.append((timestamp_ms, message))
                        tqdm.write(f"✗ {timestamp_ms} - {message}")
                except Exception as e:
                    failed_list.append((timestamp_ms, str(e)))
                    tqdm.write(f"✗ {timestamp_ms} - 异常: {str(e)[:50]}")

                pbar.update(1)
                elapsed = time.time() - start_time
                if elapsed > 0:
                    fps = pbar.n / elapsed
                    pbar.set_postfix(速度=f"{fps:.2f}帧/秒", 成功率=f"{success_count/pbar.n*100:.1f}%")

    elapsed_time = time.time() - start_time

    # 结果统计
    print(f"\n{'='*80}")
    print(f"场景 {scene_id} 处理完成")
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
    print("Depth稠密化投影 - 批量处理工具 V2 (深度图：近白远黑+稠密化)")
    print("="*80)

    if not PROJECTOR_SCRIPT.exists():
        print(f"\n❌ 找不到核心投影脚本: {PROJECTOR_SCRIPT}")
        sys.exit(1)

    # 统一交互式输入（支持批量模式）
    batch_mode = os.environ.get('PROJECTION_BATCH_MODE', 'false') == 'true'
    config = common_utils.interactive_input(batch_mode_enabled=batch_mode)
    if not config:
        print("❌ 配置输入失败")
        sys.exit(1)

    # 并行配置（支持批量模式）
    parallel_config = common_utils.get_parallel_config(batch_mode_enabled=batch_mode)
    num_processes = parallel_config['num_processes']
    threads_per_frame = parallel_config['threads_per_frame']

    # 输出根目录（固定为当前项目目录）
    output_root = Path(__file__).resolve().parent

    # 确认
    print(f"\n{'='*80}")
    print(f"📋 处理计划:")
    print(f"   场景数量: {len(config['scene_ids'])}")
    print(f"   场景列表: {', '.join(config['scene_ids'])}")
    print(f"   批次模式: {config['batch_mode']}")
    print(f"   并行配置: {num_processes}进程 × {threads_per_frame}线程")
    print(f"   输出目录: {output_root}/{{场景ID}}/")
    print(f"{'='*80}")

    # confirm = input("\n开始处理? (y/n): ").strip().lower()
    # if confirm != 'y':
    #     print("❌ 取消处理")
    #     sys.exit(0)

    # 处理每个场景
    overall_start = time.time()

    for scene_id in config['scene_ids']:
        process_single_scene(
            scene_id, config, num_processes, threads_per_frame, output_root
        )

    overall_elapsed = time.time() - overall_start

    # 总结
    print(f"\n{'='*80}")
    print(f"🎉 所有场景处理完成!")
    print(f"{'='*80}")
    print(f"场景数量: {len(config['scene_ids'])}")
    print(f"总耗时: {overall_elapsed/60:.1f} 分钟")
    print(f"输出目录: {output_root}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # 设置多进程启动方式
    mp.set_start_method('spawn', force=True)

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断处理")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
