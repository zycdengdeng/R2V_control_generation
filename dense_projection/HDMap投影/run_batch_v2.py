#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dense Projection - HDMap投影批量处理
将3D bbox投影到2D，支持动静结合的clips
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

PROJECTOR_SCRIPT = Path(__file__).resolve().parent / "undistort_projection_multithread_v2.py"


def run_single_projection(args):
    """运行单个投影任务"""
    annotation_path, timestamp_ms, output_dir, roadside_calib, vehicle_calib, \
    gt_images_folder, transform_json, ego_vehicle_id, threads_per_frame = args

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("hdmap_projector_v2", PROJECTOR_SCRIPT)
        projector_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(projector_module)

        if not hasattr(run_single_projection, 'transforms_cache'):
            run_single_projection.transforms_cache = {}

        if transform_json not in run_single_projection.transforms_cache:
            run_single_projection.transforms_cache[transform_json] = \
                common_utils.load_world2lidar_transforms(transform_json, show_range=False)

        transforms = run_single_projection.transforms_cache[transform_json]

        projector = projector_module.HDMapProjectorMultiThread(
            roadside_calib, vehicle_calib, gt_images_folder, transforms
        )

        success = projector.process_single_frame(
            annotation_path, output_dir, timestamp_ms, ego_vehicle_id, threads_per_frame
        )

        return success, "成功" if success else "处理失败", timestamp_ms

    except Exception as e:
        return False, str(e)[:200], timestamp_ms


def process_single_clip(clip_name, config, num_processes, threads_per_frame,
                        project_root, ego_vehicle_mapping):
    """处理单个clip"""
    clip_id = get_clip_id(clip_name)
    direction = config['direction']

    print(f"\n{'='*80}")
    print(f"开始处理 Clip: {clip_name}")
    print(f"{'='*80}")

    ego_vehicle_id = ego_vehicle_mapping.get(clip_name, 45)
    print(f"  自车ID: {ego_vehicle_id}")

    scene_paths = common_utils.get_scene_paths(clip_name)
    if not scene_paths:
        print(f"  场景路径获取失败，跳过")
        return

    # HDMap使用标注文件
    label_folder = Path(scene_paths.get('roadside_labels', ''))
    if not label_folder.exists():
        print(f"  标注文件夹不存在: {label_folder}")
        return

    annotation_files = sorted(label_folder.glob("*.json"))
    if not annotation_files:
        print(f"  没有找到标注文件")
        return

    annotation_files = common_utils.sort_files_by_timestamp(annotation_files)
    selected_files = common_utils.get_batch_files(annotation_files, config['batch_mode'])

    if not selected_files:
        return

    transform_json = scene_paths['transform_json']
    transforms = common_utils.load_world2lidar_transforms(transform_json, show_range=True)

    output_root = Path(project_root) / clip_id
    output_paths = common_utils.get_unified_output_paths(str(output_root), clip_name, 'hdmap')
    common_utils.create_output_dirs(output_paths)

    tasks = []
    output_root_path = Path(output_paths['root'])

    for annotation_file in selected_files:
        timestamp_ms = common_utils.extract_timestamp_from_filename(annotation_file)
        if timestamp_ms is None:
            continue

        output_frame_dir = output_root_path / str(int(timestamp_ms))

        tasks.append((
            str(annotation_file),
            int(timestamp_ms),
            str(output_frame_dir),
            scene_paths['roadside_calib'],
            scene_paths['vehicle_calib'],
            scene_paths.get('vehicle_images', scene_paths['roadside_images']),
            transform_json,
            ego_vehicle_id,
            threads_per_frame
        ))

    if not tasks:
        return

    success_count = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {executor.submit(run_single_projection, task): task for task in tasks}

        with tqdm(total=len(tasks), desc=f"Clip {clip_id}", unit="帧") as pbar:
            for future in as_completed(futures):
                try:
                    success, _, _ = future.result()
                    if success:
                        success_count += 1
                except:
                    pass
                pbar.update(1)

    elapsed_time = time.time() - start_time
    print(f"\n成功: {success_count}/{len(tasks)}, 耗时: {elapsed_time/60:.1f} 分钟")


def main():
    print("\n" + "="*80)
    print("  Dense Projection - HDMap投影 (3D bbox -> 2D bbox)")
    print("="*80)

    if not PROJECTOR_SCRIPT.exists():
        sys.exit(1)

    batch_mode = os.environ.get('PROJECTION_BATCH_MODE', 'false') == 'true'
    config = common_utils.interactive_input(batch_mode_enabled=batch_mode)
    if not config:
        sys.exit(1)

    ego_vehicle_mapping = common_utils.get_ego_vehicle_id(
        clip_names=config['clips'],
        batch_mode_enabled=batch_mode,
        default_id=45
    )

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
            output_root, ego_vehicle_mapping
        )


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
