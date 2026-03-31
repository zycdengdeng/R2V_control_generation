#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复缺失的GT图像
通过最近邻插值来填补缺失的GT图像

用法:
    python fix_missing_gt.py --input_dir <投影输出目录> --scenes 082 083 084 085 086

示例:
    python fix_missing_gt.py --input_dir /path/to/BlurProjection --scenes 082 083 084 085 086 --dry_run
    python fix_missing_gt.py --input_dir /path/to/BlurProjection --scenes 082 083 084 085 086
"""

import os
import argparse
import shutil
from pathlib import Path
from collections import defaultdict

# 相机列表
CAMERA_NAMES = ['FN', 'FW', 'FL', 'FR', 'RL', 'RR', 'RN']


def get_timestamp_from_folder(folder_name: str) -> int:
    """从文件夹名称提取时间戳（毫秒）"""
    try:
        return int(folder_name)
    except ValueError:
        return -1


def scan_scene_gt_status(scene_dir: Path, cameras: list) -> dict:
    """
    扫描场景目录，获取每个时间戳每个相机的GT状态

    Args:
        scene_dir: 场景目录路径
        cameras: 需要检查的相机列表

    Returns:
        {
            timestamp: {
                'folder': Path,
                'cameras': {
                    'FN': True/False,  # 是否存在
                    ...
                }
            }
        }
    """
    result = {}

    # 找到场景的内部目录 (scene_dir/scene_id/timestamp_folders)
    scene_id = scene_dir.name
    inner_dir = scene_dir / scene_id

    if not inner_dir.exists():
        print(f"  警告: 内部目录不存在 {inner_dir}")
        return result

    # 遍历所有时间戳文件夹
    timestamp_folders = sorted([
        d for d in inner_dir.iterdir()
        if d.is_dir() and get_timestamp_from_folder(d.name) > 0
    ], key=lambda x: get_timestamp_from_folder(x.name))

    for ts_folder in timestamp_folders:
        timestamp = get_timestamp_from_folder(ts_folder.name)
        gt_folder = ts_folder / 'gt'

        camera_status = {}
        for cam in cameras:
            gt_image = gt_folder / f"{cam}.jpg"
            camera_status[cam] = gt_image.exists()

        result[timestamp] = {
            'folder': ts_folder,
            'cameras': camera_status
        }

    return result


def find_nearest_valid_timestamp(gt_status: dict, target_ts: int, camera: str) -> int:
    """
    找到最近的有效时间戳（该相机存在GT图像）

    Args:
        gt_status: scan_scene_gt_status 的返回值
        target_ts: 目标时间戳
        camera: 相机名称

    Returns:
        最近的有效时间戳，如果没找到返回 -1
    """
    timestamps = sorted(gt_status.keys())

    if target_ts not in timestamps:
        return -1

    target_idx = timestamps.index(target_ts)

    # 向左右两边搜索最近的有效时间戳
    best_ts = -1
    best_dist = float('inf')

    # 向左搜索
    for i in range(target_idx - 1, -1, -1):
        ts = timestamps[i]
        if gt_status[ts]['cameras'].get(camera, False):
            dist = target_ts - ts
            if dist < best_dist:
                best_dist = dist
                best_ts = ts
            break  # 找到最近的就停止

    # 向右搜索
    for i in range(target_idx + 1, len(timestamps)):
        ts = timestamps[i]
        if gt_status[ts]['cameras'].get(camera, False):
            dist = ts - target_ts
            if dist < best_dist:
                best_dist = dist
                best_ts = ts
            break  # 找到最近的就停止

    return best_ts


def fix_scene_missing_gt(scene_dir: Path, cameras: list, dry_run: bool = True) -> dict:
    """
    修复场景中缺失的GT图像

    Args:
        scene_dir: 场景目录路径
        cameras: 需要检查的相机列表
        dry_run: 如果为True，只报告不实际复制

    Returns:
        修复统计信息
    """
    scene_id = scene_dir.name
    print(f"\n{'='*60}")
    print(f"处理场景: {scene_id}")
    print(f"{'='*60}")

    # 扫描GT状态
    gt_status = scan_scene_gt_status(scene_dir, cameras)

    if not gt_status:
        print(f"  未找到有效的时间戳文件夹")
        return {'scene': scene_id, 'fixed': 0, 'failed': 0}

    print(f"  找到 {len(gt_status)} 个时间戳文件夹")

    # 统计每个相机的缺失情况
    missing_stats = defaultdict(list)
    for ts, info in gt_status.items():
        for cam, exists in info['cameras'].items():
            if not exists:
                missing_stats[cam].append(ts)

    # 显示缺失统计
    print(f"\n  缺失统计:")
    for cam in cameras:
        missing_count = len(missing_stats[cam])
        total_count = len(gt_status)
        if missing_count > 0:
            print(f"    {cam}: 缺失 {missing_count}/{total_count} 帧")

    if not any(missing_stats.values()):
        print(f"  所有GT图像完整，无需修复")
        return {'scene': scene_id, 'fixed': 0, 'failed': 0}

    # 修复缺失的图像
    fixed_count = 0
    failed_count = 0

    print(f"\n  {'[DRY RUN] ' if dry_run else ''}开始修复...")

    for cam, missing_timestamps in missing_stats.items():
        for target_ts in missing_timestamps:
            # 找到最近的有效时间戳
            nearest_ts = find_nearest_valid_timestamp(gt_status, target_ts, cam)

            if nearest_ts < 0:
                print(f"    ✗ {cam} @ {target_ts}: 找不到有效的邻近帧")
                failed_count += 1
                continue

            # 构建源和目标路径
            source_folder = gt_status[nearest_ts]['folder']
            target_folder = gt_status[target_ts]['folder']

            source_path = source_folder / 'gt' / f"{cam}.jpg"
            target_path = target_folder / 'gt' / f"{cam}.jpg"

            # 确保目标目录存在
            target_path.parent.mkdir(parents=True, exist_ok=True)

            distance = abs(target_ts - nearest_ts)

            if dry_run:
                print(f"    [DRY] {cam} @ {target_ts}: 将从 {nearest_ts} 复制 (距离: {distance}ms)")
                fixed_count += 1  # dry_run模式也计数
            else:
                try:
                    shutil.copy2(source_path, target_path)
                    print(f"    ✓ {cam} @ {target_ts}: 已从 {nearest_ts} 复制 (距离: {distance}ms)")
                    fixed_count += 1
                except Exception as e:
                    print(f"    ✗ {cam} @ {target_ts}: 复制失败 - {e}")
                    failed_count += 1

    return {
        'scene': scene_id,
        'fixed': fixed_count,
        'failed': failed_count
    }


def main():
    parser = argparse.ArgumentParser(description='修复缺失的GT图像')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='投影输出目录 (如 /path/to/BlurProjection)')
    parser.add_argument('--scenes', type=str, nargs='+', required=True,
                        help='需要修复的场景ID列表 (如 082 083 084)')
    parser.add_argument('--dry_run', action='store_true',
                        help='只报告缺失情况，不实际复制')
    parser.add_argument('--cameras', type=str, nargs='+', default=CAMERA_NAMES,
                        help=f'需要检查的相机列表 (默认: {CAMERA_NAMES})')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        print(f"错误: 输入目录不存在 {input_dir}")
        return

    print("="*60)
    print("GT图像修复工具")
    print("="*60)
    print(f"输入目录: {input_dir}")
    print(f"场景列表: {', '.join(args.scenes)}")
    print(f"相机列表: {', '.join(args.cameras)}")
    print(f"模式: {'DRY RUN (仅报告)' if args.dry_run else '实际修复'}")

    # 使用用户指定的相机列表
    cameras_to_check = args.cameras

    # 处理每个场景
    total_fixed = 0
    total_failed = 0

    for scene_id in args.scenes:
        scene_dir = input_dir / scene_id

        if not scene_dir.exists():
            print(f"\n警告: 场景目录不存在 {scene_dir}")
            continue

        result = fix_scene_missing_gt(scene_dir, cameras_to_check, dry_run=args.dry_run)
        total_fixed += result['fixed']
        total_failed += result['failed']

    # 总结
    print("\n" + "="*60)
    print("修复总结")
    print("="*60)
    if args.dry_run:
        print(f"[DRY RUN] 预计修复: {total_fixed} 个图像")
        print(f"[DRY RUN] 无法修复: {total_failed} 个图像")
        print(f"\n提示: 移除 --dry_run 参数以实际执行修复")
    else:
        print(f"成功修复: {total_fixed} 个图像")
        print(f"修复失败: {total_failed} 个图像")


if __name__ == '__main__':
    main()