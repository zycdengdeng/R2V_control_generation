#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量更新 Transfer2 数据集的 caption

支持：
- 自动识别所有7个相机目录
- 支持模板变量：{camera}, {scene}, {seg}
- 预览更改后再应用
- 可选择单个或多个数据集
"""

import json
from pathlib import Path
import argparse
import sys

# 相机名称列表（Transfer2格式）
CAMERA_NAMES = [
    'ftheta_camera_front_tele_30fov',
    'ftheta_camera_front_wide_120fov',
    'ftheta_camera_cross_left_120fov',
    'ftheta_camera_cross_right_120fov',
    'ftheta_camera_rear_left_70fov',
    'ftheta_camera_rear_right_70fov',
    'ftheta_camera_rear_tele_30fov'
]

# 预定义caption模板（1-6 是预设模板，应用于不同的功能）
PRESET_TEMPLATES = {
    'depth': 'Sparse depth map from roadside LiDAR projected to {camera} viewpoint. Northern Chinese suburban intersection in early spring. Multi-lane asphalt roads with white lane markings and directional arrows. Beige/tan high-rise residential buildings. Bare deciduous trees. Red lanterns on lamp posts. Traffic signals and overhead signs. Mixed traffic including motor vehicles (cars, SUVs, buses, trucks) and non-motorized road users (pedestrians, cyclists, tricycles).',

    'depth_dense': 'Dense depth map from roadside LiDAR with interpolation, projected to {camera} viewpoint. Suburban intersection with wide multi-lane roads, white lane markings, warm-toned high-rise apartments, bare winter trees, red lanterns, traffic lights. Mixed traffic including motor vehicles (cars, SUVs, buses, trucks) and non-motorized road users (pedestrians, cyclists, tricycles).',

    'hdmap': '3D bounding box annotations projected to {camera} viewpoint. Urban intersection with detected motor vehicles (cars, SUVs, buses, trucks) and non-motorized road users (pedestrians, cyclists, tricycles). Multi-lane roads with directional markings. Chinese suburban environment with residential towers, bare trees, and traffic infrastructure.',

    'blur': 'Sparse colored point cloud from roadside LiDAR-camera fusion, projected to {camera} viewpoint. Northern Chinese suburban intersection in early spring. Wide roads with white markings. Beige high-rise apartments. Bare trees. Red lanterns. Blue sky. Mixed traffic including motor vehicles (cars, SUVs, buses, trucks) and non-motorized road users (pedestrians, cyclists, tricycles).',

    'blur_dense': 'Densified colored point cloud from roadside LiDAR-camera fusion, projected to {camera} viewpoint. Suburban intersection with multi-lane roads, warm-toned residential buildings, leafless trees, red lantern decorations, traffic signals. Mixed traffic including motor vehicles (cars, SUVs, buses, trucks) and non-motorized road users (pedestrians, cyclists, tricycles).',

    'basic': 'RGB frame from {camera} viewpoint of ego vehicle. Northern Chinese suburban intersection in early spring. Wide multi-lane roads with white lane markings. Beige/tan high-rise apartments. Bare deciduous trees. Red lanterns on lamp posts. Traffic lights and overhead signs. Blue sky. Mixed traffic including motor vehicles (cars, SUVs, buses, trucks) and non-motorized road users (pedestrians, cyclists, tricycles).'
}

# 数据集名称到模板的自动映射
DATASET_TEMPLATE_MAPPING = {
    'DepthSparse': 'depth',
    'DepthDense': 'depth_dense',
    'HDMapBbox': 'hdmap',
    'BlurProjection': 'blur',
    'BlurDense': 'blur_dense',
    'BasicProjection': 'basic'
}


def find_datasets(base_dir):
    """
    查找所有数据集目录

    Args:
        base_dir: 基础目录

    Returns:
        dataset_dirs: 数据集目录列表
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"错误: 目录不存在: {base_dir}")
        return []

    # 查找所有包含 captions/ 子目录的目录
    dataset_dirs = []
    for item in base_path.iterdir():
        if item.is_dir():
            captions_dir = item / 'captions'
            if captions_dir.exists() and captions_dir.is_dir():
                dataset_dirs.append(item)

    return sorted(dataset_dirs)


def get_caption_files(dataset_dir):
    """
    获取数据集中所有caption JSON文件

    Args:
        dataset_dir: 数据集目录

    Returns:
        caption_files: JSON文件路径列表
    """
    captions_dir = dataset_dir / 'captions'
    caption_files = []

    # 遍历所有相机目录
    for cam_name in CAMERA_NAMES:
        cam_dir = captions_dir / cam_name
        if cam_dir.exists():
            # 收集该相机的所有JSON文件
            json_files = sorted(cam_dir.glob('*.json'))
            caption_files.extend(json_files)

    return caption_files


def parse_caption_filename(json_path):
    """
    从JSON文件名解析信息

    Args:
        json_path: JSON文件路径

    Returns:
        info: {camera, scene, seg} 字典
    """
    camera = json_path.parent.name
    filename = json_path.stem  # 如 "002_seg01"

    parts = filename.split('_')
    scene = parts[0]
    seg = parts[1] if len(parts) > 1 else 'seg01'

    return {
        'camera': camera,
        'scene': scene,
        'seg': seg
    }


def generate_caption(template, info):
    """
    根据模板生成caption

    Args:
        template: Caption模板
        info: {camera, scene, seg} 字典

    Returns:
        caption: 生成的caption
    """
    return template.format(**info)


def preview_changes(caption_files, template, max_preview=5):
    """
    预览将要做的更改

    Args:
        caption_files: JSON文件路径列表
        template: 新的caption模板
        max_preview: 最多预览多少个
    """
    print(f"\n预览更改（显示前 {max_preview} 个）:")
    print("=" * 80)

    for i, json_path in enumerate(caption_files[:max_preview]):
        # 读取当前内容
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        old_caption = data.get('caption', '')

        # 生成新caption
        info = parse_caption_filename(json_path)
        new_caption = generate_caption(template, info)

        print(f"\n文件: {json_path.relative_to(json_path.parents[3])}")
        print(f"  旧: {old_caption}")
        print(f"  新: {new_caption}")

    if len(caption_files) > max_preview:
        print(f"\n... 还有 {len(caption_files) - max_preview} 个文件")

    print(f"\n总计: {len(caption_files)} 个文件将被更新")
    print("=" * 80)


def update_captions(caption_files, template, dry_run=False):
    """
    更新所有caption

    Args:
        caption_files: JSON文件路径列表
        template: 新的caption模板
        dry_run: 是否只是预览，不实际修改

    Returns:
        success_count: 成功更新的文件数
    """
    success_count = 0

    for json_path in caption_files:
        try:
            # 读取JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 生成新caption
            info = parse_caption_filename(json_path)
            new_caption = generate_caption(template, info)

            # 更新caption
            data['caption'] = new_caption

            if not dry_run:
                # 写回文件
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            success_count += 1

        except Exception as e:
            print(f"错误: 处理文件 {json_path} 失败: {e}")

    return success_count


def interactive_mode(base_dir):
    """
    交互式模式

    Args:
        base_dir: 数据集基础目录
    """
    print("=" * 80)
    print("Transfer2 Caption 批量更新工具")
    print("=" * 80)

    # 查找所有数据集
    datasets = find_datasets(base_dir)

    if not datasets:
        print(f"错误: 在 {base_dir} 下未找到任何数据集")
        return

    print(f"\n找到 {len(datasets)} 个数据集:")
    for i, ds in enumerate(datasets, 1):
        num_files = len(get_caption_files(ds))
        print(f"  {i}) {ds.name} ({num_files} 个caption文件)")

    print(f"  {len(datasets) + 1}) 全部数据集")
    print("  0) 退出")

    # 选择数据集
    choice = input(f"\n请选择数据集 [1-{len(datasets) + 1}, 0]: ").strip()

    if choice == '0':
        print("已退出")
        return

    try:
        choice_num = int(choice)
        if choice_num == len(datasets) + 1:
            selected_datasets = datasets
            print(f"\n已选择: 全部 {len(datasets)} 个数据集")
        elif 1 <= choice_num <= len(datasets):
            selected_datasets = [datasets[choice_num - 1]]
            print(f"\n已选择: {selected_datasets[0].name}")
        else:
            print("无效选择")
            return
    except ValueError:
        print("无效输入")
        return

    # 收集所有caption文件
    all_caption_files = []
    for ds in selected_datasets:
        all_caption_files.extend(get_caption_files(ds))

    if not all_caption_files:
        print("错误: 未找到任何caption文件")
        return

    print(f"\n总计: {len(all_caption_files)} 个caption文件")

    # 显示预设模板
    print("\n预设caption模板:")

    # 如果选择了多个数据集，提供自动匹配选项
    if len(selected_datasets) > 1:
        print(f"  0) 自动匹配（每个数据集使用对应的预设模板）")

    presets = list(PRESET_TEMPLATES.items())
    for i, (key, template) in enumerate(presets, 1):
        print(f"  {i}) {key}: \"{template}\"")
    print(f"  {len(presets) + 1}) 自定义模板")

    # 选择模板
    if len(selected_datasets) > 1:
        template_choice = input(f"\n请选择模板 [0-{len(presets) + 1}]: ").strip()
    else:
        template_choice = input(f"\n请选择模板 [1-{len(presets) + 1}]: ").strip()

    try:
        template_num = int(template_choice)

        # 自动匹配模式（仅在选择了多个数据集时可用）
        if template_num == 0 and len(selected_datasets) > 1:
            print("\n使用自动匹配模式：每个数据集使用对应的预设模板")

            # 显示匹配结果
            print("\n匹配结果:")
            for ds in selected_datasets:
                dataset_name = ds.name
                matched_template_key = DATASET_TEMPLATE_MAPPING.get(dataset_name, None)
                if matched_template_key:
                    print(f"  {dataset_name} → {matched_template_key}")
                else:
                    print(f"  {dataset_name} → [无匹配模板，将跳过]")

            # 确认
            confirm = input("\n确认使用自动匹配更新? [y/N]: ").strip().lower()

            if confirm != 'y':
                print("已取消")
                return

            # 执行自动匹配更新
            print("\n正在更新...")
            total_success = 0

            for ds in selected_datasets:
                dataset_name = ds.name
                matched_template_key = DATASET_TEMPLATE_MAPPING.get(dataset_name, None)

                if not matched_template_key:
                    print(f"  跳过 {dataset_name}：未找到匹配的模板")
                    continue

                template = PRESET_TEMPLATES[matched_template_key]
                caption_files = get_caption_files(ds)

                if not caption_files:
                    print(f"  跳过 {dataset_name}：未找到caption文件")
                    continue

                print(f"\n  处理 {dataset_name} (使用模板: {matched_template_key})...")
                success = update_captions(caption_files, template, dry_run=False)
                total_success += success
                print(f"    ✓ 成功更新 {success}/{len(caption_files)} 个文件")

            print(f"\n✓ 完成! 总计成功更新 {total_success}/{len(all_caption_files)} 个文件")
            return

        # 手动选择模板（单个模板应用到所有数据集）
        elif 1 <= template_num <= len(presets):
            template = presets[template_num - 1][1]
            print(f"\n使用模板: \"{template[:80]}...\"")
        elif template_num == len(presets) + 1:
            template = input("\n请输入自定义模板（可用变量: {camera}, {scene}, {seg}）: ").strip()
            if not template:
                print("错误: 模板不能为空")
                return
        else:
            print("无效选择")
            return
    except ValueError:
        print("无效输入")
        return

    # 预览更改
    preview_changes(all_caption_files, template, max_preview=5)

    # 确认
    confirm = input("\n确认更新所有caption? [y/N]: ").strip().lower()

    if confirm != 'y':
        print("已取消")
        return

    # 执行更新
    print("\n正在更新...")
    success_count = update_captions(all_caption_files, template, dry_run=False)

    print(f"\n✓ 完成! 成功更新 {success_count}/{len(all_caption_files)} 个文件")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量更新 Transfer2 数据集的 caption')

    parser.add_argument('--base-dir', type=str,
                        default='/mnt/zihanw/proj_utils_pro/transfer_video_maker/output',
                        help='数据集基础目录')

    parser.add_argument('--dataset', type=str,
                        help='指定数据集名称（如 DepthSparse）')

    parser.add_argument('--template', type=str,
                        help='Caption模板（支持 {camera}, {scene}, {seg}）')

    parser.add_argument('--preset', type=str,
                        choices=list(PRESET_TEMPLATES.keys()),
                        help='使用预设模板')

    parser.add_argument('--dry-run', action='store_true',
                        help='只预览，不实际修改')

    parser.add_argument('--interactive', action='store_true',
                        help='交互式模式（默认）')

    args = parser.parse_args()

    # 如果没有指定任何参数，使用交互式模式
    if len(sys.argv) == 1 or args.interactive:
        interactive_mode(args.base_dir)
        return

    # 命令行模式
    if args.preset:
        template = PRESET_TEMPLATES[args.preset]
    elif args.template:
        template = args.template
    else:
        print("错误: 请指定 --template 或 --preset")
        return

    base_path = Path(args.base_dir)

    if args.dataset:
        # 处理单个数据集
        dataset_path = base_path / args.dataset
        if not dataset_path.exists():
            print(f"错误: 数据集不存在: {dataset_path}")
            return
        datasets = [dataset_path]
    else:
        # 处理所有数据集
        datasets = find_datasets(base_path)

    if not datasets:
        print("错误: 未找到任何数据集")
        return

    # 收集所有文件
    all_caption_files = []
    for ds in datasets:
        all_caption_files.extend(get_caption_files(ds))

    if not all_caption_files:
        print("错误: 未找到任何caption文件")
        return

    # 预览
    preview_changes(all_caption_files, template)

    if args.dry_run:
        print("\n[Dry-run 模式] 未实际修改文件")
        return

    # 执行更新
    print("\n正在更新...")
    success_count = update_captions(all_caption_files, template, dry_run=False)
    print(f"\n✓ 完成! 成功更新 {success_count}/{len(all_caption_files)} 个文件")


if __name__ == "__main__":
    main()
