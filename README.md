# R2V Control Generation

车路协同（Road-to-Vehicle）控制信号生成工具。从路侧传感器数据生成多种投影控制帧，再组装为视频数据集用于模型训练。

## 整体流程

```
conda activate zihanw
```

**Step 1** 控制帧图像生成 → **Step 2** 控制头视频生成 → **Step 3** Caption 替换

最终用于训练的数据在 `./transfer_video_maker/output/`。

---

## Step 1: 控制帧图像生成

```bash
./run_interactive.sh
```

支持的投影类型：

| 选项 | 类型 | 说明 |
|------|------|------|
| 1 | basic | 基本路侧 merge 激光雷达点云投影 |
| 2 | blur | 路侧相机着色 merge 点云投影 |
| 3 | blur_dense | blur 投影后规则稠密化 |
| 4 | depth | merge 点云生成深度图 |
| 5 | depth_dense | 深度图后规则稠密化 |
| 6 | hdmap | HDMap 3D→2D bbox 投影 |
| 7 | batch | 批量处理（多选串行执行） |

**操作说明：**
1. 一般选 **7 (batch)**，然后多选需要的类型（如 `2 4 6` = blur + depth + HDMap）
2. 场景编号直接填 `001`-`089`，脚本会自动查找对应路径
3. 线程数、GPU 等参数有默认值，直接回车即可
4. HDMap 任务会额外要求填写自车ID，输入 `auto` 即可
5. 输出为逐帧图片，按场景分类存放在各投影子目录下

---

## Step 2: 控制头视频生成

```bash
./transfer_video_maker/generate_videos.sh
```

1. 选 **7 (batch)**，多选需要的投影类型
2. **每个 seg 帧数填 29**（模型要求），不要用默认值
3. seg 数量 = 总帧数 / 29（如产了 90 帧，则 seg 数量填 3）
4. 视频帧率、分辨率（默认 1280×720, 10fps）符合 Cosmos post-training 配置，直接回车
5. 输出在 `./transfer_video_maker/output/`，按控制头类型分类
6. **生成完视频后删除 Step 1 的帧数据**（`blur投影/`、`depth投影/`、`HDMap投影/` 下的图片），占用空间大

---

## Step 3: Caption 替换

```bash
python3 ./transfer_video_maker/caption一键修理/update_captions.py
```

1. 选”全部数据集” + “自动匹配”即可批量替换
2. 如需自定义 caption 模板，修改脚本内的预设文本

---

## 项目结构

```
.
├── run_interactive.sh          # Step 1 入口：交互式投影处理
├── common_utils.py             # 共享工具（路径查找、标定加载等）
├── blur投影/                   # blur 投影模块
├── blur稠密化投影/             # blur 稠密化模块
├── depth投影/                  # depth 投影模块
├── depth稠密化投影/            # depth 稠密化模块
├── HDMap投影/                  # HDMap bbox 投影模块
├── 基本点云投影/               # 基本点云投影模块
├── dense_projection/           # 稠密投影（含视频生成）
├── transfer_video_maker/       # Step 2 & 3：视频生成 + caption 替换
│   ├── generate_videos.sh      # Step 2 入口
│   ├── generate_transfer2_videos.py
│   ├── caption一键修理/        # Step 3：caption 批量替换
│   └── output/                 # 最终训练数据输出
└── transform_json/             # 坐标变换 JSON 配置
```

---

## 故障排查：掉帧检查与修复

某些 clip 可能出现掉帧（视频不足 29 帧），用以下命令检查和修复：

```bash
cd ./transfer_video_maker

# 检查（dry_run）
for proj in “blur投影” “depth投影” “HDMap投影”; do
    python fix_missing_gt.py --input_dir “./$proj” --scenes 082 083 084 085 086 --dry_run
done

# 确认无误后修复（邻近帧补帧）
for proj in “blur投影” “depth投影” “HDMap投影”; do
    python fix_missing_gt.py --input_dir “./$proj” --scenes 082 083 084 085 086
done
```
