# Caption 批量更新工具使用指南

## 📝 功能

批量更新 Transfer2 数据集中所有 caption JSON 文件的 caption 字段。

**特点：**
- 自动识别所有数据集（BlurDense, DepthSparse, HDMapBbox 等）
- 自动处理7个相机目录
- 支持模板变量：`{camera}`, `{scene}`, `{seg}`
- 预览更改后再应用
- 可选择单个或多个数据集

---

## 🚀 快速开始（推荐）

### 交互式模式

```bash
cd transfer_video_maker/caption一键修理
python3 update_captions.py
```

然后按照提示操作：

1. **选择数据集**
   - 显示所有可用数据集（BlurDense, DepthSparse, HDMapBbox, BlurProjection）
   - 可选择单个数据集或全部数据集

2. **选择 caption 模板**
   - 1-6：预设模板（depth, depth_dense, hdmap, blur, blur_dense, basic）
   - 7：自定义模板（输入自己的模板）

3. **预览更改**
   - 显示前5个文件的旧/新 caption 对比

4. **确认并执行**
   - 输入 `y` 确认更新

---

## 💡 使用示例

### 示例1：更新单个数据集（交互式）

```bash
python3 update_captions.py
```

```
找到 4 个数据集:
  1) BlurDense (28 个caption文件)
  2) BlurProjection (28 个caption文件)
  3) DepthSparse (28 个caption文件)
  4) HDMapBbox (20 个caption文件)
  5) 全部数据集
  0) 退出

请选择数据集 [1-5, 0]: 3

已选择: DepthSparse

预设caption模板:
  1) depth: "This is a depth map directly obtained from LiDAR points..."
  2) depth_dense: "This is a dense depth map generated from LiDAR points..."
  3) hdmap: "This is an HD map representation from an autonomous driving video..."
  4) blur: "This is a point cloud projection generated from LiDAR points..."
  5) blur_dense: "This is a denser point cloud projection created from LiDAR points..."
  6) basic: "This is a frame from an autonomous driving video..."
  7) 自定义模板

请选择模板 [1-7]: 1

预览更改（显示前 5 个）:
================================================================================

文件: DepthSparse/captions/ftheta_camera_front_tele_30fov/002_seg01.json
  旧: Scene 002 segment 1 from ftheta_camera_front_tele_30fov
  新: A depth map from ftheta_camera_front_tele_30fov

...

总计: 28 个文件将被更新
================================================================================

确认更新所有caption? [y/N]: y

正在更新...
✓ 完成! 成功更新 28/28 个文件
```

---

### 示例2：更新所有数据集（使用预设模板）

```bash
python3 update_captions.py
```

1. 选择 `5) 全部数据集`
2. 选择 `1-6` 中的任一预设模板（例如选择 `1` 使用 depth 模板）
3. 所有数据集将使用相同的模板更新

**注意：** 虽然选择了 depth 模板，但会应用到所有数据集（包括 BlurDense, HDMapBbox 等）。如需不同数据集使用不同模板，请分别运行多次。

---

### 示例3：使用自定义模板

```bash
python3 update_captions.py
```

选择数据集后，选择 `7) 自定义模板`，然后输入：

```
A driving scene from {camera} in scene {scene}
```

生成的 caption 示例：
```
A driving scene from ftheta_camera_front_tele_30fov in scene 002
```

---

## 🔧 命令行模式（高级）

### 更新单个数据集（使用预设模板）

```bash
python3 update_captions.py \
  --dataset DepthSparse \
  --preset depth
```

### 更新所有数据集（使用自定义模板）

```bash
python3 update_captions.py \
  --template "A depth map from {camera}" \
  --dry-run  # 先预览，不实际修改
```

去掉 `--dry-run` 后执行实际更新：

```bash
python3 update_captions.py \
  --template "A depth map from {camera}"
```

---

## 📊 模板变量说明

可用的模板变量：

| 变量 | 说明 | 示例值 |
|------|------|--------|
| `{camera}` | 相机名称（Transfer2格式） | `ftheta_camera_front_tele_30fov` |
| `{scene}` | 场景ID | `002` |
| `{seg}` | Segment ID | `seg01` |

**示例模板：**

```
A depth map from {camera}
→ A depth map from ftheta_camera_front_tele_30fov

Scene {scene} {seg} from {camera}
→ Scene 002 seg01 from ftheta_camera_front_tele_30fov

Driving scene {scene} captured by {camera}
→ Driving scene 002 captured by ftheta_camera_front_tele_30fov
```

---

## 🎨 预设模板列表（1-6）

工具内置了6个预设模板，每个对应不同的数据集类型：

| 模板序号 | 模板名 | Caption 内容（简要） |
|---------|--------|---------------------|
| 1 | `depth` | 从LiDAR点获得的稀疏深度图，来自自动驾驶视频 |
| 2 | `depth_dense` | 通过插值填充的稠密深度图，来自LiDAR点 |
| 3 | `hdmap` | 高精地图表示，包含交通灯、信号杆等城市元素 |
| 4 | `blur` | 路侧相机部分着色的稀疏点云投影 |
| 5 | `blur_dense` | 路侧相机着色的稠密点云投影 |
| 6 | `basic` | 自动驾驶视频帧，展示真实城市交叉路口场景 |

**注意：** 所有模板都包含 `{camera}` 变量，会自动替换为实际相机名称。完整模板内容较长，详细描述了场景特征（城市交叉路口、交通灯、车辆、绿化等）。

---

## 📂 支持的数据集结构

脚本自动识别以下结构的数据集：

```
output/
├── DepthSparse/
│   └── captions/
│       ├── ftheta_camera_front_tele_30fov/
│       │   ├── 002_seg01.json
│       │   ├── 002_seg02.json
│       │   └── ...
│       ├── ftheta_camera_front_wide_120fov/
│       └── ... (7个相机)
├── HDMapBbox/
│   └── captions/
│       └── ...
└── ...
```

---

## ✅ 操作检查清单

使用前确认：

- [ ] Python 3 已安装
- [ ] 数据集目录存在且包含 `captions/` 子目录
- [ ] 有足够的权限修改 JSON 文件

使用后检查：

- [ ] 所有文件成功更新（查看输出统计）
- [ ] 随机抽查几个 JSON 文件，确认 caption 正确
- [ ] 其他字段（scene_id, segment_id, camera）未被修改

---

## 🐛 常见问题

**Q: 如何只预览而不实际修改？**

交互式模式：在确认时输入 `n`

命令行模式：添加 `--dry-run` 参数

**Q: 可以批量更新多个数据集但使用不同的模板吗？**

不能。如需不同模板，请分别运行多次，每次选择不同的数据集。

**Q: 更新后如何恢复？**

脚本不创建备份。建议在使用前先备份数据集：

```bash
cp -r output output_backup
```

**Q: JSON 文件的其他字段会被修改吗？**

不会。脚本只修改 `caption` 字段，保留其他字段不变。

---

## 📝 Caption JSON 文件格式

**更新前：**
```json
{
  "scene_id": "002",
  "segment_id": "seg01",
  "camera": "ftheta_camera_front_tele_30fov",
  "caption": "Scene 002 segment 1 from ftheta_camera_front_tele_30fov"
}
```

**更新后：**
```json
{
  "scene_id": "002",
  "segment_id": "seg01",
  "camera": "ftheta_camera_front_tele_30fov",
  "caption": "A depth map from ftheta_camera_front_tele_30fov"
}
```

只有 `caption` 字段被修改，其他字段保持不变。

---

**创建日期：** 2025-11-20
**版本：** v1.0
