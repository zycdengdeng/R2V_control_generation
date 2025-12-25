### 这里是专门给老年痴呆写的readme，老奶奶看了都用的明白

## 主要流程： 启动环境conda activate zihanw ->控制帧图像生成方法 -> 控制头视频生成方法 -> caption替换方法
## 控制帧图像生成方法

/mnt/zihanw/proj_utils_pro/run_interactive.sh
可用的投影项目:
  1) basic        - 基本路侧merge激光雷达点云投影
  2) blur         - blur投影（路侧相机着色merge激光雷达点云投影）
  3) blur_dense   - blur稠密化投影（路端merge点云投影完规则稠密化）
  4) depth        - depth投影（merge点云生成深度）
  5) depth_dense  - depth稠密化投影（merge点云生成深度后规则稠密化）
  6) hdmap        - HDMap投影（3D→2D bbox）

  7) batch        - 批量处理（选择多个项目串行执行）

  注意输入：选项里有auto,直接输入auto
  如果是线程数 gpu 直接enter，带有默认....直接enter

1. 在任意的目录下都可以执行主脚本，直接执行绝对路径，这得益于主要的 .py 和 .sh 脚本都使用 Path(__file__).resolve().parent，自动解析绝对路径和父目录
2. 运行我们的主脚本 run_interactive.sh，会在terminal给很多选项。 1-6 是某个功能的实现指令。 7 是批量处理多个指令，一般来讲选7就可以。
3. 选了 7 之后就可以开始多选功能，如果你需要 blur + depth + HDMap 那就选择 2 4 6. 
4. 填入场景，直接填001-089的编号就可以，他自己可以找到对应的地址，运行配置直接enter使用默认，跑就行了
5. 但是注意，他们跑完除了HDMap其余的任务之后，还需要你填写HDMap的运行config，这次会多填写一个自车ID，直接选择auto就可以了。
6. 生成的是逐帧的图片，分别在自己的子功能文件夹下，按场景分类的。每一帧都可以验证质量。

## 控制头视频生成方法
这步的时候一开始要选7
1. 脚本在这里 /mnt/zihanw/proj_utils_pro/transfer_video_maker/generate_videos.sh
2. 一开始的菜单同上，一般来讲直接选Batch就可以，然后选择自己想要的功能。
3. 关于每个seg的帧数，seg的数量和视频帧率这些默认值是符合cosmos post training demo的配置，默认1280 720p，生成在/mnt/zihanw/proj_utils_pro/transfer_video_maker/output，主要分类依据不是场景号，是根据transfer控制头分类的。

## 因为有一些clip有问题，写指令看看哪个里面缺少GT，然后确定后再次检查
for clip in 082 083 084 085 086; do
  for d in /mnt/zihanw/proj_utils_pro/blur投影/$clip/$clip/[0-9]*/; do name=$(basename "$d"); gt=$(ls "$d/gt" 2>/dev/null | wc -l); proj=$(ls "$d/proj" 2>/dev/null | wc -l); if ! { [ "$gt" -eq 7 ] && { [ "$proj" -eq 7 ] || [ "$proj" -eq 14 ]; }; }; then echo "blur/$clip/$name: gt=$gt, proj=$proj"; fi; done
  for d in /mnt/zihanw/proj_utils_pro/depth投影/$clip/$clip/[0-9]*/; do name=$(basename "$d"); gt=$(ls "$d/gt" 2>/dev/null | wc -l); depth=$(ls "$d/depth" 2>/dev/null | wc -l); if ! { [ "$gt" -eq 7 ] && { [ "$depth" -eq 7 ] || [ "$depth" -eq 14 ]; }; }; then echo "depth/$clip/$name: gt=$gt, depth=$depth"; fi; done
  for d in /mnt/zihanw/proj_utils_pro/HDMap投影/$clip/$clip/[0-9]*/; do name=$(basename "$d"); gt=$(ls "$d/gt" 2>/dev/null | wc -l); overlay=$(ls "$d/overlay" 2>/dev/null | wc -l); if ! { [ "$gt" -eq 7 ] && { [ "$overlay" -eq 7 ] || [ "$overlay" -eq 14 ]; }; }; then echo "HDMap/$clip/$name: gt=$gt, overlay=$overlay"; fi; done
done

有一些clip缺部分视角真值，补齐的方法：
修复步骤
1. 先 dry_run 检查缺失情况
cd /mnt/zihanw/proj_utils_pro/transfer_video_maker

# 检查 blur投影
python fix_missing_gt.py \
    --input_dir "/mnt/zihanw/proj_utils_pro/blur投影" \
    --scenes 082 083 084 085 086 \
    --dry_run

# 检查 depth投影
python fix_missing_gt.py \
    --input_dir "/mnt/zihanw/proj_utils_pro/depth投影" \
    --scenes 082 083 084 085 086 \
    --dry_run

# 检查 HDMap投影
python fix_missing_gt.py \
    --input_dir "/mnt/zihanw/proj_utils_pro/HDMap投影" \
    --scenes 082 083 084 085 086 \
    --dry_run

2. 确认后执行实际修复
# 修复 blur投影
python fix_missing_gt.py \
    --input_dir "/mnt/zihanw/proj_utils_pro/blur投影" \
    --scenes 082 083 084 085 086

# 修复 depth投影
python fix_missing_gt.py \
    --input_dir "/mnt/zihanw/proj_utils_pro/depth投影" \
    --scenes 082 083 084 085 086

# 修复 HDMap投影
python fix_missing_gt.py \
    --input_dir "/mnt/zihanw/proj_utils_pro/HDMap投影" \
    --scenes 082 083 084 085 086

或者一条命令搞定所有（修复模式）
cd /mnt/zihanw/proj_utils_pro/transfer_video_maker

for proj in "blur投影" "depth投影" "HDMap投影"; do
    echo "========== 修复 $proj =========="
    python fix_missing_gt.py \
        --input_dir "/mnt/zihanw/proj_utils_pro/$proj" \
        --scenes 082 083 084 085 086
done

## caption替换方法
1. 执行脚本 /mnt/zihanw/proj_utils_pro/transfer_video_maker/caption一键修理/update_captions.py
2. 如果是替换单个头，就选单个，然后选择适应这个头的选项，每个选项都说明自己是哪个控制头的caption
3. 如果替换多个头，直接选替换全部，然后会多一个自适应匹配的选项，选那个就可以
4. 想替换预设caption直接改py脚本里面的txt就可以了

### 仅限于学习交流，商业用途可以给我磕头笑鼠
