## 主要流程： 启动环境conda activate zihanw ->控制帧图像生成方法 -> 控制头视频生成方法 -> caption替换方法
## 控制帧图像生成方法

指令： ./run_interactive.sh
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
4. 填入场景，直接填001-089的编号就可以，他自己可以找到对应的地址，运行配置直接enter使用默认
5. 可以选择处理多少数据，例如中间90帧（过路口帧），所有帧等
6. 但是注意，他们跑完除了HDMap其余的任务之后，还需要你填写HDMap的运行config，这次会多填写一个自车ID，直接选择auto就可以了。
7. 生成的是逐帧的图片，分别在自己的子功能文件夹下，按场景分类的。每一帧都可以验证质量。

## 控制头视频生成方法
这步的时候一开始要选7
1. 脚本在这里 ./transfer_video_maker/generate_videos.sh
2. 一开始的菜单同上，一般来讲直接选Batch就可以，然后选择自己想要的功能。
3. 关于每个seg的帧数，目前的模型接受的是29帧，所以每个seg帧数要输入29帧，不要选默认值。seg的数量（这取决于有多少帧，我们可以把它切段切成几个29帧，就输入几。例如在上一步第5点，比如选择产了90帧数据，那那么就可以产出来3个29帧的seg，seg数量输出为“3”）和视频帧率这些默认值是符合cosmos post training demo的配置，默认1280 720p，生成在./transfer_video_maker/output（！记得修改这个绝对路径），主要分类依据不是场景号，是根据transfer控制头分类的。
4. 生成完视频后需要把之前blur投影，depth投影，hdmap投影当中的帧数据删掉，占用空间过大。

## caption替换方法
1. 执行脚本 ./transfer_video_maker/caption一键修理/update_captions.py
数据集基础目录要修改，我写的绝对路径。

2. 如果替换多个头，直接选全部数据集，然后选自动匹配
3. 想替换预设caption（文本引导）直接改py脚本里面的txt就可以了

！ 最终用于训练模型的引到就是 ./transfer_video_maker/output中的内容


### 额外补充
### 因为有一些clip有问题，写指令看看哪个里面缺少GT，然后确定后再次检查。这是数据集里某些clip也许会出现掉帧的情况，如果发现有掉帧，检查了videos.mp4不足29帧，就需要用邻近帧补帧
for clip in 082 083 084 085 086; do
  for d in ./blur投影/$clip/$clip/[0-9]*/; do name=$(basename "$d"); gt=$(ls "$d/gt" 2>/dev/null | wc -l); proj=$(ls "$d/proj" 2>/dev/null | wc -l); if ! { [ "$gt" -eq 7 ] && { [ "$proj" -eq 7 ] || [ "$proj" -eq 14 ]; }; }; then echo "blur/$clip/$name: gt=$gt, proj=$proj"; fi; done
  for d in ./depth投影/$clip/$clip/[0-9]*/; do name=$(basename "$d"); gt=$(ls "$d/gt" 2>/dev/null | wc -l); depth=$(ls "$d/depth" 2>/dev/null | wc -l); if ! { [ "$gt" -eq 7 ] && { [ "$depth" -eq 7 ] || [ "$depth" -eq 14 ]; }; }; then echo "depth/$clip/$name: gt=$gt, depth=$depth"; fi; done
  for d in ./HDMap投影/$clip/$clip/[0-9]*/; do name=$(basename "$d"); gt=$(ls "$d/gt" 2>/dev/null | wc -l); overlay=$(ls "$d/overlay" 2>/dev/null | wc -l); if ! { [ "$gt" -eq 7 ] && { [ "$overlay" -eq 7 ] || [ "$overlay" -eq 14 ]; }; }; then echo "HDMap/$clip/$name: gt=$gt, overlay=$overlay"; fi; done
done

有一些clip缺部分视角真值，补齐的方法：
修复步骤
1. 先 dry_run 检查缺失情况
cd ./transfer_video_maker

### 检查 blur投影
python fix_missing_gt.py \
    --input_dir "./blur投影" \
    --scenes 082 083 084 085 086 \
    --dry_run

### 检查 depth投影
python fix_missing_gt.py \
    --input_dir "./depth投影" \
    --scenes 082 083 084 085 086 \
    --dry_run

### 检查 HDMap投影
python fix_missing_gt.py \
    --input_dir "./HDMap投影" \
    --scenes 082 083 084 085 086 \
    --dry_run

2. 确认后执行实际修复
### 修复 blur投影
python fix_missing_gt.py \
    --input_dir "./blur投影" \
    --scenes 082 083 084 085 086

### 修复 depth投影
python fix_missing_gt.py \
    --input_dir "./depth投影" \
    --scenes 082 083 084 085 086

### 修复 HDMap投影
python fix_missing_gt.py \
    --input_dir "./HDMap投影" \
    --scenes 082 083 084 085 086

或者一条命令搞定所有（修复模式）
cd ./transfer_video_maker

for proj in "blur投影" "depth投影" "HDMap投影"; do
    echo "========== 修复 $proj =========="
    python fix_missing_gt.py \
        --input_dir "./$proj" \
        --scenes 082 083 084 085 086
done

### 仅限于学习交流
