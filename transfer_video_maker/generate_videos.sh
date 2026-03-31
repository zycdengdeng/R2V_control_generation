#!/bin/bash
# Transfer2 视频生成快速启动脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
PROJECT_ROOT="/mnt/zihanw/proj_utils_pro/transfer_video_maker"
OUTPUT_BASE="/mnt/zihanw/proj_utils_pro/transfer_video_maker/output"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Transfer2 视频数据集生成器${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 显示菜单
show_menu() {
    echo -e "${GREEN}请选择项目类型:${NC}"
    echo "  1) depth          - 深度图投影"
    echo "  2) depth_dense    - 深度图稠密化投影"
    echo "  3) hdmap          - HDMap bbox投影"
    echo "  4) blur           - Blur投影（路侧着色）"
    echo "  5) blur_dense     - Blur稠密化投影"
    echo "  6) basic          - 基本点云投影"
    echo ""
    echo "  7) batch          - 批量生成（选择多个项目类型）"
    echo ""
    echo "  0) 自定义命令"
    echo "  q) 退出"
    echo ""
    read -p "请选择 [1-7, 0, q]: " choice
}

# 获取用户输入
get_user_input() {
    echo ""
    echo -e "${GREEN}配置参数:${NC}"

    # 场景ID
    read -p "场景ID列表（空格分隔，如 002 004 006）: " scenes
    if [ -z "$scenes" ]; then
        echo -e "${RED}错误: 场景ID不能为空${NC}"
        exit 1
    fi

    # 每个seg的帧数
    read -p "每个seg的帧数 [默认: 21]: " frames_per_seg
    frames_per_seg=${frames_per_seg:-21}

    # seg数量
    read -p "每个场景的seg数量 [默认: 4]: " num_segs
    num_segs=${num_segs:-4}

    # 帧率
    read -p "视频帧率 [默认: 10]: " fps
    fps=${fps:-10}

    # 输出目录名
    read -p "输出目录名 [默认: ${project_name}]: " output_name
    output_name=${output_name:-$project_name}

    OUTPUT_DIR="${OUTPUT_BASE}/${output_name}"
}

# 执行命令
run_command() {
    local project_type=$1
    local control_subdir=$2
    local control_type=$3
    local caption=$4
    local auto_confirm=${5:-false}  # 第5个参数：是否自动确认（批量模式用）

    echo ""
    echo -e "${YELLOW}执行命令...${NC}"
    echo ""

    cmd="python3 ${PROJECT_ROOT}/generate_transfer2_videos.py \
  --project-type ${project_type} \
  --scenes ${scenes} \
  --frames-per-seg ${frames_per_seg} \
  --num-segs ${num_segs} \
  --fps ${fps} \
  --output-dir ${OUTPUT_DIR} \
  --control-subdir ${control_subdir} \
  --control-input-type ${control_type}"

    if [ -n "$caption" ]; then
        cmd="${cmd} --caption-template \"${caption}\""
    fi

    echo -e "${BLUE}${cmd}${NC}"
    echo ""

    # 批量模式自动确认，单个模式需要用户确认
    if [ "$auto_confirm" = true ]; then
        confirm="y"
        echo "批量模式：自动执行"
    else
        confirm="y"
    fi

    if [[ $confirm == [yY] ]]; then
        eval $cmd

        if [ $? -eq 0 ]; then
            echo ""
            echo -e "${GREEN}========================================${NC}"
            echo -e "${GREEN}  生成完成！${NC}"
            echo -e "${GREEN}========================================${NC}"
            echo -e "输出目录: ${OUTPUT_DIR}"
            echo ""
            echo -e "生成内容:"
            echo -e "  ✓ videos/                  (GT去畸变视频)"
            echo -e "  ✓ control_input_${control_type}/ (控制输入视频)"
            echo -e "  ✓ captions/                (caption JSON)"
            return 0
        else
            echo -e "${RED}执行失败${NC}"
            if [ "$auto_confirm" = true ]; then
                echo -e "${RED}批量模式：继续处理下一个项目${NC}"
                return 1
            else
                exit 1
            fi
        fi
    else
        echo "已取消"
        exit 0
    fi
}

# 主流程
show_menu

case $choice in
    1)
        project_name="DepthSparse"
        get_user_input
        run_command "depth" "depth" "depth" "A depth map from {camera}"
        ;;
    2)
        project_name="DepthDense"
        get_user_input
        run_command "depth_dense" "depth" "depth_dense" "A dense depth map from {camera}"
        ;;
    3)
        project_name="HDMapBbox"
        get_user_input
        run_command "hdmap" "overlay" "hdmap_bbox" "High-definition map with bounding boxes from {camera}"
        ;;
    4)
        project_name="BlurProjection"
        get_user_input
        run_command "blur" "proj" "blur" "A roadside-colored point cloud projection from {camera}"
        ;;
    5)
        project_name="BlurDense"
        get_user_input
        run_command "blur_dense" "proj" "blur_dense" "A dense roadside-colored projection from {camera}"
        ;;
    6)
        project_name="BasicProjection"
        get_user_input
        run_command "basic" "proj" "basic" "A basic point cloud projection from {camera}"
        ;;
    7)
        # 批量生成模式
        echo ""
        echo -e "${BLUE}========================================${NC}"
        echo -e "${BLUE}  批量生成模式${NC}"
        echo -e "${BLUE}========================================${NC}"
        echo ""
        echo "可选项目类型："
        echo "  1) depth          - 深度图投影"
        echo "  2) depth_dense    - 深度图稠密化投影"
        echo "  3) hdmap          - HDMap bbox投影"
        echo "  4) blur           - Blur投影"
        echo "  5) blur_dense     - Blur稠密化投影"
        echo "  6) basic          - 基本点云投影"
        echo ""
        read -p "选择要生成的项目（用逗号或空格分隔，如 1,2,3 或 1 2 3）: " batch_choices

        # 转换逗号为空格
        batch_choices=${batch_choices//,/ }

        # 获取通用参数（只需要输入一次）
        get_user_input

        # 保存用户输入的参数
        saved_scenes="$scenes"
        saved_frames_per_seg="$frames_per_seg"
        saved_num_segs="$num_segs"
        saved_fps="$fps"

        # 处理每个选择的项目类型
        for batch_choice in $batch_choices; do
            # 恢复参数
            scenes="$saved_scenes"
            frames_per_seg="$saved_frames_per_seg"
            num_segs="$saved_num_segs"
            fps="$saved_fps"

            echo ""
            echo -e "${YELLOW}========================================${NC}"
            case $batch_choice in
                1)
                    echo -e "${YELLOW}处理: depth (深度图投影)${NC}"
                    project_name="DepthSparse"
                    OUTPUT_DIR="${OUTPUT_BASE}/${project_name}"
                    run_command "depth" "depth" "depth" "A depth map from {camera}"
                    ;;
                2)
                    echo -e "${YELLOW}处理: depth_dense (深度图稠密化投影)${NC}"
                    project_name="DepthDense"
                    OUTPUT_DIR="${OUTPUT_BASE}/${project_name}"
                    run_command "depth_dense" "depth" "depth_dense" "A dense depth map from {camera}"
                    ;;
                3)
                    echo -e "${YELLOW}处理: hdmap (HDMap bbox投影)${NC}"
                    project_name="HDMapBbox"
                    OUTPUT_DIR="${OUTPUT_BASE}/${project_name}"
                    run_command "hdmap" "overlay" "hdmap_bbox" "High-definition map with bounding boxes from {camera}"
                    ;;
                4)
                    echo -e "${YELLOW}处理: blur (Blur投影)${NC}"
                    project_name="BlurProjection"
                    OUTPUT_DIR="${OUTPUT_BASE}/${project_name}"
                    run_command "blur" "proj" "blur" "A roadside-colored point cloud projection from {camera}"
                    ;;
                5)
                    echo -e "${YELLOW}处理: blur_dense (Blur稠密化投影)${NC}"
                    project_name="BlurDense"
                    OUTPUT_DIR="${OUTPUT_BASE}/${project_name}"
                    run_command "blur_dense" "proj" "blur_dense" "A dense roadside-colored projection from {camera}"
                    ;;
                6)
                    echo -e "${YELLOW}处理: basic (基本点云投影)${NC}"
                    project_name="BasicProjection"
                    OUTPUT_DIR="${OUTPUT_BASE}/${project_name}"
                    run_command "basic" "proj" "basic" "A basic point cloud projection from {camera}"
                    ;;
                *)
                    echo -e "${RED}跳过无效选择: $batch_choice${NC}"
                    ;;
            esac
            echo -e "${YELLOW}========================================${NC}"
        done

        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}  批量生成完成！${NC}"
        echo -e "${GREEN}========================================${NC}"
        ;;
    0)
        echo ""
        echo -e "${YELLOW}自定义模式${NC}"
        echo "请直接运行以下命令:"
        echo ""
        echo "python3 ${PROJECT_ROOT}/generate_transfer2_videos.py \\"
        echo "  --project-type <type> \\"
        echo "  --scenes <scene_ids> \\"
        echo "  --frames-per-seg <num> \\"
        echo "  --num-segs <num> \\"
        echo "  --output-dir <output_dir> \\"
        echo "  --control-subdir <subdir> \\"
        echo "  --control-input-type <type>"
        echo ""
        echo "查看完整文档: cat TRANSFER2_VIDEO_GUIDE.md"
        ;;
    q|Q)
        echo "退出"
        exit 0
        ;;
    *)
        echo -e "${RED}无效选择${NC}"
        exit 1
        ;;
esac
