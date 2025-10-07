#!/bin/bash

# --- 用户配置部分 ---
# 模型检查点所在的根目录
ROOT_DIR="/root/autodl-tmp/tictactoe-math"
# 合并后模型保存的根目录
TAR_ROOT_DIR="/root/autodl-tmp/tictactoe-math"
# Hugging Face 模型的路径
HF_PATH="/root/autodl-tmp/Qwen2.5-1.5B-Instruct"
# --- 用户配置部分结束 ---


# --- 脚本逻辑部分 ---
# 获取所有 global_step_* 目录，并按修改时间排序
# -t：按修改时间排序
# -r：倒序排列，最新修改的在最前面
# head -n 2：取前两个，即最新和倒数第二个
# tail -n 1：从前两个中取最后一个，即倒数第二个
OLD_MODEL_DIR=$(ls -d -t "$ROOT_DIR"/global_step_* 2>/dev/null | head -n 2 | tail -n 1)

# 检查是否成功获取到目录
if [ -z "$OLD_MODEL_DIR" ]; then
    echo "Warning: Could not find the second newest global_step directory. Exiting."
    exit 0
fi

# 从目录名中提取 global_step 数字（如 global_step_40 -> 40）
STEP_NUMBER=$(basename "$OLD_MODEL_DIR" | sed 's/global_step_//')

# 设置 ORI_PATH 和 TAR_PATH
ORI_PATH="$OLD_MODEL_DIR/actor"
TAR_PATH="$TAR_ROOT_DIR/game$STEP_NUMBER"

# 打印将要使用的路径，方便调试
echo "-----------------------------------"
echo "Found old model dir: $OLD_MODEL_DIR"
echo "Step number: $STEP_NUMBER"
echo "Original path: $ORI_PATH"
echo "Target path:   $TAR_PATH"
echo "HF model path: $HF_PATH"
echo "-----------------------------------"

# 执行 Python 合并脚本
python verl/scripts/model_merger.py --backend fsdp --local_dir "$ORI_PATH" --target_dir "$TAR_PATH" --hf_model_path "$HF_PATH"

# 检查 Python 脚本是否成功执行
if [ $? -eq 0 ]; then
    echo "Python script finished successfully. Deleting old model directory."
    # 删除旧目录
    rm -r "$OLD_MODEL_DIR"
    echo "Directory $OLD_MODEL_DIR has been removed."
else
    echo "Python script failed. Old directory will not be removed."
fi