#!/bin/bash

# 监控的目录
WATCH_DIR="/root/autodl-tmp/tictactoe-gemini-think"

# 你的合并脚本路径
MERGE_SCRIPT="/root/RAGEN/merge_dl.sh"

echo "Starting directory monitor on $WATCH_DIR..."

# 监控 WATCH_DIR 下的目录创建事件
inotifywait -m -e create -q --format '%w%f' "$WATCH_DIR" | while read FILE
do
    # 检查新创建的文件是否是 global_step_* 目录
    if [[ "$FILE" =~ global_step_[0-9]+$ ]]; then
        echo "New global_step directory detected: $FILE"
        # 运行合并脚本
        /bin/bash "$MERGE_SCRIPT"
    fi
done
