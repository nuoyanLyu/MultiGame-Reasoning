#!/bin/bash
# 测试模型是否能够正常vllm serve而后正常终止
START=180
END=300
STEP=20
ROOT_DIR="/root/autodl-tmp"
model_path="Qwen2.5-1.5B-Instruct"
for ((i=$START; i<=$END; i+=$STEP)); do
    echo "[INFO] Starting vLLM for $model_path ..."
    setsid env CUDA_VISIBLE_DEVICES=2 vllm serve "$ROOT_DIR/$model_path" \
        --port 4040 \
        --max-model-len 6000 \
        --host 0.0.0.0 \
        --gpu-memory-utilization 0.9 
    vllm_pid=$!
    pgid=$(ps -o pgid= -p "$vllm_pid" | tr -d ' ')

    echo "[INFO] Waiting for vLLM to load..."
    sleep 60
    # [4] 杀掉 vLLM serve，通过端口号和服务路径的方式精准清理
    echo "[INFO] Killing vLLM serve running on port 4040..."
    # 主要方法：通过端口号找到并杀死进程
    # lsof -t -i:4040 会返回监听该端口的进程PID
    VLLM_PIDS=$(lsof -t -i:4040)
    if [ -n "$VLLM_PIDS" ]; then
        echo "[INFO] Found vLLM processes with PIDs: $VLLM_PIDS. Terminating..."
        # 直接杀死这些PID，-9是强制信号
        kill -9 $VLLM_PIDS
        sleep 2 # 等待进程退出
    else
        echo "[WARN] No process found listening on port 4040."
    fi

    # 补充方法：通过启动命令的关键字进行更广泛的清理
    # 这可以捕获一些没有监听端口但相关的残留进程
    echo "[INFO] Performing secondary cleanup using pkill..."
    pkill -9 -f "vllm"
    pkill -9 -f "$model_path" # 使用模型路径作为独特的关键字，非常精确

    echo "[INFO] Cleanup complete."
    # check GPU memory确定GPU已经空出来了
    echo "[INFO] Checking GPU memory..."
    nvidia-smi
done