#!/bin/bash
set -e

# Section 1: Base Experiments
USE_GRPO="algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True"
USE_PPO="algorithm.adv_estimator=gae" # by default.
USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"

TRAIN_ENV="tictactoe-selfplay"
LOG_DIR="/root/RAGEN/logs/$TRAIN_ENV"
ROOT_DIR="/root/autodl-tmp"
BASE_MODEL="Qwen2.5-1.5B-Instruct"
MODEL_DIR="$ROOT_DIR/$TRAIN_ENV"
mkdir -p "$LOG_DIR" # 如果目录不存在，则创建它

START=100
END=300
STEP=50

# 对手模型的GPU id以及对应的vllm serve端口
GPU_DEVICE_ID=2
VLLM_PORT=4040

set -e

for ((i=$START; i<=$END; i+=$STEP)); do
    echo "===== Round $i ====="
    # 获取当前时间，格式为 YYYY-MM-DD-HHMMSS
    # 每一轮都获取 这样日志不会被覆盖
    TIMESTAMP=$(date +"%m-%d-%H-%M")

    if [[ $i -eq 0 ]]; then
        # 设置的是纯self-play，最开始其实也可以通过调用API的方式进行
        model_name="$BASE_MODEL"
        model_path="$model_name"
    else
        model_name="game${i}"
        model_path="$TRAIN_ENV/$model_name"
    fi

    # [1] 启动 vLLM serve
    echo "[INFO] Starting vLLM for $model_path ..."
    setsid env CUDA_VISIBLE_DEVICES=$GPU_DEVICE_ID vllm serve "$ROOT_DIR/$model_path" \
        --port $VLLM_PORT \
        --max-model-len 6000 \
        --host 0.0.0.0 \
        --gpu-memory-utilization 0.9 \
        > "$LOG_DIR/vllm_monitor.log" 2>&1 &
    vllm_launcher_pid=$!
    # pgid=$(ps -o pgid= -p "$vllm_pid" | tr -d ' ')

    echo "[INFO] Waiting for vLLM to load..."
    sleep 60

    echo "[INFO] Model should be loaded. Current status on GPU $GPU_DEVICE_ID:"
    nvidia-smi -i $GPU_DEVICE_ID

    # [2] 运行训练
    player_info="{model_name: '${model_path}', port: '4040'}"
    echo "[INFO] Set Env Player info $player_info"
    WANDB_MODE=offline \
    python train.py \
        --config-name _8_tictactoe \
        system.CUDA_VISIBLE_DEVICES=\"0,1\" \
        model_path="$ROOT_DIR/$BASE_MODEL" \
        trainer.default_local_dir="$ROOT_DIR/$TRAIN_ENV" \
        trainer.total_training_steps=$((i + STEP)) \
        trainer.n_gpus_per_node=2 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        trainer.experiment_name="${TRAIN_ENV}-grpo" \
        custom_envs.TicTacToe.env_config.player_info="[${player_info}]" \
        $USE_GRPO \
        2>&1 | tee "$LOG_DIR/grpo_${TIMESTAMP}.log"

    # [3] 模型转换
    echo "[INFO] Converting model..."
    python verl/scripts/model_merger.py \
        --backend fsdp \
        --hf_model_path "$ROOT_DIR/$BASE_MODEL" \
        --local_dir "$ROOT_DIR/$TRAIN_ENV/global_step_$((i + STEP))/actor" \
        --target_dir "$MODEL_DIR/game$((i + STEP))"

    # [4] 杀掉 vLLM serve，通过端口号和服务路径的方式精准清理
    echo "[INFO] Killing vLLM serve running on port 4040..."
    # --- [3] 杀掉 vLLM Serve，基于 nvidia-smi 的精准方案 ---
    echo "[INFO] Starting cleanup: Terminating all processes on GPU $GPU_DEVICE_ID..."

    # 使用 nvidia-smi 查询指定GPU上的所有计算进程的PID
    # --query-compute-apps=pid: 只查询计算应用的PID
    # --format=csv,noheader: 以CSV格式输出，且不带标题行，方便脚本处理
    # -i $GPU_DEVICE_ID: 指定要查询的GPU ID
    PIDS_ON_GPU=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $GPU_DEVICE_ID)

    if [ -n "$PIDS_ON_GPU" ]; then
        echo "[INFO] Found the following PIDs on GPU $GPU_DEVICE_ID: $PIDS_ON_GPU"
        # 直接将PID列表传递给kill命令，-9表示强制终止
        kill -9 $PIDS_ON_GPU
        echo "[INFO] Kill signal sent to all found PIDs."
        sleep 5 # 等待5秒，让操作系统有时间回收进程
    else
        echo "[INFO] No running compute processes found on GPU $GPU_DEVICE_ID."
    fi

    # --- [4] 验证清理结果 ---
    echo "[INFO] Verifying GPU status after cleanup..."
    # 再次检查GPU状态，理想情况下“Processes”部分应该为空
    nvidia-smi -i $GPU_DEVICE_ID

    # 作为最后的保险措施，可以再用pkill清理一下，以防万一
    echo "[INFO] Performing final secondary cleanup with pkill..."
    pkill -9 -f "$model_path" || true # 使用模型路径作为独特的关键字，非常精确

    echo "[INFO] Cleanup complete."
    # check GPU memory确定GPU已经空出来了
    echo "[INFO] Checking GPU memory..."
    nvidia-smi

    # 删除旧模型目录（如果存在）
    MODEL_BEFORE="$ROOT_DIR/$TRAIN_ENV/global_step_${i}"
    if [[ -d "$MODEL_BEFORE" ]]; then
        echo "[INFO] Removing old model $MODEL_BEFORE"
        rm -rf "$MODEL_BEFORE"
    fi

    echo "===== Round $i Done ====="
    echo
done