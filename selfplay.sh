#!/bin/bash
set -e

# Section 1: Base Experiments
USE_GRPO="algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True"
USE_PPO="algorithm.adv_estimator=gae" # by default.
USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"

TRAIN_ENV="tictactoe-gemini"
LOG_DIR="/root/RAGEN/logs/$TRAIN_ENV"
ROOT_DIR="/root/autodl-tmp"
BASE_MODEL="Qwen2.5-1.5B-Instruct"
MODEL_DIR="$ROOT_DIR/$TRAIN_ENV"
mkdir -p "$LOG_DIR" # 如果目录不存在，则创建它

START=100
END=200
STEP=20

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
    setsid env CUDA_VISIBLE_DEVICES=2 vllm serve "$ROOT_DIR/$model_path" \
        --port 4040 \
        --max-model-len 6000 \
        --host 0.0.0.0 \
        --gpu-memory-utilization 0.9 \
        > "$LOG_DIR/vllm_monitor.log" 2>&1 &
    vllm_pid=$!
    pgid=$(ps -o pgid= -p "$vllm_pid" | tr -d ' ')

    echo "[INFO] Waiting for vLLM to load..."
    sleep 60

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
        --target_dir "$MODEL_DIR/game_$((i + STEP))"

    # [4] 杀掉 vLLM serve
    # [4] 杀掉 vLLM serve，通过进程组的方式打包送走
    echo "[INFO] Killing vLLM serve (PID $vllm_pid, PGID $pgid)"
    kill -9 -"$pgid" || true
    sleep 2
    pkill -9 -f "vllm" || true
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