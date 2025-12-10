set -e

# Section 1: Base Experiments
USE_GRPO="algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True"
USE_PPO="algorithm.adv_estimator=gae" # by default.
USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"

LOCAL_PATH="/root/autodl-tmp/mnt-mix"
LOG_DIR="/root/RAGEN/logs/mnt-mix"
MODEL_PATH="/root/autodl-tmp/mnt-mix/mix50"
mkdir -p "$LOG_DIR" # 如果目录不存在，则创建它

# 获取当前时间，格式为 YYYY-MM-DD-HHMMSS
TIMESTAMP=$(date +"%m-%d-%H-%M")

WANDB_MODE=offline RAY_DEDUP_LOGS=0 python train.py --config-name _19_MNT_mix \
 system.CUDA_VISIBLE_DEVICES=\"0,1\" \
 model_path=$MODEL_PATH \
 trainer.default_local_dir=$LOCAL_PATH \
 trainer.total_training_steps=150 \
 trainer.n_gpus_per_node=2 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
 trainer.experiment_name=MNT-mix \
 $USE_GRPO 2>&1 | tee "$LOG_DIR/grpo_${TIMESTAMP}.log" &
# WANDB_MODE=offline python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES=\"3,4\" trainer.n_gpus_per_node=2 actor_rollout_ref.rollout.tensor_model_parallel_size=2 trainer.experiment_name=sokoban-grpo actor_rollout_ref.rollout.rollout_filter_ratio=0.25 $USE_GRPO &
