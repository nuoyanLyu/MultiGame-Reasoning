set -e

# Section 1: Base Experiments
USE_GRPO="algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True"
USE_PPO="algorithm.adv_estimator=gae" # by default.
USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"

LOCAL_PATH="/root/autodl-tmp/"

# WANDB_MODE=offline python train.py --config-name _1_bandit system.CUDA_VISIBLE_DEVICES=\"0,4\" trainer.n_gpus_per_node=2 actor_rollout_ref.rollout.tensor_model_parallel_size=2 trainer.experiment_name=bandit-ppo $USE_PPO &
WANDB_MODE=offline python train.py --config-name _7_connect4 system.CUDA_VISIBLE_DEVICES=\"0,1\" model_path=/root/autodl-tmp/Qwen3-1.7B trainer.default_local_dir=$LOCAL_PATH trainer.total_training_steps=200 trainer.n_gpus_per_node=2 actor_rollout_ref.rollout.tensor_model_parallel_size=2 trainer.experiment_name=connect4-grpo $USE_GRPO &
# WANDB_MODE=offline python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES=\"3,4\" trainer.n_gpus_per_node=2 actor_rollout_ref.rollout.tensor_model_parallel_size=2 trainer.experiment_name=sokoban-grpo actor_rollout_ref.rollout.rollout_filter_ratio=0.25 $USE_GRPO &
