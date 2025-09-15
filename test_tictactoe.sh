USE_GRPO="algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True"

WANDB_MODE=offline RAY_DEDUP_LOGS=0\
    python train.py \
        --config-name _8_tictactoe \
        system.CUDA_VISIBLE_DEVICES=\"0,1\" \
        model_path="/root/autodl-tmp/Qwen2.5-1.5B-Instruct" \
        trainer.default_local_dir="/root/autodl-tmp/tictactoe" \
        trainer.total_training_steps=221 \
        trainer.n_gpus_per_node=2 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        trainer.experiment_name="tictactoe-grpo" \
        custom_envs.TicTacToe.env_config.player_info="[{model_name: 'deepseek'}]" \
        $USE_GRPO
        # 2>&1 | tee "$LOG_DIR/grpo_${TIMESTAMP}.log"