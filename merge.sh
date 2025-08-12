ORI_PATH="/root/autodl-tmp/global_step_20/actor"
TAR_PATH="/root/autodl-tmp/grpo/game_20"
HF_PATH="/root/autodl-tmp/Qwen3-1.7B"

python verl/scripts/model_merger.py --backend fsdp --local_dir $ORI_PATH --target_dir $TAR_PATH --hf_model_path $HF_PATH