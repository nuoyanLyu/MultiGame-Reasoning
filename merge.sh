ORI_PATH="/root/autodl-tmp/tictactoe/global_step_20/actor"
TAR_PATH="/root/autodl-tmp/tictactoe/grpo/game_20"
HF_PATH="/root/autodl-tmp/Qwen2.5-1.5B-Instruct"

python verl/scripts/model_merger.py --backend fsdp --local_dir $ORI_PATH --target_dir $TAR_PATH --hf_model_path $HF_PATH
