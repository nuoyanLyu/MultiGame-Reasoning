#!/bin/bash
ORI_FILE="undercover"
TAR_FILE="undercover"
HF_PATH="/root/autodl-tmp/Qwen2.5-1.5B-Instruct"

for i in {50..100..50}
do
    ORI_PATH="/root/autodl-tmp/${ORI_FILE}/global_step_${i}/actor"
    TARGET_PATH="/root/autodl-tmp/${ORI_FILE}/${TAR_FILE}${i}"
    python verl/scripts/model_merger.py --backend fsdp --local_dir $ORI_PATH --target_dir $TARGET_PATH --hf_model_path $HF_PATH
done

echo "✅ All models merged successfully."