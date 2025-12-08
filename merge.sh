#!/bin/bash
ORI_FILE="math-nash-update"
TAR_FILE="mnu"
HF_PATH="/root/autodl-tmp/Qwen2.5-1.5B-Instruct"

for i in {50..300..50}
do
    ORI_PATH="/root/autodl-tmp/${ORI_FILE}/global_step_${i}/actor"
    TARGET_PATH="/root/autodl-tmp/${ORI_FILE}/${TAR_FILE}${i}"
    python verl/scripts/model_merger.py --backend fsdp --local_dir $ORI_PATH --target_dir $TARGET_PATH --hf_model_path $HF_PATH
done

echo "✅ All models merged successfully."