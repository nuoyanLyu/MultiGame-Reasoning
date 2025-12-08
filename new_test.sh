#!/bin/bash

# test new model
MODEL_PATH="math-nash-update"

for i in {50..300..50}
do
    MODEL_NAME="mnu${i}"
    echo "===== Testing model: $MODEL_NAME ====="
    # game test
    python reason_test/nash-new.py --model_path $MODEL_PATH --model_name $MODEL_NAME
    python reason_test/tictactoe.py --model_path $MODEL_PATH --model_name $MODEL_NAME
    python reason_test/undercover.py --model_path $MODEL_PATH --model_name $MODEL_NAME
    # mmlu test
    python reason_test/mmlu_dl.py --model_path $MODEL_PATH --model_name $MODEL_NAME
    python reason_test/mmlu_pro_dl.py --model_path $MODEL_PATH --model_name $MODEL_NAME
    # # math test
    python reason_test/math_dl.py --model_path $MODEL_PATH --model_name $MODEL_NAME
    python reason_test/lv3to5_dl.py --model_path $MODEL_PATH --model_name $MODEL_NAME
    # social test
    python reason_test/social.py --model_path $MODEL_PATH --model_name $MODEL_NAME
    # common test
    python reason_test/common.py --model_path $MODEL_PATH --model_name $MODEL_NAME
done

echo "✅ All models tested successfully."