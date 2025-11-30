#!/bin/bash

# test new model
MODEL_PATH="nash-math"

for i in {50..200..50}
do
    MODEL_NAME="nm${i}"
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

done

echo "✅ All models tested successfully."