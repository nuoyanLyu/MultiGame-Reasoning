# test new model
MODEL_PATH="tictactoe-mix"
MODEL_NAME="tictactoe_mix150"

# game test
python reason_test/nash.py --model_path $MODEL_PATH --model_name $MODEL_NAME
python reason_test/tictactoe.py --model_path $MODEL_PATH --model_name $MODEL_NAME

# mmlu test
python reason_test/mmlu_dl.py --model_path $MODEL_PATH --model_name $MODEL_NAME
python reason_test/mmlu_pro_dl.py --model_path $MODEL_PATH --model_name $MODEL_NAME
# math test
python reason_test/math_dl.py --model_path $MODEL_PATH --model_name $MODEL_NAME
python reason_test/lv3to5_dl.py --model_path $MODEL_PATH --model_name $MODEL_NAME

