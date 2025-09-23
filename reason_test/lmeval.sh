#!/usr/bin/env bash

# 清空代理环境变量
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

# 使用lm-eval测试模型，传递模型路径等参数
lm-eval --model local-completions \
    --model_args pretrained=/data1/lvnuoyan/llm_model/game100,base_url=http://localhost:2121/v1/completions \
    --batch_size 16 \
    --gen_kwargs "max_tokens=2000,temperature=0.5" \
    --tasks mmlu_pro --apply_chat_template \
    --output_path "lm-eval/" \
    --system_instruction You're a helpful assistant. 