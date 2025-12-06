# 参考tictactoe测试代码生成undercover测试代码
from ragen.env.undercover.config import UndercoverEnvConfig
from ragen.env.undercover.env import UndercoverEnv
# 根据infer得到的结果，运行函数提取模型生成的信息
import json
import re
import time
import datasets
from tqdm import trange
import random
import os, logging
from collections import Counter
from openai import OpenAI
from verl.utils import hf_tokenizer
import argparse

root_path = '/root/autodl-fs'  # '/data1/lvnuoyan' 
test_round = 100
config = UndercoverEnvConfig(
    max_env_try=1,  # 修改最大尝试次数
    player_info=[
        # 固定对手gemini-2.5-flash
        {'model_name': 'google/gemini-2.5-flash'},
        {'model_name': 'google/gemini-2.5-flash'},
        {'model_name': 'google/gemini-2.5-flash'}
    ]
)
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="undercover")
parser.add_argument("--model_name", type=str, default="Qwen3-14B")
args = parser.parse_args()
model_path = args.model_path
model_name = args.model_name
port = '1414'
# tokenizer = hf_tokenizer(f"{root_path}/{model_path}/{model_name}")
tokenizer = hf_tokenizer(f"{root_path}/{model_name}")

client = OpenAI(
    api_key="EMPTY",  # vLLM 无需认证密钥
    base_url=f"http://localhost:{port}/v1"
)


def call_model(prompt):
    message0 = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=f"{root_path}/{model_name}",
        messages=message0,
        temperature=0.5,
        max_tokens=200,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
    return response.choices[0].message.content


def reformat_prompt(prompt0):
    # 将prompt句子末尾的 Let\'s think step by step and output the final answer after "####".
    # 替换为Let\'s think step by step and output your think and final answer in this format: 
    # <think> [your thought] </think> <answer> [your answer] </answer>
    prompt = prompt0 + "Always output: <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: 200 words (tokens)."
    message = [{"role": "system", "content": "You're a helpful assistant. "},
               {"role": "user", "content": prompt}]
    # apply_chat_template
    prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
    return prompt


if __name__ == '__main__':
    env = UndercoverEnv(config=config)
    info_list = []
    for t in trange(100):
        env.reset(seed=random.randint(1, 1000))
        # 游戏进行
        prompt = env.render()
        while True:
            # print(prompt + prompt0)
            # 得到trainer的行动
            prompt = reformat_prompt(prompt)
            print(prompt)
            outputs = call_model(prompt)
            print(outputs)
            pattern = r'<answer>(.*?)</answer>'
            match = re.search(pattern, outputs, re.DOTALL)
            if not match:
                info_list.append('trainer-invalid-format')
                print('trainer-invalid-format')
                break
            action = match.group(1)
            # 更新环境信息，得到对手操作以及下一步信息
            prompt, reward, done, info = env.step(action)
            # print(reward, done, info)
            # 检查游戏是否结束，更改了invalid——这里设置invalid为游戏失败、直接结束
            if "Invalid action" in prompt:
                info_list.append('trainer-invalid-output')
                print('trainer-invalid-output')
                break
            if done:
                # 如果结束检查游戏状态，添加对应的状态信息
                # if "Congratulations!" in prompt:
                #     info_list.append('success')
                #     print('success')
                # invalid: env_player不符合指令遵循
                # wrong：env_player的动作不在available_action中
                if "env invalid output" in prompt:
                    info_list.append('env_player-invalid-output')
                    print('env_player-invalid-output')
                else:
                    result = 'success' if reward else 'fail'
                    info_list.append(result)
                    print(result)
                break
    # 统计info_list中出现的不同情况的次数并计算对应的比例
    counter = Counter(info_list)
    total = len(info_list)
    # 存储结果文件
    with open('reason_test/undercover-log.txt', 'a') as f:
        f.write(f"\n=== Model: {model_name} ===\n")
        f.write(f"undercover v.s. {config.player_info[0]['model_name']}\n")
        f.write("undercover test set\n")
        for key, value in counter.items():
            f.write(f"{key}: {value / total:.2%}\n") 

