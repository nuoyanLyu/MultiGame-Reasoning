# 改动之前的tictactoe测试代码，改为使用python的VLLM load训练模型、并尝试初始化对手的config（是否有可能）
from ragen.env.tictactoe.config import TicTacToeEnvConfig
from ragen.env.tictactoe.env import TicTacToeEnv
# 根据infer得到的结果，运行函数提取模型生成的信息
import json
import re
import time
import datasets
from tqdm import trange
import random
from math_verify import parse, verify
import os, logging
from collections import Counter
from vllm import LLM, SamplingParams
from verl.utils import hf_tokenizer
import argparse

root_path = '/root/autodl-tmp'  # '/data1/lvnuoyan' 
test_round = 100
config = TicTacToeEnvConfig(
    max_env_try=1,  # 修改最大尝试次数
    player_info=[
        {'model_name': 'google/gemini-2.5-flash'}
    ]
)
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="tictactoe")
parser.add_argument("--model_name", type=str, default="game100")
args = parser.parse_args()
model_path = args.model_path
model_name = args.model_name
tokenizer = hf_tokenizer(f"{root_path}/{model_path}/{model_name}")
# tokenizer = hf_tokenizer(f"{root_path}/{model_name}")

def load_llm():
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    model = f'{root_path}/{model_path}/{model_name}'
    # model = f"{root_path}/{model_name}"
    # ro_config = config.actor_rollout_ref.rollout
    llm = LLM(
		model,
        max_model_len=6000,
	)
    print("LLM initialized")
    sampling_params = SamplingParams(
		max_tokens=600, # ro_config.response_length,
		temperature=0.5,  # ro_config.val_kwargs.temperature,
	)
    return llm, sampling_params


def reformat_prompt(prompt0):
    # 将prompt句子末尾的 Let\'s think step by step and output the final answer after "####".
    # 替换为Let\'s think step by step and output your think and final answer in this format: 
    # <think> [your thought] </think> <answer> [your answer] </answer>
    prompt = prompt0 + "Let\'s think step by step and always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: 200 words (tokens)."
    message = [{"role": "system", "content": "You're a helpful assistant. "},
               {"role": "user", "content": prompt}]
    # apply_chat_template
    prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
    return prompt

if __name__ == '__main__':
    llm, sampling_params = load_llm()
    env = TicTacToeEnv(config=config)
    info_list = []
    for t in trange(100):
        env.reset0(seed=random.randint(1, 1000))
        # 游戏进行
        prompt = env.render()
        while True:
            # print(prompt + prompt0)
            # 得到trainer的行动
            prompt = reformat_prompt(prompt)
            outputs = llm.generate([prompt], sampling_params)
            output = outputs[0].outputs[0].text
            pattern = r'.*<think>(.*?)</think>\s*<answer>(.*?)</answer>$'
            match = re.match(pattern, output, re.DOTALL)
            if not match:
                info_list.append('trainer-invalid-format')
                print('trainer-invalid-format')
                break
            action = match.group(2)
            # 更新环境信息，得到对手操作以及下一步信息
            prompt, reward, done, info = env.step(action)
            # 检查游戏是否结束，更改了invalid——这里设置invalid为游戏失败、直接结束
            if "Invalid action:" in prompt:
                info_list.append('trainer-invalid-output')
                print('trainer-invalid-output')
                break
            if done:
                # 如果结束检查游戏状态，添加对应的状态信息
                if "Congratulations!" in prompt:
                    info_list.append('success')
                    print('success')
                elif "Draw!" in prompt:
                    info_list.append('draw')
                    print('draw')
                # invalid: env_player不符合指令遵循
                # wrong：env_player的动作不在available_action中
                elif "Your opponent made a mistake" in prompt:
                    info_list.append('env_player-invalid-output')
                    print('env_player-invalid-output')
                elif "Your opponent action is wrong" in prompt:
                    info_list.append('env_player-wrong-output')
                    print('env_player-wrong-output')
                elif "Failed! " in prompt:
                    info_list.append('fail')
                    print('fail')
                break
    # 统计info_list中出现的不同情况的次数并计算对应的比例
    counter = Counter(info_list)
    total = len(info_list)
    # 存储结果文件
    with open('reason_test/tictactoe-log.txt', 'a') as f:
        f.write(f"\n=== Model: {model_name} ===\n")
        f.write("tictactoe test set\n")
        for key, value in counter.items():
            f.write(f"{key}: {value / total:.2%}\n")
        f.write(f"tictactoe v.s. {config.player_info[0]['model_name']}\n")
        f.write(f"model: {model_name}\n\n")
 

