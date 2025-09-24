from openai import OpenAI
import datasets
import re
import tqdm
import json
import argparse
import os
import time
from datetime import datetime
from verl.utils import hf_tokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--port', type=str, default="2100")
# tictactoe/grpo/game_200
# Qwen2.5-1.5B-Instruct
parser.add_argument('--model_name', type=str, default='game100')
# model_name = 'Qwen3-1.7B'
args = parser.parse_args()

port = args.port
model_name = args.model_name
model_folder = 'tictactoe-gemini'
root_path = '/root/autodl-tmp'# '/data1/lvnuoyan'
# tokenizer = hf_tokenizer(f"{root_path}/llm_model/{model_name}")
tokenizer = hf_tokenizer(f"{root_path}/{model_folder}/{model_name}")
# 禁用代理（只在本脚本有效）——服务器联网有问题，这样保证正常访问VLLM load的模型
for key in ["http_proxy", "https_proxy", "all_proxy", 
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    os.environ.pop(key, None)

# 配置 OpenAI 客户端（兼容 vLLM 的 OpenAPI 接口）
client = OpenAI(
    api_key="EMPTY",  # vLLM 无需认证密钥，任意字符串均可
    base_url=f"http://localhost:{port}/v1"  # 与 vLLM 服务端口一致 3333没人用之后我都用这个端口吧
)

def llm_output(text: str) -> str:
    try:
        message = [{"role": "system", "content": "You're a helpful assistant. "},
                   {"role": "user", "content": text}]
        prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        response = client.completions.create(
            model=f"{root_path}/{model_folder}/{model_name}",
            prompt=prompt,
            max_tokens=1000,
            temperature=0.5,
        )
        # print(response.choices[0].text)
        return response.choices[0].text
    except Exception as e:
        raise RuntimeError(f"API 调用失败：{str(e)}")

def reformat_prompt(prompt0):
    # 将prompt句子末尾的 Let\'s think step by step and output the final answer after "####".
    # 替换为Let\'s think step by step and output your think and final answer in this format: 
    # <think> [your thought] </think> <answer> [your answer] </answer>
    prompt = prompt0.replace("Let\'s think step by step and output the final answer after \"####\".", 
                             "Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: 200 words (tokens).")
    message = [{"role": "system", "content": "You're a helpful assistant. "},
               {"role": "user", "content": prompt}]
    # apply_chat_template
    prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
    return prompt

def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]
    if method == "strict":
        # this also tests the formatting of the model
        # 更精确的正则表达式
        pattern = r"<answer>.*?(-?\d+\.?\d*)[^0-9]*</answer>"
        # 使用 re.search() 提取最后一个数字
        match = re.search(pattern, solution_str, re.DOTALL)
        if match:
            last_number = match.group(1)  # 提取匹配到的最后一个数字
            return last_number
        else:
            return None 
    elif method == "flexible":
        # 但是这个flexible匹配明明已经是宽松版本了，但还是匹配不到——可能还是训废了？        
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def test_math(method='strict'):
    accs = []
    answers = []
    for i in tqdm.trange(len(math['test'])):  # len(math['test'])
        # 调整prompt内容，之前的格式不太对劲
        q = math['test']['prompt'][i][0]['content']
        q = reformat_prompt(q)
        # print(q)
        ground_truth = math['test']['reward_model'][i]['ground_truth']
        a = llm_output(q)
        # print(a)
        answers.append(a)
        solution = extract_solution(a, method)
        # print(solution)
        # print(type(ground_truth))
        # 之前发现可能表达式不同但实际上是一个数值的情况，比如100.00和100，然后可能有多余的
        # 先不考虑这个情况
        if solution is None:
            accs.append(None)
        elif solution == ground_truth:
            accs.append(1)
        else:
            accs.append(0)
    return accs, answers


path0 = f'{root_path}/reasoning'
math = datasets.load_dataset("parquet", 
              data_files={'train': path0 + '/gsm8k/train.parquet', 'test': path0 + '/gsm8k/test.parquet'})
# print(math['test']['prompt'][0])

# exit(0)
accs, answers = test_math('strict')
acc0 = accs.count(1) / len(accs)
print('total acc:', acc0)
print('invalid output:', accs.count(None))
TIME = datetime.now().strftime("%m-%d-%H-%M")
# 删除特殊字符
model_name = model_name.replace('/', '').replace('\\', '')
with open(f'{model_name}-gsm8k-{TIME}.json', 'w') as f:
    f.write(json.dumps(answers))
