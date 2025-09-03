from openai import OpenAI
import datasets
import re
import tqdm
import json
import argparse
import os
import time
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--port', type=str, default='20202')
# tictactoe/grpo/game_200
# Qwen2.5-1.5B-Instruct
parser.add_argument('--model_name', type=str, default='tictactoe/grpo/game_200')
# model_name = 'Qwen3-1.7B'
args = parser.parse_args()

port = args.port
model_name = args.model_name
root_path = '/data1/lvnuoyan'

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
        response = client.chat.completions.create(
            model=f"{root_path}/llm_model/{model_name}",  # 使用模型路径，如通过--served-model-name指定名称需与 vLLM 服务启动时指定的名称一致
            messages=[{
                "role": "user",
                "content": f"{text}"
            }],
            max_tokens=1024,  # 控制生成文本长度[4](@ref)
            temperature=0.5,  # 控制生成随机性（0-1，越高越随机）[4](@ref)
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"API 调用失败：{str(e)}")


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    if method == "strict":
        # this also tests the formatting of the model
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "")
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
        q = math['test']['prompt'][i]
        ground_truth = math['test']['reward_model'][i]['ground_truth']
        a = llm_output(q)
        # print(a)
        answers.append(a)
        solution = extract_solution(a, method)
        # print(solution)
        # nprint(ground_truth)
        if solution == ground_truth:
            accs.append(1)
            # print(1)
        else:
            accs.append(0)
            # print(0)

    return accs, answers


path0 = f'{root_path}/reasoning'
math = datasets.load_dataset("parquet", 
              data_files={'train': path0 + '/gsm8k/train.parquet', 'test': path0 + '/gsm8k/test.parquet'})

accs, answers = test_math('flexible')
acc0 = sum(accs) / len(accs)
print('total acc:', acc0)
TIME = datetime.now().strftime("%m-%d-%H-%M")
# 删除特殊字符
model_name = model_name.replace('/', '').replace('\\', '')
with open(f'{model_name}-gsm8k-{TIME}.json', 'w') as f:
    f.write(json.dumps(answers))
