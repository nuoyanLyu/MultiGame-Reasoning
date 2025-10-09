# 根据infer得到的结果，运行函数提取模型生成的信息
import json
import re
import time
import datasets
import tqdm
from math_verify import parse, verify
import os
from vllm import LLM, SamplingParams
from verl.utils import hf_tokenizer
import argparse
from datasets import load_dataset

root_path = '/root/autodl-tmp'  # '/data1/lvnuoyan' 
batch_size = 16
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="mix")
parser.add_argument("--model_name", type=str, default="mix50")
args = parser.parse_args()
model_path = args.model_path
model_name = args.model_name
tokenizer = hf_tokenizer(f"{root_path}/{model_path}/{model_name}")
# tokenizer = hf_tokenizer(f"{root_path}/{model_name}")
time_str = time.strftime("%m-%d-%H-%M", time.localtime())
# file_name = 'game100-gsm8k-09-23-17-39.json'
# file_name = 'Qwen2.5-1.5B-Instruct-gsm8k-09-23-17-44.json'
# game100不使用原始prompt17-39： 0.4025, strict 0.3161
# Game100使用原始prompt21-55：0.3321——所以改了prompt反而效果更差，，strict 0.2009

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


def extract_solution(solution_str):
    pattern = r"<answer>(.*)</answer>"
    # 使用 re.search() 提取最后一个数字
    match = re.search(pattern, solution_str, re.DOTALL)
    if match:
        last_number = match.group(1)  # 提取匹配到的最后一个数字
        solu_strict = last_number
    else:
        solu_strict = None 
    return solu_strict


def extract_choice(text: str):
    """
    从文本中提取第一个A-D字母（句首答案），允许格式：
    - A
    - A.
    - A.hello
    - A something
    不匹配：
    - ADHD, BADGE 等单词中嵌入的情况
    """
    text = text.strip()
    match = re.search(r'(?<![A-Za-z])([A-D])(\.|(?:\s|$))', text)
    if match:
        return match.group(1).upper()
    return None


def test_math(llm, sampling_params, mmlu):
    answers = []
    accs = {}
    for i in tqdm.trange(0, len(mmlu['question']), batch_size):  # len(math['test'])
        # 调整prompt内容，之前的格式不太对劲，导致模型输出的最后一个数字不是最后一个数字
        data = mmlu[i: i + batch_size]
        prompt = [reformat_prompt(data['question'][j]) for j in range(len(data['question']))]# 模型推理
        outputs = llm.generate(prompt, sampling_params)
        for j, out in enumerate(outputs):
            # answer, choices, subject
            solution = extract_solution(out.outputs[0].text)
            answers.append(out.outputs[0].text)
            label = data['subject'][j]
            if solution is None:
                choice = None
            else:
                choice = extract_choice(solution)
                ground_truth = ['A', 'B', 'C', 'D'][data['answer'][j]]
                choice = (choice == ground_truth)
            if label not in accs:
                accs[label] = []
            accs[label].append(choice)
            exit(0)
            
    return answers, accs

if __name__ == '__main__':
    path0 = f'{root_path}/reasoning'
    # 加载 MMLU 数据集，全部数据集all、对应数据字段test
    mmlu = load_dataset(f"{path0}/mmlu/", 'all')['test']
    llm, sampling_params = load_llm()
    # exit(0)
    accs, answers = test_mmlu(llm, sampling_params, mmlu)
    acc0 = accs.count(1) / len(accs)
    print('model:', model_name)
    print('mmlu test set')
    print('-----strict mode-----')
    acc_list = []
    for k in acc.keys():
        acc0 = acc[k].count(1) / len(acc[k])
        invalid = acc[k].count(None) / len(acc[k])
        print(k, format(acc0, '.4f'))
        print('invalid_output', format(invalid, '.4f'))
        acc_list += acc0
    print('total acc:', format(acc_list.count(1) / len(acc_list), '.4f'))
    print('invalid output:', format(acc_list.count(None) / len(acc_list), '.4f'))
    # answers存储
    with open(f'reason_test/mmlu-{model_name}-{time_str}.json', 'w') as f:
        json.dump(answers, f)
