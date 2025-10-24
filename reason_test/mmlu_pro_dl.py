# 根据infer得到的结果，运行函数提取模型生成的信息
import json
import re
import time
import datasets
import tqdm
from math_verify import parse, verify
import os
import logging
from vllm import LLM, SamplingParams
from verl.utils import hf_tokenizer
import argparse
from datasets import load_dataset
import string

root_path = '/root/autodl-fs'  # '/data1/lvnuoyan' 
batch_size = 16
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="tictactoe")
parser.add_argument("--model_name", type=str, default="game100")
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


def reformat_prompt(prompt0, choice):
    # 将prompt句子末尾的 Let\'s think step by step and output the final answer after "####".
    # 替换为Let\'s think step by step and output your think and final answer in this format: 
    # <think> [your thought] </think> <answer> [your answer] </answer>
    formatted_question = prompt0.strip() + "\n"
    for i, option in enumerate(choice):
        letter = string.ascii_uppercase[i]  # 依次取 A, B, C, ...
        formatted_question += f"{letter}. {option}\n"
    prompt = formatted_question + "Let\'s think step by step and always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: 200 words (tokens)."
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
    从文本中提取第一个A-J字母（句首答案），允许格式：
    - A
    - A.
    - A.hello
    - A something
    不匹配：
    - ADHD, BADGE 等单词中嵌入的情况
    """
    text = text.strip()
    match = re.search(r'(?<![A-Za-z])([A-J])(\.|(?:\s|$))', text)
    if match:
        return match.group(1).upper()
    return None


def test_mmlu(llm, sampling_params, mmlu):
    answers = []
    accs = {}
    for i in tqdm.trange(0, len(mmlu), batch_size):  # len(math['test'])
        # 调整prompt内容，之前的格式不太对劲，导致模型输出的最后一个数字不是最后一个数字
        data = mmlu[i: i + batch_size]
        prompt = [reformat_prompt(data['question'][j], data['options'][j]) for j in range(len(data['question']))]# 模型推理
        outputs = llm.generate(prompt, sampling_params)
        for j, out in enumerate(outputs):
            solution = extract_solution(out.outputs[0].text)
            answers.append(out.outputs[0].text)
            label = data['category'][j]
            if solution is None:
                choice = None
            else:
                choice = extract_choice(solution)
                ground_truth = data['answer'][j]
                choice = (choice == ground_truth)
            if label not in accs:
                accs[label] = []
            accs[label].append(choice)
            
    return answers, accs


def save_results_to_markdown(acc, model_name, output_file='reason_test/mmlu-pro-results.txt'):
    acc_list = []
    rows = []
    print('mmlu pro, model', model_name)
    # 计算每个 key 的准确率和 invalid 率
    for k in acc.keys():
        total = len(acc[k])
        valid = acc[k].count(1)
        acc_rate = valid / total

        # 构造 markdown 表格每一行
        rows.append(f"| {k} | {acc_rate:.4f} |")

        acc_list += acc[k]

    total_acc = acc_list.count(1) / len(acc_list)
    total_invalid = acc_list.count(None) / len(acc_list)

    with open(output_file, "a") as f:
        f.write(f"\n## Model: {model_name}\n\n")
        f.write(f"| Subject | {model_name} |\n")
        f.write("|----------|-----------|\n")
        for row in rows:
            f.write(row + "\n")
        f.write(f"| **Total** | **{total_acc:.4f}** |\n\n")
    print(f"total acc: {total_acc:.4f}")
    print(f"Results saved to {output_file}")


if __name__ == '__main__':
    os.environ["VLLM_DISABLE_PROGRESS_BAR"] = "1"
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

    logging.getLogger("vllm").setLevel(logging.ERROR)
    path0 = f'/root/autodl-tmp/reasoning'
    # 加载 MMLU 数据集，全部数据集all、对应数据字段test
    # mmlu = load_dataset(f"{path0}/mmlu/", 'all')['test']
    mmlu_pro = load_dataset(f"{path0}/MMLU-Pro")['test']
    llm, sampling_params = load_llm()
    # exit(0)
    answers, acc = test_mmlu(llm, sampling_params, mmlu_pro)
    acc_list = []
    save_results_to_markdown(acc, model_name)
