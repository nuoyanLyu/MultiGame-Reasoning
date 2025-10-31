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
import random

root_path = '/root/autodl-tmp'  # '/data1/lvnuoyan' 
batch_size = 16
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="nash-new")
parser.add_argument("--model_name", type=str, default="nash50")
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
    formatted_question = (
        prompt0.strip() + "\n"
        + f"A. {choice[0]}\n"
        + f"B. {choice[1]}\n"
        + f"C. {choice[2]}\n"
        + f"D. {choice[3]}\n"
    )
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


def test_mmlu(llm, sampling_params, mmlu, num_samples=5):
    answers = []
    accs = {}
    acc_list = []
    for i in tqdm.trange(0, len(mmlu), batch_size):  # len(math['test'])
        # 调整prompt内容，之前的格式不太对劲，导致模型输出的最后一个数字不是最后一个数字
        data = mmlu[i: i + batch_size]
        prompt = [reformat_prompt(data['question'][j], data['choices'][j]) for j in range(len(data['question']))]# 模型推理
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
            acc_list.append(choice)
            
    return answers, accs, acc_list


def save_results_to_markdown(acc, model_name, output_file='reason_test/mmlu-results.txt'):
    acc_list = []
    rows = []
    print('mmlu, model', model_name)
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
    print(f"total invalid: {total_invalid:.4f}")
    print(f"Results saved to {output_file}")


def save_sample_results(model_name, acc_list, answers, mmlu_data, 
                       num_samples=5, output_dir="reason_test/results"):
    """
    保存部分正确和错误的测试结果，便于分析
    
    Args:
        model_name: 模型名称
        accs: 准确率列表、不区分类别
        answers: 模型生成的答案列表
        mmlu_data: 测试数据集
        num_samples: 每种类型保存的样本数量
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    time_str = time.strftime("%m-%d-%H-%M", time.localtime())
    output_file = f"{output_dir}/{model_name}-mmlu-{time_str}.json"
    
    # 收集正确和错误的样本索引
    correct_indices = []
    incorrect_indices = []
    invalid_indices = []
    
    # 遍历所有类别和样本
    idx = 0
    for result in acc_list:
        if result is None:
            invalid_indices.append(idx)
        elif result == 1:
            correct_indices.append(idx)
        else:
            incorrect_indices.append(idx)
        idx += 1
    
    # 限制样本数量
    if len(correct_indices) > num_samples:
        correct_indices = correct_indices[:num_samples]
    if len(incorrect_indices) > num_samples:
        incorrect_indices = incorrect_indices[:num_samples]
    if len(invalid_indices) > num_samples:
        invalid_indices = invalid_indices[:num_samples]
    
    # 准备保存的数据
    samples = {
        "model_name": model_name,
        "correct_samples": [],
        "incorrect_samples": [],
        "invalid_samples": []
    }
    
    # 添加正确样本
    for idx in correct_indices:
        # 获取对应的题目信息
        data_idx = idx % len(mmlu_data)
        samples["correct_samples"].append({
            "question": mmlu_data['question'][data_idx],
            "choices": mmlu_data['choices'][data_idx],
            "model_answer": answers[idx],
            "ground_truth": ['A', 'B', 'C', 'D'][mmlu_data['answer'][data_idx]],
            "subject": mmlu_data['subject'][data_idx]
        })
    
    # 添加错误样本
    for idx in incorrect_indices:
        data_idx = idx % len(mmlu_data)
        samples["incorrect_samples"].append({
            "question": mmlu_data['question'][data_idx],
            "choices": mmlu_data['choices'][data_idx],
            "model_answer": answers[idx],
            "ground_truth": ['A', 'B', 'C', 'D'][mmlu_data['answer'][data_idx]],
            "subject": mmlu_data['subject'][data_idx]
        })
    
    # 添加无效样本
    for idx in invalid_indices:
        data_idx = idx % len(mmlu_data)
        samples["invalid_samples"].append({
            "question": mmlu_data['question'][data_idx],
            "choices": mmlu_data['choices'][data_idx],
            "model_answer": answers[idx],
            "ground_truth": ['A', 'B', 'C', 'D'][mmlu_data['answer'][data_idx]],
            "subject": mmlu_data['subject'][data_idx]
        })
    
    # 保存到文件
    with open(output_file, "w") as f:
        json.dump(samples, f, indent=4)
    
    print(f"✅ Sample results saved to {output_file}")
    print(f"Saved {len(correct_indices)} correct, {len(incorrect_indices)} incorrect and {len(invalid_indices)} invalid samples")


if __name__ == '__main__':
    os.environ["VLLM_DISABLE_PROGRESS_BAR"] = "1"
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

    logging.getLogger("vllm").setLevel(logging.ERROR)
    path0 = f'/root/autodl-tmp/reasoning'
    # 加载 MMLU 数据集，全部数据集all、对应数据字段test
    mmlu = load_dataset(f"{path0}/mmlu/", 'all')['test']
    llm, sampling_params = load_llm()
    # exit(0)
    answers, acc, acc_list = test_mmlu(llm, sampling_params, mmlu)
    save_results_to_markdown(acc, model_name)
    # 保存部分测试结果，便于分析
    save_sample_results(model_name, acc_list, answers, mmlu, num_samples=20)
