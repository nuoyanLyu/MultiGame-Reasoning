import json
from collections import Counter
from vllm import LLM, SamplingParams
from verl.utils import hf_tokenizer
import argparse
import tqdm
import random
import re
import os
import logging
import time

root_path = '/root/autodl-tmp'
batch_size = 16

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="nash-math")
parser.add_argument("--model_name", type=str, default="nm50")
args = parser.parse_args()
model_path = args.model_path
model_name = args.model_name
tokenizer = hf_tokenizer(f"{root_path}/{model_path}/{model_name}")
# tokenizer = hf_tokenizer(f"{root_path}/{model_name}")

# jsonl不能直接读取，每一行是一个单独的json对象
# test_data = json.load(open(f"{root_path}/reasoning/socialiqa-train-dev/dev.jsonl"))
social = []
with open(f"{root_path}/reasoning/socialiqa-train-dev/dev.jsonl", "r") as f:
    for line in f:
        social.append(json.loads(line))
with open(f"{root_path}/reasoning/socialiqa-train-dev/dev-labels.lst", "r") as f:
    labels = f.read().splitlines()
labels = [int(l) - 1 for l in labels]

def load_llm():
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    model = f'{root_path}/{model_path}/{model_name}'
    # model = f"{root_path}/{model_name}"
    # ro_config = config.actor_rollout_ref.rollout
    llm = LLM(
		model,
        max_model_len=8000,
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


def test_social(llm, sampling_params, social, ground_truth):
    answers = []
    accs = {}
    acc_list = []
    for i in tqdm.trange(0, 10, batch_size):  # len(math['test'])
        # 调整prompt内容，之前的格式不太对劲，导致模型输出的最后一个数字不是最后一个数字
        data = social[i: i + batch_size]
        labels = ground_truth[i: i + batch_size]
        print(labels)
        # key类型以及lm_eval_harness对应的prompt设计
        '''
        doc_to_text: "{{context}}\nQuestion: {{question}}\nAnswer:"  
        doc_to_target: "{{answer}}"  # 正确答案的索引  
        doc_to_choice: "{{[answerA, answerB, answerC]}}"  # 三个选项
        '''
        prompt = [reformat_prompt(data[j]['context'] + '\nQuestion: ' + 
                                  data[j]['question'], 
                                 [data[j]['answerA'], data[j]['answerB'], data[j]['answerC']]) for j in range(len(data))]# 模型推理
        # outputs = llm.generate(prompt, sampling_params)
        outputs = [None] * len(prompt)
        for j, out in enumerate(outputs):
            # answer, choices, subject
            solution = extract_solution(out.outputs[0].text)
            answers.append(out.outputs[0].text)
            if solution is None:
                choice = None
            else:
                choice = extract_choice(solution)
                print(choice)
                ground_truth = ['A', 'B', 'C'][labels[j]]
                choice = (choice == ground_truth)
            acc_list.append(choice)
    return answers, acc_list


def save_sample_results(model_name, acc_list, answers, social, ground_truth, 
                       num_samples=5, output_dir="reason_test/results"):
    """
    保存部分正确和错误的测试结果，便于分析
    
    Args:
        model_name: 模型名称
        acc_list: 准确率列表、不区分类别
        answers: 模型生成的答案列表
        social: 测试数据集
        ground_truth: 正确答案列表
        num_samples: 每种类型保存的样本数量
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    time_str = time.strftime("%m-%d-%H-%M", time.localtime())
    output_file = f"{output_dir}/{model_name}-socialIQA-{time_str}.json"
    
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
        data_idx = idx
        samples["correct_samples"].append({
            "question": social[data_idx]['context'] + '\nQuestion: ' + social[data_idx]['question'],
            "choices": [social[data_idx]['answerA'], social[data_idx]['answerB'], social[data_idx]['answerC']],
            "model_answer": answers[idx],
            "ground_truth": ['A', 'B', 'C'][ground_truth[data_idx]],
        })
    
    # 添加错误样本
    for idx in incorrect_indices:
        # 获取对应的题目信息
        data_idx = idx
        samples["incorrect_samples"].append({
            "question": social[data_idx]['context'] + '\nQuestion: ' + social[data_idx]['question'],
            "choices": [social[data_idx]['answerA'], social[data_idx]['answerB'], social[data_idx]['answerC']],
            "model_answer": answers[idx],
            "ground_truth": ['A', 'B', 'C'][ground_truth[data_idx]],
        })
    
    # 添加无效样本
    for idx in invalid_indices:
        # 获取对应的题目信息
        data_idx = idx
        samples["invalid_samples"].append({
            "question": social[data_idx]['context'] + '\nQuestion: ' + social[data_idx]['question'],
            "choices": [social[data_idx]['answerA'], social[data_idx]['answerB'], social[data_idx]['answerC']],
            "model_answer": answers[idx],
            "ground_truth": ['A', 'B', 'C'][ground_truth[data_idx]],
        })
    
    # 保存到文件
    with open(output_file, "w") as f:
        json.dump(samples, f, indent=4)
    
    print(f"✅ Sample results saved to {output_file}")
    print(f"Saved {len(correct_indices)} correct, {len(incorrect_indices)} incorrect and {len(invalid_indices)} invalid samples")


def save_log(model_name, accs_list, output_file="reason_test/socialIQA-log.txt"):  
    acc0_strict = accs_list.count(1) / len(accs_list)
    invalid_strict = accs_list.count(None)

    # 写入日志文件
    with open(output_file, "a") as f:
        f.write(f"\n=== Model: {model_name} ===\n")
        f.write("socialiqa dev test set\n")
        f.write("strict mode\n")
        f.write(f"total acc: {acc0_strict:.4f}\n")
        f.write(f"invalid output: {invalid_strict}\n")

    print(f"✅ Log saved to {output_file}")


if __name__ == '__main__':
    os.environ["VLLM_DISABLE_PROGRESS_BAR"] = "1"
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

    logging.getLogger("vllm").setLevel(logging.ERROR)
    path0 = f'/root/autodl-tmp/reasoning'
    llm, sampling_params = load_llm()
    answers, acc_list = test_social(llm, sampling_params, social, labels)
    # 保存测试结果到日志文件
    save_log(model_name, acc_list)
    # 保存部分测试结果，便于分析
    save_sample_results(model_name, acc_list, answers, social, labels, num_samples=20)
