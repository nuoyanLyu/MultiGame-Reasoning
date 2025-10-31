# 根据infer得到的结果，运行函数提取模型生成的信息
import json
import re
import time
import datasets
import tqdm
from math_verify import parse, verify
import os, logging
from vllm import LLM, SamplingParams
from verl.utils import hf_tokenizer
import argparse
import random

root_path = '/root/autodl-tmp'  # '/data1/lvnuoyan' 
batch_size = 16
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="tictactoe")
parser.add_argument("--model_name", type=str, default="game50")
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
    prompt = prompt0.replace("Let\'s think step by step and output the final answer after \"####\".", 
                             "Let\'s think step by step and always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: 200 words (tokens).")
    message = [{"role": "system", "content": "You're a helpful assistant. "},
               {"role": "user", "content": prompt}]
    # apply_chat_template
    prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
    return prompt


def extract_solution(solution_str):
    pattern = r"<answer>(.*)</answer>"
    # 使用 re.search() 提取最后一个数字
    match = re.search(pattern, solution_str, re.DOTALL)
    solu_flex = solution_str
    if match:
        last_number = match.group(1)  # 提取匹配到的最后一个数字
        solu_strict = last_number
        solu_flex = last_number
    else:
        solu_strict = None 
    return solu_strict, solu_flex


def test_math(llm, sampling_params, math):
    accs_strict = []
    accs_flex = []
    answers = []
    for i in tqdm.trange(0, len(math['test']), batch_size):  # len(math['test'])
        # 调整prompt内容，之前的格式不太对劲，导致模型输出的最后一个数字不是最后一个数字
        data = math['test'][i: i + batch_size]
        prompt = [reformat_prompt(data['prompt'][j][0]['content']) for j in range(len(data['prompt']))]# 模型推理
        outputs = llm.generate(prompt, sampling_params)
        for j, out in enumerate(outputs):
            solu_strict, solu_flex = extract_solution(out.outputs[0].text)
            answers.append(out.outputs[0].text)
            # print(outputs[0].outputs[0].text)
            ground_truth = parse(data['reward_model'][j]['ground_truth'])
            correct_strict = verify(parse(solu_strict), ground_truth)
            if solu_strict is None:
                accs_strict.append(None)
            elif correct_strict:
                accs_strict.append(1)
            else:
                # print(solution, ground_truth)
                accs_strict.append(0)
            correct_flex = verify(parse(solu_flex), ground_truth)
            if solu_flex is None:
                accs_flex.append(None)
            elif correct_flex:
                accs_flex.append(1)
            else:
                accs_flex.append(0)
    return accs_strict, accs_flex, answers


def save_log(model_name, accs_strict, accs_flex, output_file="reason_test/gsm8k-log.txt"):
    acc0_strict = accs_strict.count(1) / len(accs_strict)
    invalid_strict = accs_strict.count(None)
    acc0_flex = accs_flex.count(1) / len(accs_flex)

    # 写入日志文件
    with open(output_file, "a") as f:
        f.write(f"\n=== Model: {model_name} ===\n")
        f.write("gsm8k test set\n")
        f.write("strict mode\n")
        f.write(f"total acc: {acc0_strict:.4f}\n")
        f.write(f"invalid output: {invalid_strict}\n")
        f.write("flexible mode\n")
        f.write(f"total acc: {acc0_flex:.4f}\n")

    print(f"✅ Log saved to {output_file}")


def save_sample_results(model_name, accs_strict, accs_flex, answers, math_data, 
                       num_samples=5, output_dir="reason_test/results"):
    """
    保存部分正确和错误的测试结果，便于分析
    
    Args:
        model_name: 模型名称
        accs_strict: 严格模式下的准确率列表
        accs_flex: 灵活模式下的准确率列表
        answers: 模型生成的答案列表
        math_data: 测试数据集
        num_samples: 每种类型保存的样本数量
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    time_str = time.strftime("%m-%d-%H-%M", time.localtime())
    output_file = f"{output_dir}/{model_name}-gsm8k-{time_str}.json"
    
    # 收集正确和错误的样本索引
    correct_indices = []
    incorrect_indices = []
    invalid_indices = []
    
    for i, (acc_strict, acc_flex) in enumerate(zip(accs_strict, accs_flex)):
        if acc_strict is None:
            invalid_indices.append(i)
        elif acc_strict == 1:
            correct_indices.append(i)
        else:
            incorrect_indices.append(i)
    
    # 限制样本数量
    correct_indices = correct_indices[:num_samples]
    incorrect_indices = incorrect_indices[:num_samples]
    # invalid担心数量不够——先判断一下
    if len(invalid_indices) > num_samples:
        invalid_indices = invalid_indices[:num_samples]
    
    # 准备保存的数据
    samples = {
        "model_name": model_name,
        "correct_samples": [],
        "incorrect_samples": [],
        "invalid_samples": [],
    }
    
    # 添加正确样本
    for idx in correct_indices:
        samples["correct_samples"].append({
            "question": math_data['test']['prompt'][idx][0]['content'],
            "model_answer": answers[idx],
            "ground_truth": math_data['test']['reward_model'][idx]['ground_truth'],
        })
    
    # 添加错误样本
    for idx in incorrect_indices:
        samples["incorrect_samples"].append({
            "question": math_data['test']['prompt'][idx][0]['content'],
            "model_answer": answers[idx],
            "ground_truth": math_data['test']['reward_model'][idx]['ground_truth'],
        })
    
    # 添加invalid样本
    for idx in invalid_indices:
        samples["invalid_samples"].append({
            "question": math_data['test']['prompt'][idx][0]['content'],
            "model_answer": answers[idx],
            "ground_truth": math_data['test']['reward_model'][idx]['ground_truth'],
        })
    
    
    # 保存到文件
    with open(output_file, "w") as f:
        json.dump(samples, f, indent=4)
    
    print(f"✅ Sample results saved to {output_file}")
    print(f"Saved {len(correct_indices)} correct, {len(incorrect_indices)} incorrect samples and {len(invalid_indices)} invalid samples.")


if __name__ == '__main__':
    os.environ["VLLM_DISABLE_PROGRESS_BAR"] = "1"
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

    logging.getLogger("vllm").setLevel(logging.ERROR)
    path0 = f'/root/autodl-tmp/reasoning'
    math = datasets.load_dataset("parquet", 
                  data_files={'train': path0 + '/gsm8k/train.parquet', 'test': path0 + '/gsm8k/test.parquet'})
    # print(math['test']['prompt'][0])
    llm, sampling_params = load_llm()
    # exit(0)
    accs_strict, accs_flex, answers = test_math(llm, sampling_params, math)
    acc0 = accs_strict.count(1) / len(accs_strict)
    print('model:', model_name)
    print('gsm8k test set')
    print('-----strict mode-----')
    print('total acc:', format(acc0, '.4f'))
    print('invalid output:', accs_strict.count(None))
    print('----flexible mode----')
    acc0 = accs_flex.count(1) / len(accs_flex)
    print('total acc:', format(acc0, '.4f'))
    # 写入日志文件
    save_log(model_name, accs_strict, accs_flex)
    # 保存部分测试结果，便于分析
    save_sample_results(model_name, accs_strict, accs_flex, answers, math, num_samples=20)
