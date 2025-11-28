# NashEnv evaluation script using vLLM-served model
import sys
import os
from ragen.env.nash_new.env import NashNew
from ragen.env.nash_new.config import NashNewConfig
import json
import re
import time
from tqdm import trange
import random
from collections import Counter
import argparse
from vllm import LLM, SamplingParams
from verl.utils import hf_tokenizer

# Setup
root_path = '/root/autodl-tmp'
test_round = 100

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="nash-new")
parser.add_argument("--model_name", type=str, default='Qwen2.5-1.5B-Instruct')
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
    # Append fixed instruction suffix (no chat template, plain concatenation)
    # Guide model to analyze payoffs without mentioning "Nash equilibrium"
    prompt = prompt0 + "Let\'s think step by step and always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: 200 words (tokens)."
    message = [{"role": "system", "content": "You're a helpful assistant. "},
               {"role": "user", "content": prompt}]
    # apply_chat_template
    prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
    return prompt

def extract_action(output):
    """Extract action from <answer>...</answer> tags"""
    # First try to match the full pattern
    pattern = r'<answer>\s*(.*)\s*</answer>'
    match = re.search(pattern, output, re.DOTALL)
    if match:
        return match.group(1).strip()
    # # Fallback: find any <answer> tag and extract first 1 or 2
    # answer_pattern = r'<answer>(.*?)</answer>'
    # match = re.search(answer_pattern, output, re.DOTALL)
    # if match:
    #     answer_content = match.group(1)
    #     # Find first occurrence of 1 or 2
    #     digit_match = re.search(r'[12]', answer_content)
    #     if digit_match:
    #         return digit_match.group(0)
    
    return None

if __name__ == '__main__':
    env = NashNew()
    info_list = []
    run_details = []  # For saving detailed run info
    llm, sampling_params = load_llm()
    for t in trange(test_round):
        seed = random.randint(1, 1000)
        prompt = env.reset(seed=seed)
        # Format prompt with instruction suffix
        formatted_prompt = reformat_prompt(prompt)
        outputs = llm.generate([formatted_prompt], sampling_params)
        output = outputs[0].outputs[0].text
        # Extract action from output
        action_str = extract_action(output)
        if action_str is None:
            info_list.append('invalid-format')
            print('invalid-format')
            continue
        # Keep action as string for env.step() (导师版本需要字符串)
        action = action_str
        # Step environment
        prompt, reward, done, info = env.step(action)
        # Record result
        if info.get('success', False):
            info_list.append('success')
            status = 'success'
            print('success')
        else:
            if not info.get('action_is_valid', True):
                info_list.append('invalid-action')
                status = 'invalid-action'
                print('invalid-action')
            else:
                info_list.append('fail')
                status = 'fail'
                print('fail')
    
    # Statistics
    counter = Counter(info_list)
    total = len(info_list)
    
    # Print summary
    print("\n" + "="*20)
    print(f"NashEnv Test Results (n={total})")
    print("="*20)
    for key, value in counter.items():
        print(f"{key}: {value / total:.2%}")
    
    # success_count = counter.get('success', 0)
    # print(f"\nSuccess rate: {success_count / total:.2%}")
    print("="*20)
    
    # Save results to log file
    with open('reason_test/nashenv-log.txt', 'a') as f:
        f.write(f"\n=== Model: {args.model_name} ===\n")
        f.write("NashEnv test results:\n")
        for key, value in counter.items():
            f.write(f"{key}: {value / total:.2%}\n")
        # f.write(f"Success rate: {success_count / total:.2%}\n")
    print(f"\nResults saved to reason_test/nashenv-log.txt")