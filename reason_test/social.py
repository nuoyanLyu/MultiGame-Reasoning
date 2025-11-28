import json
from collections import Counter
from vllm import LLM, SamplingParams
from verl.utils import hf_tokenizer
import argparse
from tqdm import trange
import random
import re

root_path = '/root/autodl-tmp'


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="undercover")
parser.add_argument("--model_name", type=str, default="Qwen2.5-1.5B-Instruct")
args = parser.parse_args()
model_path = args.model_path
model_name = args.model_name
tokenizer = hf_tokenizer(f"{root_path}/{model_path}/{model_name}")
# tokenizer = hf_tokenizer(f"{root_path}/{model_name}")

# jsonl不能直接读取，每一行是一个单独的json对象
# test_data = json.load(open(f"{root_path}/reasoning/socialiqa-train-dev/dev.jsonl"))
test_data = []
with open(f"{root_path}/reasoning/socialiqa-train-dev/dev.jsonl", "r") as f:
    for line in f:
        test_data.append(json.loads(line))
with open(f"{root_path}/reasoning/socialiqa-train-dev/dev-labels.lst", "r") as f:
    labels = f.read().splitlines()


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


