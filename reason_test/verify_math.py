# 根据infer得到的结果，运行函数提取模型生成的信息
import json
import re
import datasets
from pydantic import model_validator
import tqdm
from math_verify import parse, verify

root_path = '/root/autodl-tmp'  # '/data1/lvnuoyan' 
file_name = 'game100-gsm8k-09-23-17-39.json'
mode = 'strict'
# file_name = 'Qwen2.5-1.5B-Instruct-gsm8k-09-23-17-44.json'
# game100不使用原始prompt17-39： 0.4025, strict 0.3161
# Game100使用原始prompt21-55：0.3321——所以改了prompt反而效果更差，，strict 0.2009
def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]
    if method == "strict":
        # this also tests the formatting of the model
        # 更精确的正则表达式
        pattern = r"<answer>(.*)</answer>"
        # 使用 re.search() 提取最后一个数字
        match = re.search(pattern, solution_str, re.DOTALL)
        if match:
            last_number = match.group(1)  # 提取匹配到的最后一个数字
            return last_number
        else:
            return None 
    elif method == "flexible":
        # 但是这个flexible匹配明明已经是宽松版本了，但还是匹配不到——可能还是训废了？        
        return solution_str


def extract_math(method='strict'):
    accs = []
    answers = json.load(open(file_name))
    for i in tqdm.trange(len(math['test'])):  # len(math['test'])
        # 调整prompt内容，之前的格式不太对劲
        a = answers[i]
        solution = extract_solution(a, method)
        # print(solution)
        # print(type(ground_truth))
        # 之前发现可能表达式不同但实际上是一个数值的情况，比如100.00和100，然后可能有多余的
        # 先不考虑这个情况
        ground_truth = math['test']['reward_model'][i]['ground_truth']
        # print(solution)
        correct = verify(parse(solution), parse(ground_truth))
        if solution is None:
            accs.append(None)
        elif correct:
            accs.append(1)
        else:
            # print(solution, ground_truth)
            accs.append(0)
    return accs, answers


path0 = f'{root_path}/reasoning'
math = datasets.load_dataset("parquet", 
              data_files={'train': path0 + '/gsm8k/train.parquet', 'test': path0 + '/gsm8k/test.parquet'})
# print(math['test']['prompt'][0])

# exit(0)
accs, answers = extract_math(mode)
acc0 = accs.count(1) / len(accs)
print('total acc:', acc0)
print('invalid output:', accs.count(None))
