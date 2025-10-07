# 根据infer得到的结果，运行函数提取模型生成的信息
import json
import re
import time
import datasets
import tqdm
from math_verify import parse, verify
import os


# file_name = 'game100-gsm8k-09-23-17-39.json'
root_path = '/root/autodl-tmp'
file_name = 'tictactoe-math-game20-10-01-19-00.json'
# file_name = 'Qwen2.5-1.5B-Instruct-gsm8k-09-23-17-44.json'
# game100不使用原始prompt17-39： 0.4025, strict 0.3161
# Game100使用原始prompt21-55：0.3321——所以改了prompt反而效果更差，，strict 0.2009


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


def extract_math(answers, math):
    accs_strict = []
    accs_flex = []
    for i in tqdm.trange(len(math['test'])):  # len(math['test'])
        # 调整prompt内容，之前的格式不太对劲，导致模型输出的最后一个数字不是最后一个数字
        answer = answers[i]
        solu_strict, solu_flex = extract_solution(answer)
        # answers.append(outputs[0].outputs[0].text)
        # print(outputs[0].outputs[0].text)
        # ground_truth = parse(math[i]['extra_info']['answer'])
        ground_truth = parse(math['test']['reward_model'][i]['ground_truth'])
        # print(answers[0])
        # print(solu_strict)
        # print(solu_flex)
        # print(ground_truth)
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
        # print(accs_strict[0], accs_flex[0])
        # exit(0)
    return accs_strict, accs_flex, answers

if __name__ == '__main__':
    path0 = f'{root_path}/reasoning'
    math = datasets.load_dataset("parquet", 
                  data_files={'train': path0 + '/gsm8k/train.parquet', 'test': path0 + '/gsm8k/test.parquet'})
    # print(math['test']['prompt'][0])
    # math = datasets.load_dataset('parquet', 
    #                 data_files=f'{path0}/SimpleRL-Zoo-Data/simplelr_abel_level3to5/test.parquet')['train']
    # # exit(0)
    with open(file_name, 'r') as f:
        answers = json.load(f)
    accs_strict, accs_flex, answers = extract_math(answers, math)
    acc0 = accs_strict.count(1) / len(accs_strict)
    print('----------strict mode------------')
    print('total acc:', acc0)
    print('invalid output:', accs_strict.count(None))
    print('----------flexible mode----------')
    acc0 = accs_flex.count(1) / len(accs_flex)
    print('total acc:', acc0)


