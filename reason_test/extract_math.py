# 根据infer得到的结果，运行函数提取模型生成的信息
import json
import re
import datasets
import tqdm
from math_verify import parse, verify

root_path = '/data1/lvnuoyan' # '/root/autodl-tmp'# 
file_name = 'game100-gsm8k-09-23-17-39.json'
# file_name = 'Qwen2.5-1.5B-Instruct-gsm8k-09-23-17-44.json'

def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]
    if method == "strict":
        # this also tests the formatting of the model
        # 更精确的正则表达式
        pattern = r"<answer>(-?\d+\.?\d*)</answer>"
        # 使用 re.search() 提取最后一个数字
        match = re.search(pattern, solution_str, re.DOTALL)
        if match:
            last_number = match.group(1)  # 提取匹配到的最后一个数字
            return last_number
        else:
            return None 
    elif method == "flexible":
        # 但是这个flexible匹配明明已经是宽松版本了，但还是匹配不到——可能还是训废了？        
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        # print(answer)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            r = 1
            invalid_str = ["", ".", ',']
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    r = 0
                    break
            if r:
                return None
            # 删除多余的逗号、末尾的多余字符（可能会匹配到末尾的小数点）
            while not final_answer[-1].isdigit():
                final_answer = final_answer[:-1]
            final_answer = final_answer.replace(',', '')
    return final_answer


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
        if solution is None:
            accs.append(None)
        elif float(solution) == float(ground_truth):
            accs.append(1)
        else:
            print(solution, ground_truth)
            accs.append(0)
    return accs, answers


path0 = f'{root_path}/reasoning'
math = datasets.load_dataset("parquet", 
              data_files={'train': path0 + '/gsm8k/train.parquet', 'test': path0 + '/gsm8k/test.parquet'})
# print(math['test']['prompt'][0])

# exit(0)
accs, answers = extract_math('flexible')
acc0 = accs.count(1) / len(accs)
print('total acc:', acc0)
print('invalid output:', accs.count(None))
