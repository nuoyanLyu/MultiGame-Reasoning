from datasets import load_dataset
import random
import pandas as pd
import os

root_path = '/root/autodl-tmp/reasoning'
dir_name = 'mmlu_format'

# --- MMLU 字段格式化 ---
def format_mmlu_item(example):
    # 格式化 question + 选项
    formatted_question = (
        example["question"].strip() + "\n"
        + f"A. {example['choices'][0]}\n"
        + f"B. {example['choices'][1]}\n"
        + f"C. {example['choices'][2]}\n"
        + f"D. {example['choices'][3]}\n"
    )

    # 将数字答案（0-3）转换为字母（A-D）
    num_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
    formatted_answer = num_to_letter.get(example["answer"], "")
    # 直接用原始字段不行，
    # huggingface dataset对象是强类型的，
    # 每个字段在 schema（即 Arrow table schema）中都有固定类型
    # 类型相同可以覆盖，类型不同——比如这里的answer从int类型变成了str类型就无法覆盖
    # 新命名一个字段
    return {
        "formatted_question": formatted_question,
        "formatted_answer": formatted_answer
    }


# 加载 MMLU 数据集，全部数据集all、对应训练数据字段auxiliary_train
mmlu = load_dataset(f"{root_path}/mmlu/", 'all')['auxiliary_train']

# 随机抽取test数据各100条
test_mmlu = load_dataset(f"{root_path}/mmlu/", 'all')['test']

# 字段统一
# MMLU：格式化
# 对 mmlu_sample 应用格式化函数
mmlu_formatted = mmlu.map(format_mmlu_item)
# 转换为 DataFrame
mmlu_df = pd.DataFrame({
    "question": mmlu_formatted["formatted_question"],
    "answer": mmlu_formatted["formatted_answer"],
    "type": "mmlu"
})

mmlu_test_form = test_mmlu.map(format_mmlu_item)
mmlu_test_df = pd.DataFrame({
    "question": mmlu_test_form["formatted_question"],
    "answer": mmlu_test_form["formatted_answer"],
    "type": "mmlu"
})

# 如果文件夹不存在则创建
os.makedirs(f"{root_path}/{dir_name}", exist_ok=True)

# 保存为train.parquet文件
output_path = f"{root_path}/{dir_name}/train.parquet"
mmlu_df.to_parquet(output_path, index=False)
# 保存对应的test文件
output_test_path = f"{root_path}/{dir_name}/test.parquet"
mmlu_test_df.to_parquet(output_test_path, index=False)



print(f"已成功生成合并后的train文件：{output_path}")
print(mmlu_df.head())
print(f"已成功生成合并后的test文件：{output_test_path}")
print(mmlu_test_df.head())
