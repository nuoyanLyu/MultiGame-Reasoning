from ragen.env.tictactoe.env import TicTacToeEnv
from tqdm import trange
import os
from openai import OpenAI
import random
import re
from collections import Counter
from verl.utils import hf_tokenizer
tokenizer = hf_tokenizer("/root/autodl-tmp/tictactoe/grpo/game_220")

# for key in ["http_proxy", "https_proxy", "all_proxy", 
#             "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
#     os.environ.pop(key, None)


root_path = "/data1/lvnuoyan/llm_model"
tokenizer = hf_tokenizer(f"{root_path}/tictactoe/grpo/game_220")

# 配置 OpenAI 客户端（兼容 vLLM 的 OpenAPI 接口）
client = OpenAI(
    api_key="EMPTY",  # vLLM 无需认证密钥，任意字符串均可
    base_url=f"http://localhost:2220/v1"  # 与 vLLM 服务端口一致 3333没人用之后我都用这个端口吧
)

model_path = '/root/autodl-tmp'
def trainer_output(text: str) -> str:
    message = [{"role": "system", "content": "You're a helpful assistant. "},
               {"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
    prompt += "<think>"
    try:
        response = client.completions.create(
            model=f"{root_path}/tictactoe/grpo/game_220",
            prompt=prompt,
            max_tokens=600,
            temperature=0.5,
        )
        # print(response.choices[0].text)
        return response.choices[0].text
    except Exception as e:
        raise RuntimeError(f"API 调用失败：{str(e)}")


prompt0 = f"""Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: 200 words (tokens). <think>"""

# 自动进行初始测试流程，测试自己-base模型胜率如何
# 初始化环境信息
env = TicTacToeEnv()
# 游戏进行100次测试胜率
info_list = []
for t in trange(100):
    # 初始化环境，设置不同的seed信息
    env.reset(seed=random.randint(1, 1000))
    # 游戏进行
    prompt = env.render()
    while True:
        # print(prompt + prompt0)
        # 得到trainer的行动
        output = trainer_output(prompt + prompt0)
        # print(output[:200])
        output = '<think>' + output
        print(output)
        pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>'
        match = re.search(pattern, output, re.DOTALL)
        if not match:
            info_list.append('trainer-invalid-output')
            print(output[:200])
            print('trainer-invalid-output')
            break
        action = match.group(2)
        # 更新环境信息，得到对手操作以及下一步信息
        prompt, reward, done, info = env.step(action)
        # 检查游戏是否结束
        if done:
            # 如果结束检查游戏状态，添加对应的状态信息
            if "Invalid action:" in prompt:
                info_list.append('trainer-invalid-output')
                print('trainer-invalid-output')
            elif "Congratulations!" in prompt:
                info_list.append('success')
                print('success')
            elif "Draw!" in prompt:
                info_list.append('draw')
                print('draw')
            # invalid: env_player不符合指令遵循
            # wrong：env_player的动作不在available_action中
            elif "Your opponent made a mistake" in prompt:
                info_list.append('env_player-invalid-output')
                print('env_player-invalid-output')
            elif "Your opponent action is wrong" in prompt:
                info_list.append('env_player-wrong-output')
                print('env_player-wrong-output')
            elif "Failed! " in prompt:
                info_list.append('fail')
                print('fail')
            break

# 统计info_list中出现的不同情况的次数并计算对应的比例
counter = Counter(info_list)
total = len(info_list)
for key, value in counter.items():
    print(f"{key}: {value / total:.2%}")
 