from ragen.env.tictactoe.env import TicTacToeEnv
from tqdm import trange
import os
from openai import OpenAI
import random
import re
from collections import Counter
from verl.utils import hf_tokenizer


# root_path = "/data1/lvnuoyan/llm_model"
root_path = '/root/autodl-tmp'
tokenizer = hf_tokenizer(f"{root_path}/tictactoe-gemini/game100")

# 配置 OpenAI 客户端（兼容 vLLM 的 OpenAPI 接口）
client = OpenAI(
    api_key="EMPTY",  # vLLM 无需认证密钥，任意字符串均可
    base_url=f"http://localhost:2100/v1"  # 与 vLLM 服务端口一致 3333没人用之后我都用这个端口吧
)

# model_path = '/data1/lvnuoyan/llm_model'
model_path = '/root/autodl-tmp'
def trainer_output(text: str) -> str:
    message = [{"role": "system", "content": "You're a helpful assistant. "},
               {"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
    try:
        response = client.completions.create(
            model=f"{root_path}/tictactoe-gemini/game100",
            prompt=prompt,
            max_tokens=600,
            temperature=0.5,
        )
        # print(response.choices[0].text)
        return response.choices[0].text
    except Exception as e:
        raise RuntimeError(f"API 调用失败：{str(e)}")

init_prompt = f"""
## Game Rules: TicTacToe

**Objective**: Be the first player to connect 3 of your pieces in a continuous line.

**Player Pieces**:
- Player 1: 'O'
- Player 2: 'X'
- Empty Slot: '.'

**How to Play**:
1. The game is played on a 3x3 vertical grid.
2. Players take turns dropping one of their pieces into any available column from the top.
3. The piece falls to the lowest unoccupied slot in that column.

**Winning Conditions**:
The game ends when a player forms a line of 3 of their own pieces. The line can be:

1.  **Horizontal** (side-by-side in a row)
    *Example of a horizontal win for Player 1 ('O'):*
    ```
    X . . 
    O O O   <-- 3 'O's in row 2
    . X . 
    ```

2.  **Vertical** (stacked on top of each other in a column)
    *Example of a vertical win for Player 2 ('X'):*
    ```
    . X O 
    O X O   <-- 3 'X's in column 2
    . X . 
    ```

3.  **Diagonal** (connected at an angle)
    *Example of a diagonal win (bottom-left to top-right) for Player 1:*
    ```
    . . O 
    . O X   <-- 3 'O's in a diagonal line
    O X . 
    ```
    *Example of another diagonal win (top-left to bottom-right) for Player 2:*
    ```
    X . O 
    . X O   <-- 3 'X's in a diagonal line
    . O X 
    ```

**Draw Condition**:
If the entire grid is filled with pieces and no player has won, the game is a draw.

"""

# prompt0 = f"""Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: 200 words (tokens). """
prompt0 = f"""Let\'s think step by step and always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: 200 words (tokens). """

# 自动进行初始测试流程，测试自己-base模型胜率如何
# 初始化环境信息
env = TicTacToeEnv()
# 游戏进行100次测试胜率
info_list = []
for t in trange(100):
    # 初始化环境，设置不同的seed信息
    # 使用reset0，初始空棋盘，随机选取先后手
    env.reset0(seed=random.randint(1, 1000))
    # 游戏进行
    prompt = env.render()
    while True:
        # print(prompt + prompt0)
        # 得到trainer的行动
        output = trainer_output(init_prompt + prompt + prompt0)
        # print(output[:200])
        # print(output)
        pattern = r'.*<think>(.*?)</think>\s*<answer>(.*?)</answer>$'
        match = re.match(pattern, output, re.DOTALL)
        if not match:
            info_list.append('trainer-invalid-output')
            print(output[:200])
            print('trainer-invalid-output')
            break
        action = match.group(2)
        # 更新环境信息，得到对手操作以及下一步信息
        prompt, reward, done, info = env.step(action)
        # 检查游戏是否结束，更改了invalid——这里设置invalid为游戏失败、直接结束
        if "Invalid action:" in prompt:
            info_list.append('trainer-invalid-output')
            print('trainer-invalid-output')
            break
        if done:
            # 如果结束检查游戏状态，添加对应的状态信息
            if "Congratulations!" in prompt:
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
 