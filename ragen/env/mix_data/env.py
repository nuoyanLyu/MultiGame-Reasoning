import numpy as np
from typing import List, Optional, Tuple, Dict
import re
from math_verify import parse, verify

from openai import NotFoundError

from ragen.env.base import BaseLanguageBasedEnv, EnvPlayer, seed_everything, timed
from .config import MixEnvConfig
import gymnasium as gym
import datasets
import random
import os

class MixEnv(BaseLanguageBasedEnv, gym.Env):
    """
    A Math game environment.
    Inherits from LLMGameEnv and implements the game-specific logic.
    """
    def __init__(self, config=None):
        BaseLanguageBasedEnv.__init__(self)
        self.config = config if config is not None else MixEnvConfig()
        # self.ACTION_SPACE = gym.spaces.discrete.Discrete(2, start=self.config.action_space_start)
        self.render_cache = None
        self.seed = int(self.config.seed)
        self.render_mode = self.config.render_mode
        assert self.render_mode == 'text'
        data_path = self.config.data_path
        data_files = self.config.data_files
        self.data_path = f"{data_path}/reasoning/{data_files}"
        # NOTE：这么多数据集能否支持同时load多个环境的设置？
        # 如果不行可能需要每一次加载环境的时候需要通过stream的方式读取数据
        self.mode = self.config.mode
        self.data_train =  datasets.load_dataset("parquet", 
            data_files=f"{self.data_path}/{self.mode}.parquet")['train']
        # test好像没办法加载……先这么放着
        self.question0 = None
        self.history = []
        self.reset(self.seed)

    def reset(self, seed=None, **kwargs):
        """Initializes the environment for a new question"""
        self.seed = seed
        seed_everything(seed)
        # 根据随机数种子初始化抽取一个问题
        self.question0 = random.choice(self.data_train)

    def render(self) -> str:
        # 和之前的get_state_prompt相同的作用 生成当前棋局的信息
        assert self.question0 is not None, "question0 is None, please reset the environment first"
        prompt = self.question0['question']
        return prompt

    def render_for_test(self) -> str:
        prompt = self.question0['question']
        prompt += f'\nAnswer is: {self.question0["answer"]}'
        return prompt

    def _extract_choice(self, text: str):
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

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        Execute one step in the environment.
        In MixEnv, the action is the answer to the question.
        NOTE should also handle predefined invalid action (0)
        Args:
            action: Action to take, must be in action space, or default invalid action
        Returns:
            observation, reward, done, info
            observation: updated game prompt;
            info: dictionary
        """
        # 数学题没必要再问一次，直接设置max_turn数量为1
        # 直接对比action和答案是否相同
        type = self.question0['type']
        if type == 'math':
            ground_truth = parse(self.question0['answer'])
            answer = parse(action)
            valid = not (len(answer) == 0)  # 如果没有匹配到数字，输出的是空列表
            success = verify(answer, ground_truth)
            reward = int(success)
            prompt = 'Answer is correct!' if success else 'Answer is incorrect.'
            done = True
        else:
            # 判定选项是否正确，答案为ABCD其中的一个            
            answer = self._extract_choice(action)
            if answer is None or answer not in ['A', 'B', 'C', 'D']:
                success = False
                valid = False
            else:
                success = answer == self.question0['answer']
                valid = True
            reward = int(success)
            prompt = 'Answer is correct!' if success else 'Answer is incorrect.'
            done = True
        info = {'action_is_effective': success, 'action_is_valid': valid, 'success': success}
        self.history.append({
            "action": answer,
        })
        return (prompt, reward, done, info)

    def close():
        # 清空数据相关的变量信息
        self.data_train = None
        self.question0 = None


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    for key in ["http_proxy", "https_proxy", "all_proxy", 
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
        os.environ.pop(key, None)
    config = MixEnvConfig()
    env = MixEnv(config)
    # print(env.reset(seed=42))
    while 1:
        env.reset(random.randint(0, 1000))
        print(env.render_for_test())
        keyboard = input("Enter answer: ")
        if keyboard == 'q':
            break
        # action = int(keyboard)
        # assert action in env.ACTION_LOOKUP, f"Invalid action: {action}"
        obs, reward, done, info = env.step(keyboard)
        for o in [obs, reward, done, info]:
            print(o)
