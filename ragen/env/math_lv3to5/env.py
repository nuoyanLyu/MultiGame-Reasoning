import numpy as np
from typing import List, Optional, Tuple, Dict
import re
from math_verify import parse, verify

from openai import NotFoundError

from ragen.env.base import BaseLanguageBasedEnv, EnvPlayer, seed_everything, timed
from .config import MathEnvConfig
import gymnasium as gym
import datasets
import random
import os

class MathEnv(BaseLanguageBasedEnv, gym.Env):
    """
    A Math game environment.
    Inherits from LLMGameEnv and implements the game-specific logic.
    """
    def __init__(self, config=None):
        BaseLanguageBasedEnv.__init__(self)
        self.config = config if config is not None else MathEnvConfig()
        # self.ACTION_SPACE = gym.spaces.discrete.Discrete(2, start=self.config.action_space_start)
        self.render_cache = None
        self.seed = int(self.config.seed)
        self.render_mode = self.config.render_mode
        assert self.render_mode == 'text'
        self.max_env_try = self.config.max_env_try
        data_path = self.config.data_path
        data_files = self.config.data_files
        self.data_path = f"{data_path}/reasoning/{data_files}"
        # NOTE：这么多数据集能否支持同时load多个环境的设置？
        # 如果不行可能需要每一次加载环境的时候需要通过stream的方式读取数据
        self.mode = self.config.mode
        self.data_train =  datasets.load_dataset("parquet", 
            data_files=f"{self.data_path}/{self.mode}.parquet")["train"]
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
        prompt = self.question0['extra_info']['question']
        return prompt

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        Execute one step in the environment.
        In MathEnv, the action is the answer to the question.
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
        ground_truth = parse(self.question0['extra_info']['answer'])
        # 提取答案中的数字部分，参考verl中的gsm8k代码设置，匹配最后一位数字并删除多余的字符
        # match = re.search("(\\-?[0-9\\.\\,]+)", action)
        # ground_truth中包含大量特殊字符，全部交给math_verify处理，不用过多考虑
        answer = parse(action)
        success = verify(answer, ground_truth)
        reward = int(success)
        prompt = 'Answer is correct!' if success else 'Answer is incorrect.'
        done = True
        info = {'action_is_effective': success, 'action_is_valid': success, 'success': success}
        self.history.append({
            "action": answer
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
    config = MathEnvConfig()
    env = MathEnv(config)
    # print(env.reset(seed=42))
    while 1:
        env.reset(random.randint(0, 1000))
        print(env.render())
        keyboard = input("Enter answer: ")
        if keyboard == 'q':
            break
        # action = int(keyboard)
        # assert action in env.ACTION_LOOKUP, f"Invalid action: {action}"
        obs, reward, done, info = env.step(keyboard)
        for o in [obs, reward, done, info]:
            print(o)
