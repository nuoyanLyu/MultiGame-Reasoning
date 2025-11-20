from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import nashpy as nash  
import random
import gymnasium as gym
from .config import ComposeConfig
from ragen.env.base import BaseLanguageBasedEnv, seed_everything, timed


class ComposeEnv(BaseLanguageBasedEnv, gym.Env):
    def __init__(self, config: Optional[ComposeConfig] = None):
        BaseLanguageBasedEnv.__init__(self) 
        self.config = config or ComposeConfig()
        self.seed = int(self.config.seed)
        self.render_mode = self.config.render_mode
        assert self.render_mode == 'text'
        base_envs = self.config.base_env_list
        mode = self.config.mode
        self.reward0 = self.config.reward0
        # 初始化训练阶段
        self.phase = None
        # 初始化嵌套的子env
        self.sub_envs = []
        # 延迟导入，避免循环依赖
        from ragen.env import REGISTERED_ENV_CONFIGS, REGISTERED_ENVS
        
        for env_name in base_envs:
            Config = REGISTERED_ENV_CONFIGS[env_name]
            if hasattr(Config, 'mode'):
                env_config = Config(mode=mode)
            else:
                env_config = Config()
            env = REGISTERED_ENVS[env_name](config=env_config)
            self.sub_envs.append(env)
        self.reset(self.seed)
    
    def reset(self, seed, **kwargs):
        seed_everything(seed)
        for env in self.sub_envs:
            env.reset(seed, **kwargs)
        # 不再随机生成任务顺序——担心最开始不好收敛
        # 初始化任务阶段
        self.phase = 0

    def render(self) -> str:
        env = self.sub_envs[self.phase]
        return env.render()

    def step(self, action):
        # 找到当前step对应环境并调用对应的step函数
        env = self.sub_envs[self.phase]
        prompt, reward, done, info = env.step(action)
        # 子任务未结束、继续子任务
        if not done:
            return prompt, reward, done, info
        # 子任务结束，如果为False、无法进入下一阶段，计算reward
        if not info['success']:
            reward = self.reward0 * self.phase
            done = True
            print(f'Sub task failed. Reward: {reward}')
            return prompt, reward, done, info
        # 子任务为True，进入下一阶段或直接返回
        if self.phase == len(self.sub_envs) - 1:
            done = True
            print('All task done!')
            return prompt, reward, done, info
        # 进入下一阶段
        self.phase += 1
        prompt = self.render()
        reward = 0
        done = False
        info = dict(
            action_is_valid=True,
            action_is_effective=True,
            success=True
        )
        return prompt, reward, done, info

    def close(self):
        for env in self.sub_envs:
            env.close()
        self.sub_envs = None


    
if __name__ == '__main__':
    config = ComposeConfig()
    env = ComposeEnv(config)
    r = 1
    while r:
        env.reset(random.randint(0, 1000))
        done = False
        while not done:
            print(env.render())
            print('game', r)
            action = input('Input action: ')
            if action == 'q':
                r = -1
                break
            prompt, reward, done, info = env.step(action)
            print(prompt, reward, done, info)
        r += 1
        