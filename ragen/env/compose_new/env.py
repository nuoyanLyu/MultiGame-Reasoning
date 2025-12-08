from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import nashpy as nash  
import random
import gymnasium as gym
from .config import ComposeNewConfig
from ragen.env.base import BaseLanguageBasedEnv, seed_everything, timed
from ragen.env.env_factory import create_success_info, clean_success_info

class ComposeNewEnv(BaseLanguageBasedEnv, gym.Env):
    def __init__(self, config: Optional[ComposeNewConfig] = None):
        BaseLanguageBasedEnv.__init__(self) 
        self.config = config or ComposeNewConfig()
        self.seed = int(self.config.seed)
        self.render_mode = self.config.render_mode
        assert self.render_mode == 'text'
        base_envs = self.config.base_env_list
        self.env_names = base_envs
        mode = self.config.mode
        # 初始化训练阶段
        self.phase = None
        # 初始化嵌套的子env
        self.sub_envs = []
        self.env_success_info = []
        self.sub_rewards = []
        self.sub_success = []
        self.reward_improve = self.config.reward_improve
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
            self.env_success_info.append(create_success_info(env_name, self.config.k))
        self.reset(self.seed)
    
    def reset(self, seed, **kwargs):
        seed_everything(seed)
        for env in self.sub_envs:
            env.reset(seed, **kwargs)
        # 不再随机生成任务顺序——担心最开始不好收敛
        # 初始化任务阶段
        self.phase = 0
        # 初始化子任务成功率
        self.sub_rewards = []
        self.sub_success = []
        # 成功率不清零从而记录之前的历史信息

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
        if self.phase != len(self.sub_envs) - 1:
            # 子任务并没有全部完成，但是需要记录当前子任务的成功情况
            # 记录 方便后续返回info字段的成功率信息
            self.sub_success.append(info['success'])
            # 更新子任务成功率
            success_rate = self.env_success_info[self.phase].update(info['success'])
            if self.reward_improve and info['success']:
                reward = 1 - success_rate
            # 记录子任务reward 
            self.sub_rewards.append(reward)
            # 进入下一个子任务
            self.phase += 1
            prompt = self.render()
            # 这里的reward相当于step reward，设置为0没有问题
            reward = 0
            done = False
            return prompt, reward, done, info
        # 同样更新成功率信息字典
        self.sub_success.append(info['success'])
        success_rate = self.env_success_info[self.phase].update(info['success'])
        if self.reward_improve and info['success']:
            reward = 1 - success_rate
        self.sub_rewards.append(reward)
        # 计算总体的reward以及info等信息
        # 这个地方的设计可能不靠谱——都对应该设置为1，只有一个对才应该是一半的奖励
        # 求平均值对两个都作对的奖励可能不够？整体reward都偏小了
        reward = sum(self.sub_rewards) / len(self.sub_rewards)
        prompt = self.render()
        done = True
        if False not in self.sub_success:
            success = True
        else:
            success = False
        info = dict(
            action_is_valid=True,
            action_is_effective=True,
            success=success,
        )
        # TODO：验证是否可以在所有步骤全结束之后再记录而不是每一次都返回？
        if self.reward_improve:
            # 记录任务成功率方便查看模型情况
            for i in range(len(self.env_names)):
                info[f'{self.env_names[i]}_success_rate'] = self.env_success_info[i].get()
        return prompt, reward, done, info

    def close(self):
        for env in self.sub_envs:
            env.close()
        self.sub_envs = None
        for env_name in self.env_names:
            clean_success_info(env_name)


    
if __name__ == '__main__':
    config = ComposeNewConfig()
    env = ComposeNewEnv(config)
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
        