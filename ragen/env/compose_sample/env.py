from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import nashpy as nash  
import random
import gymnasium as gym
from .config import ComposeSampleConfig
from ragen.env.base import BaseLanguageBasedEnv, seed_everything, timed
from ragen.env.env_factory import create_success_info, clean_success_info


train_counting = {}


def train_count(k):
    if k not in train_counting.keys():
        print('add train count for', k)
        train_counting[k] = 0
    train_counting[k] += 1

def get_count(k):
    if k not in train_counting.keys():
        return 0
    return train_counting[k]


class ComposeSampleEnv(BaseLanguageBasedEnv, gym.Env):
    def __init__(self, config: Optional[ComposeSampleConfig] = None):
        BaseLanguageBasedEnv.__init__(self) 
        self.config = config or ComposeSampleConfig()
        self.seed = int(self.config.seed)
        self.render_mode = self.config.render_mode
        assert self.render_mode == 'text'
        base_env0 = self.config.env0
        self.sample_env_names = self.config.sample_env_list
        self.sample_method = self.config.sample_method
        mode = self.config.mode
        # 初始化训练阶段
        self.phase = None
        # 初始化嵌套的子env
        self.sub_envs = []
        self.success_info = []
        self.sub_rewards = []
        self.sub_success = []
        # 存储各个任务的难度信息——使用这个采样子任务以及计算reward
        self.diff_list = []
        self.k = self.config.k
        # 延迟导入，避免循环依赖
        from ragen.env import REGISTERED_ENV_CONFIGS, REGISTERED_ENVS
        config0 = REGISTERED_ENV_CONFIGS[base_env0]
        if hasattr(config0, 'mode'):
            env_config0 = config0(mode=mode)
        else:
            env_config0 = config0()
        self.env0 = REGISTERED_ENVS[base_env0](config=env_config0)
        # 初始化用于训练的env1对应的id
        self.env1_id = None
        for i in range(len(self.sample_env_names)):
            env_name = self.sample_env_names[i]
            config1 = REGISTERED_ENV_CONFIGS[env_name]
            if self.config.player_nums[i]:
                player_info = self.config.player_infos[i]
                env_config = config1(player_info=player_info)
            else:
                env_config = config1()
            env = REGISTERED_ENVS[env_name](config=env_config)
            self.sub_envs.append(env)
            self.success_info.append(create_success_info(env_name))
        # self.reset(self.seed)
    
    def reset(self, seed, **kwargs):
        seed_everything(seed)
        # 重置env0
        self.env0.reset(seed, **kwargs)
        for env in self.sub_envs:
            env.reset(seed, **kwargs)
        self.phase = 0
        # 初始化子任务成功率
        self.sub_rewards = []
        self.sub_success = []
        # 采样，抽取用于嵌套的第二个任务
        if self.sample_method == 'random':
            self.env1 = random.choice(self.sub_envs)
        elif self.sample_method == 'difficulty-square':
            # 计算每个任务的难度，为 1 - 成功率
            self.diff_list = [1 - success.get() for success in self.success_info]
            print(self.diff_list)
            # 难度的平方设置为对应子环境被采样的概率
            p_list = [d**2 for d in self.diff_list]
            # 归一化概率
            p_list = [p / sum(p_list) for p in p_list]
            print(p_list)
            # 进行采样，验证了这里就是概率值
            task_id = np.random.choice(len(self.diff_list), p=p_list)
            print('set task', task_id)
            self.env1_id = task_id
        else:
            print('[ERROR] sample method not implemented!')
            exit(1)

    def render(self) -> str:
        if self.phase == 0:
            return self.env0.render()
        else:
            return self.sub_envs[self.env1_id].render()

    def step(self, action):
        # 找到当前step对应环境并调用对应的step函数
        if self.phase == 0:
            env = self.env0
        else:
            env = self.sub_envs[self.env1_id]
        prompt, reward, done, info = env.step(action)
        # 子任务未结束、继续子任务
        if not done:
            return prompt, reward, done, info
        # 完成，判断是否全部完成
        if self.phase == 0:
            # 子任务并没有全部完成，但是需要记录当前子任务的成功情况
            # 记录 方便后续返回info字段的成功率信息
            self.sub_success.append(info['success'])
            # # 更新子任务成功率
            # self.env_success_info[self.phase].update(info['success'])
            # if self.reward_improve and info['success']:
            #     reward = 1 - success_rate
            # # 记录子任务reward 
            self.sub_rewards.append(reward)
            # 进入下一个子任务
            self.phase += 1
            prompt = self.render()
            # 这里的reward相当于step reward，设置为0没有问题
            reward = 0
            done = False
            return prompt, reward, done, info
        # 两个阶段任务都已完成，记录成功率、根据难度计算reward信息
        self.sub_success.append(info['success'])
        # 记录子任务成功率
        self.success_info[self.env1_id].record(info['success'])
        # 计数，统计是否到更新参数的阶段，如果到了预设值则更新成功率信息
        train_count('compose')
        if get_count('compose') % self.k == 0:
            for s in self.success_info:
                s.update()
        # 计算第二个任务的reward信息
        difficulty = self.diff_list[self.env1_id] / max(self.diff_list)
        reward = difficulty * reward
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
        # 验证是否可以在所有步骤全结束之后再记录而不是每一次都返回？OK的
        # 记录任务成功率方便查看模型情况
        for i in range(len(self.sample_env_names)):
            info[f'{self.sample_env_names[i]}_success_rate'] = 1 - self.diff_list[i]
        # 被采样到的game计数，方便查看采样情况
        info[f'{self.sample_env_names[self.env1_id]}_count'] = 1
        return prompt, reward, done, info

    def close(self):
        for env in self.sub_envs:
            env.close()
        self.sub_envs = None
        for env_name in self.sample_env_names:
            clean_success_info(env_name)
        self.env0.close()


    
if __name__ == '__main__':
    config = ComposeSampleConfig()
    env = ComposeSampleEnv(config)
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
        