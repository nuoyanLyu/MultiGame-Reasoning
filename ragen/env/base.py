from abc import ABC, abstractmethod
import re
from typing import Optional, List, Tuple, Any, Dict
from openai import OpenAI
import os
import random 
import numpy as np

import time, logging, functools, itertools
import json
import threading

# add time track part
def timed(label):
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*args, **kw):
            t0 = time.monotonic()
            try:
                return fn(*args, **kw)
            finally:
                dt = time.monotonic() - t0
                print(f"[TIMER] {label} took {dt:.3f}s")
        return wrap
    return deco



class ThreadSafeCycle:
    def __init__(self, iterable):
        self._lock = threading.Lock()
        self._iterator = itertools.cycle(iterable)

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            return next(self._iterator)


# 环境测试命令：
# python -m ragen.env.connect4.env

deepseek_keys = json.load(open('ragen/env/api-keys.json'))['deepseek']
# deepseek_key = os.environ.get('DEEPSEEK_KEY')
if not deepseek_keys:
    print('No deepseek keys found, please set it in the environment variable DEEPSEEK_KEY')
    exit(1)
deepseek_keys = ThreadSafeCycle(deepseek_keys)
# open_router_key = os.environ.get('OPEN_ROUTER_KEY')
# if not open_router_key:
#     print('No open router key found, please set it in the environment variable OPEN_ROUTER_KEY')
#     exit(1)
# model_name = 'Qwen3-8B-Base'
# PORT = '38388'


class BaseEnv(ABC):
    """
    Abstract base class for all environments.
    The class needs to handle text-based input, input may be invalid
        - Environment will track the total reward for the trajectory

    """
    def __init__(self):
        pass

    @abstractmethod
    def reset(self, seed=None, **kwargs) -> Any:
        """
        Reset the environment.
        NOTE: the environment should be same for the same seed
        Returns:
            rendered environment
        """
        pass

    @abstractmethod
    def step(self, action) -> Tuple[Any, float, bool, Dict]:
        """
        Execute one step in the environment.
        NOTE should also handle predefined invalid action (0)
        Args:
            action: Action to take, must be in action space, or default invalid action
            
        Returns:
            observation (rendered environment), reward, done, info
        """
        pass

    # below are optional methods

    def render(self, mode: str = 'text') -> Any:
        """Render the environment. Optional method."""
        pass

    def compute_reward(self, action, **kwargs) -> float:
        """Compute reward for the action."""
        pass

    def close(self):
        """Close the environment."""
        pass


class BaseDiscreteActionEnv(BaseEnv, ABC):
    """
    Abstract base class for environments with discrete action spaces
    This class provides common functionality for environments like FrozenLakeEnv and SokobanEnv.
    """

    @abstractmethod
    def step(self, action: int) -> Tuple[Any, float, bool, Dict]:
        """
        Execute one step in the environment.
        Args:
            action: Action to take, must be in action space, or default invalid action
        Returns:
            observation (rendered environment), reward, done, info
        """
        pass

    @abstractmethod
    def get_all_actions(self) -> List[int]:
        """Get list of all valid actions."""
        pass


class BaseLanguageBasedEnv(BaseEnv, ABC):
    """
    Abstract base class for environments with language-based action space environment
    This class provides common functionality for environments like countdown from TinyZero
    """

    @abstractmethod
    def step(self, action: str) -> Tuple[Any, float, bool, Dict]:
        """
        Execute one step in the environment.
        Args:
            action: Action to take, must be in action space, or default invalid action
        Returns:
            observation (rendered environment), reward, done, info
        """
        pass


class BaseEnvConfig(ABC):
    """
    Abstract base class for environment configurations.
    """
    def __init__(self):
        self.invalid_act = ""
        self.invalid_act_score = 0

'''
class EnvPlayer():
    """_summary_
    在双人或多人玩家场景下初始化环境中的其他玩家
    """
    def __init__(self, num_players, player_info, system_prompt='', temperature=0.5) -> None:
        self.system_prompt = system_prompt
        self.temperature = temperature
        # 根据游戏中玩家数量，初始化除了训练模型之外的其他模型
        self.num_players = num_players
        # 根据player info初始化这些其他的player信息
        # print(player_info)
        assert self.num_players == len(player_info) + 1
        self.players = []
        for i in range(self.num_players - 1):
            model_name = player_info[i]['model_name']
            if model_name == 'deepseek':
                client = OpenAI(api_key=next(deepseek_keys),
                                base_url='https://api.deepseek.com')
                model = 'deepseek-chat'
            elif 'Qwen' in model_name:
                port = player_info[i]['port']
                client = OpenAI(
                    api_key="EMPTY",  # vLLM 无需认证密钥，任意字符串均可
                    base_url=f"http://localhost:{port}/v1"  # 与 vLLM 服务端口一致 3333没人用之后我都用这个端口吧
                )
                model = f"/data1/lvnuoyan/llm_model/{model_name}"
            else:
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=open_router_key,
                )
                model = model_name            
            self.players.append({'client':client, 'model':model})
            # print(f'Set Player {i} with model {model}')
    
    # @timed('env_player_act')
    def act(self, message, player_id):
        """
        调用API、获得指定玩家回复
        Args:
            system_prompt: 系统提示
            message: 用户消息
            player_id: 玩家ID，注意这里是除了训练模型外的其他agent的ID信息，
            如0代表除了玩家外第一个Player
        Returns:
            API返回的消息
        """
        assert player_id < self.num_players - 1
        if self.system_prompt != '':
            message0 = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": message},
            ]
        else:
            message0 = [
                {"role": "user", "content": message},
            ]
        client = self.players[player_id]['client']
        model = self.players[player_id]['model']
        response = client.chat.completions.create(
            model=model,
            messages=message0,
            temperature=self.temperature,
            max_tokens=200,
        )
        return response.choices[0].message.content
'''

class EnvPlayer():
    """在双人或多人玩家场景下初始化环境中的其他玩家"""

    def __init__(self, num_players, player_info, system_prompt='', temperature=0.5) -> None:
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.num_players = num_players
        assert self.num_players == len(player_info) + 1

        # 存储玩家信息，不直接初始化带key的client
        self.players = []
        for i in range(self.num_players - 1):
            model_name = player_info[i]['model_name']
            if model_name == 'deepseek':
                # 只保存模型类型，key 在 act 时动态获取
                self.players.append({
                    'type': 'deepseek',
                    'model': 'deepseek-chat'
                })
            elif 'game' in model_name or 'Qwen' in model_name:
                port = player_info[i]['port']
                client = OpenAI(
                    api_key="EMPTY",  # vLLM 无需认证密钥
                    base_url=f"http://localhost:{port}/v1"
                )
                self.players.append({
                    'type': 'qwen',
                    'client': client,
                    'model': f"/root/autodl-tmp/{model_name}"
                })
            else:
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=open_router_key,
                )
                self.players.append({
                    'type': 'openrouter',
                    'client': client,
                    'model': model_name
                })

    def act(self, message, player_id):
        """
        调用API、获得指定玩家回复
        Args:
            message: 用户消息
            player_id: 玩家ID (0 表示第一个非训练模型的 Player)
        """
        assert player_id < self.num_players - 1

        # 组装消息
        if self.system_prompt:
            message0 = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": message},
            ]
        else:
            message0 = [{"role": "user", "content": message}]

        player = self.players[player_id]

        # 根据类型动态获取 client
        if player['type'] == 'deepseek':
            client = OpenAI(
                api_key=next(deepseek_keys),  # 每次调用都换一个 key
                base_url='https://api.deepseek.com'
            )
            model = player['model']
        elif player['type'] == 'qwen':
            client = player['client']  # Qwen 类型可直接复用
            model = player['model']
        else:  # openrouter
            client = player['client']
            model = player['model']

        # 请求模型
        response = client.chat.completions.create(
            model=model,
            messages=message0,
            temperature=self.temperature,
            max_tokens=600,
        )
        return response.choices[0].message.content


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # 测试next的方法是否能够正确生成api-key
    client = OpenAI(api_key=next(deepseek_keys),
                    base_url='https://api.deepseek.com')
    message = [{"role": "user", "content": 'Hi, who are you?'}]
    response = client.chat.completions.create(
            model='deepseek-chat',
            messages=message,
            # temperature=self.temperature,
            max_tokens=600,
        )
    print(response.choices[0].message.content)
