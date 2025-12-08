from abc import ABC, abstractmethod
import re
from typing import Optional, List, Tuple, Any, Dict
from openai import OpenAI, APITimeoutError, APIConnectionError, APIStatusError
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

keys = json.load(open('ragen/env/api-keys.json'))
deepseek_keys = keys['deepseek']
# deepseek_key = os.environ.get('DEEPSEEK_KEY')
if not deepseek_keys:
    print('No deepseek keys found, please set it in file `ragen/env/api-keys.json`.')
    exit(1)
deepseek_keys = ThreadSafeCycle(deepseek_keys)
openrouter_keys = keys['openrouter']
if not openrouter_keys:
    print('No openrouter keys found, please set it in file `ragen/env/api-keys.json`.')
    print('Cannot use openrouter key.')
    exit(1)
openrouter_keys = ThreadSafeCycle(openrouter_keys)
dmx_keys = keys['dmx']
if not dmx_keys:
    print('No dmx keys found, please set it in file `ragen/env/api-keys.json`.')
    exit(1)
dmx_keys = ThreadSafeCycle(dmx_keys)


class Simplifier():
    def __init__(self, model_name):
        # gemini: google/gemini-2.5-flash-lite
        self.player = dict()
        if model_name == 'deepseek':
            self.player = {
                    'type': 'deepseek',
                    'model': 'deepseek-chat'
                }
        elif '/' in model_name:   # openrouter
            self.player = {
                    'type': 'openrouter',
                    'model': model_name
                }
        else:
            self.player = {
                    'type': 'dmx',
                    'model': model_name
                }
            
    def simplify(self, history, prompt=None):
        if prompt is None:
            prompt = "\nPlease simplify this history conversation in **no more than 100** words."
        if self.player['type'] == 'deepseek':
            client = OpenAI(
                api_key=next(deepseek_keys),  # 每次调用都换一个 key
                base_url='https://api.deepseek.com'
            )
        elif self.player['type'] == 'openrouter':  # openrouter
            client = OpenAI(
                api_key=next(openrouter_keys),  # 每次调用都换一个 key
                base_url='https://openrouter.ai/api/v1',
            )
        elif self.player['type'] == 'dmx':  # dmx
            client = OpenAI(
                api_key=next(dmx_keys),  # 每次调用都换一个 key
                base_url='https://www.dmxapi.cn/v1',
            )
        response = client.chat.completions.create(
            model=self.player['model'],
            messages=[{"role": "user", "content": history + prompt}],
            max_tokens=150,
            temperature=0.5,
        )
        return response.choices[0].message.content


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
        

class EnvPlayer():
    """在双人或多人玩家场景下初始化环境中的其他玩家"""

    def __init__(self, num_players, player_info, system_prompt='', temperature=0.5, model_path='/root/autodl-tmp', max_tokens=200, max_retries=3) -> None:
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.num_players = num_players
        assert self.num_players == len(player_info) + 1
        # print(player_info[0]['model_name'], player_info[0]['port'])
        # 存储玩家信息，不直接初始化带key的client
        self.players = []
        self.max_tokens = max_tokens
        # 新增request_timeout时间限制以及retry信息
        self.max_retries = max_retries
        for i in range(self.num_players - 1):
            model_name = player_info[i]['model_name']
            if model_name == 'deepseek':
                # 只保存模型类型，key 在 act 时动态获取
                self.players.append({
                    'type': 'deepseek',
                    'model': 'deepseek-chat',
                    'request_timeout': 15
                })
            elif 'game' in model_name or 'Qwen' in model_name or '50' in model_name or '00' in model_name:
                port = player_info[i]['port']
                client = OpenAI(
                    api_key="EMPTY",  # vLLM 无需认证密钥
                    base_url=f"http://localhost:{port}/v1"
                )
                self.players.append({
                    'type': 'qwen',
                    'client': client,
                    'model': f"{model_path}/{model_name}",
                    'request_timeout': 5
                })
            elif '/' in model_name:
                self.players.append({
                    'type': 'openrouter',
                    'model': model_name,
                    'request_timeout': 10
                })
            else:
                self.players.append({
                    'type': 'dmx',
                    'model': model_name,
                    'request_timeout': 10
                })


    def act(self, message, player_id):
        """
        调用API、获得指定玩家回复
        添加超时和重试逻辑
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
        # print(player)
        # 根据类型动态获取 client
        retries = 0
        timeout = player['request_timeout'] # 从配置中获取，默认 10 秒
        while retries < self.max_retries:
            try:
                if player['type'] == 'deepseek':
                    client = OpenAI(
                        api_key=next(deepseek_keys),  # 每次调用都换一个 key
                        base_url='https://api.deepseek.com'
                    )
                    model = player['model']
                elif player['type'] == 'qwen':
                    client = player['client']  # Qwen 类型可直接复用
                    model = player['model']
                elif player['type'] == 'openrouter':  # openrouter
                    client = OpenAI(
                        api_key=next(openrouter_keys),  # 每次调用都换一个 key
                        base_url='https://openrouter.ai/api/v1',
                    )
                    model = player['model']
                elif player['type'] == 'dmx':  # dmx
                    client = OpenAI(
                        api_key=next(dmx_keys),  # 每次调用都换一个 key
                        base_url='https://www.dmxapi.cn/v1',
                    )
                    model = player['model']

                # 请求模型
                if 'Qwen3-14B' in model:
                    response = client.chat.completions.create(
                        model=model,
                        messages=message0,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        extra_body={
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                        timeout=timeout, # <--- 设置超时
                    )
                else:
                    response = client.chat.completions.create(
                        model=model,
                        messages=message0,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        timeout=timeout, # <--- 设置超时
                    )
                
                # 成功，返回结果
                return response.choices[0].message.content
            except (APITimeoutError, APIConnectionError, APIStatusError) as e:
                retries += 1
                wait_time = 2 ** retries # 指数退避
                print(f"⚠️ 玩家 {player_id} ({player['type']}) 请求超时 (尝试 {retries}/{self.max_retries}). {wait_time} 秒后重试... 错误: {e}")
                time.sleep(wait_time) # 等待后重试
            except Exception as e:
                retries += 1
                wait_time = 2 ** retries # 指数退避
                # 捕获其他不可重试的错误 (如: 认证失败、请求格式错误 4xx)
                print(f"❌ 玩家 {player_id} ({player['type']}) 发生不可重试的错误: {e}")
                time.sleep(wait_time) # 等待后重试
        
        # 如果循环结束 (所有重试均失败)
        print(f"❌ 玩家 {player_id} ({player['type']}) 在 {self.max_retries} 次尝试后仍失败。")
        return ''
        # exit(1)


class SuccessRate():
    def __init__(self, k, alpha=1, beta=1):
        self.success_list = []
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.success_rate = None
    
    def update(self, success):
        self.success_list.append(success)
        if len(self.success_list) > self.k:
            self.success_list.pop(0)
        # reward改为beta分布，s + alpha / (n + alpha + beta)
        return self.calcu()
    
    def calcu(self):
        self.success_rate = (sum(self.success_list) + self.alpha) / (len(self.success_list) + self.alpha + self.beta)
        return self.success_rate

    def get(self):
        return self.success_rate


class MultiGameEnv(ABC):
    """
    Abstract base class for all multi-game environments.
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
    def step(self, trainer_action: str) -> Tuple[Any, float, bool, Dict]:        
        """
        Execute one step in the environment.
        NOTE should also handle predefined invalid action (0)
        Args:
            trainer_action: Action to take, must be in action space, or default invalid action
            
        Returns: information for training LLM agents.
            observation (rendered environment), reward, done, info
        """
        pass
    
    @abstractmethod
    def render(self) -> str:
        """Render the environment. Optional method."""
        pass

    @abstractmethod
    def close(self):
        """Close the environment."""
        pass

    def compute_reward(self, action, **kwargs) -> float:
        """Compute reward for the action."""
        pass
    
    def is_valid(self, action):
        """Check if the action is valid."""
        pass

    def check_state(self):
        """Check if the game state is win, draw or fail."""
        pass

    def update_state(self, action):
        """Update the game state.
        Args:
            action: valid agents action information.
        """
        pass


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # 测试simplifier是否能够正常使用 google/gemini-2.5-flash-lite
#     simpifier = Simplifier('google/gemini-2.5-flash')
#     history = """
# Player 2: 1. You can use it to make juice or pie.
# Player 4: 1. I think a teacher once gave one to me. 2. It's a very crisp fruit, and it's a good source of fiber.
# Player 1: 1. It's a very common fruit. 2. It's a healthy snack that you can pack for lunch. 3. I'm not going to give any more clues. It's too risky.
# Player 3: 1. It's very sweet and you can peel it. 2. I think it's part of a very popular saying. 3. It's a classic snack, and it's a common color.
# Player 5: 1. It's typically round and can be red or green. 2. You can find it in a lot of different seasons. 3.I'm feeling confident about who the spy is.

#     """
#     prompt = "\nSummarize the conversation history of each player in format: `Player x: [conversation]\n`. Keep the summary ** under 100 words. **"

#     print(simpifier.simplify(history, prompt=prompt))
    # 测试request_timeout设置是否合理
    env_player = EnvPlayer(
        num_players=2,
        player_info=[{'model_name': 'deepseek-1-1-1'}]
    )
    print(env_player.act('hello', 0))
