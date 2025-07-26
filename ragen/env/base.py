from abc import ABC, abstractmethod
import re
from typing import Optional, List, Tuple, Any, Dict
from openai import OpenAI
import random 
import numpy as np

# 未来开源的时候记得删了！！！！
deepseek_key = 'sk-9071ab7ceef44f17a184ff1757962580'
open_router_key = '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'
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
                client = OpenAI(api_key=deepseek_key,
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
            # max_tokens=2000,
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