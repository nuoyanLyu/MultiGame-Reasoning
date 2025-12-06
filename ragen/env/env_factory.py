from typing import List, Dict, Any, Optional
from .base import EnvPlayer, SuccessRate

# 全局EnvPlayer实例缓存
_env_player_cache = {}

def get_env_player(player_num: int, player_info: List[Dict[str, Any]], 
                  temperature: float = 0.7, model_path: str = "", 
                  max_tokens: int = 1024, max_retries: int = 3,
                  system_prompt: Optional[str] = None) -> EnvPlayer:
    """
    获取或创建EnvPlayer实例，使用缓存避免重复创建
    
    Args:
        player_num: 玩家数量
        player_info: 玩家信息列表
        temperature: 温度系数
        model_path: 模型路径
        max_tokens: 最大token数
        max_retries: 最大重试次数
        system_prompt: 系统提示
        
    Returns:
        EnvPlayer实例
    """
    # 创建缓存键
    cache_key = (
        player_num,
        tuple(sorted((info['model_name'], info.get('port', 0)) for info in player_info)),
        temperature,
        model_path,
        max_tokens,
        max_retries,
        system_prompt
    )
    
    # 如果缓存中存在，直接返回
    if cache_key in _env_player_cache:
        return _env_player_cache[cache_key]
    print(f'create new EnvPlayer instance. cache_key: {cache_key}')
    # 创建新实例并缓存
    env_player = EnvPlayer(
        num_players=player_num,
        player_info=player_info,
        temperature=temperature,
        model_path=model_path,
        max_tokens=max_tokens,
        max_retries=max_retries,
        system_prompt=system_prompt
    )
    
    _env_player_cache[cache_key] = env_player
    return env_player

def clear_env_player_cache():
    """
    清空EnvPlayer缓存
    """
    global _env_player_cache
    _env_player_cache = {}

def create_env_player_for_config(config) -> EnvPlayer:
    """
    根据配置对象创建EnvPlayer实例的便捷函数
    
    Args:
        config: 环境配置对象，需包含player_num, player_info等属性
        
    Returns:
        EnvPlayer实例
    """
    # temperature=0.5, model_path='/root/autodl-tmp', max_tokens=200, max_retries=3
    return get_env_player(
        player_num=config.player_num,
        player_info=config.player_info,
        temperature=getattr(config, 'temperature', 0.5),
        model_path=getattr(config, 'model_path', "/root/autodl-tmp"),
        max_tokens=getattr(config, 'env_max_tokens', 200),
        max_retries=getattr(config, 'max_retries', 3),
        system_prompt=getattr(config, 'system_prompt', '')
    )

# 同理，创造字典管理多个环境成功率计算方案
_env_success_info = {}


def create_success_info(env_name, k=100):
    if env_name not in _env_success_info:
        _env_success_info[env_name] = SuccessRate(k=k)
        print('create success_info dictionary for ENV', env_name)
    return _env_success_info[env_name]


def clean_success_info(env_name):
    if env_name in _env_success_info:
        del _env_success_info[env_name]