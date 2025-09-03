from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class UndercoverEnvConfig:
    render_mode: str = "text"
    action_lookup: Dict[int, str] = None # defined in env.py\
    # 总玩家数量，env_player数量+1
    player_num: int = 5
    # TODO：后续需要修改，看看self-play需要如何设置
    player_info: List[Dict[str, Any]] = field(
        # default_factory=lambda: [{'model_name': 'deepseek'}]
        # default_factory=lambda: [{'model_name': 'tictactoe/grpo/game_40', 'port': '4040'}]
        default_factory=lambda: 
            [{'model_name': 'deepseek'},{'model_name': 'deepseek'},
             {'model_name': 'deepseek'},{'model_name': 'deepseek'},]
    )
    temperature: float = 0.5
    seed: int = 123
    max_env_try: int = 3