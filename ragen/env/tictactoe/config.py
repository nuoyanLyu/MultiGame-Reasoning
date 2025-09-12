from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class TicTacToeEnvConfig:
    # lo_arm_name: str = "phoenix"
    # hi_arm_name: str = "dragon"
    # action_space_start: int = 1
    # lo_arm_score: float = 0.1
    # hi_arm_loscore: float = 0.0
    # hi_arm_hiscore: float = 1.0
    # hi_arm_hiscore_prob: float = 0.25
    # dataclass修饰：只会将带有“类型注解 (Type Hint)”的类属性，作为参数添加到自动生成的 __init__ 方法中。
    rows: int = 3
    cols: int = 3
    win_condition: int = 3
    render_mode: str = "text"
    action_lookup: Dict[int, str] = None # defined in env.py\
    # 另一个玩家的默认设置，后续可能可以更改
    player_num: int = 2
    # TODO：后续需要修改，看看self-play需要如何设置
    player_info: List[Dict[str, Any]] = field(
        # default_factory=lambda: [{'model_name': 'deepseek'}]
        default_factory=lambda: [{'model_name': "google/gemini-2.5-flash-lite"}]
        # default_factory=lambda: [{'model_name': 'tictactoe/grpo/game_40', 'port': '4040'}]
        # default_factory=lambda: [{'model_name': 'Qwen2.5-1.5B-Instruct', 'port': '2515'}]
    )
    model_path: str = "/root/autodl-tmp"  # "/data1/lvnuoyan/llm_model"
    temperature: float = 0.5
    seed: int = 123
    max_env_try: int = 3
