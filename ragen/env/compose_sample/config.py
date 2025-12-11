from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

@dataclass
class ComposeSampleConfig:
    env0: str = 'math_lv3to5'
    sample_env_list: List[str] = field(
        default_factory=lambda: ["nash_new", "tictactoe"]
    )
    player_nums: List[int] = field(
        default_factory=lambda: [0, 2]
    )
    player_infos: List[List[Dict[str, Any]]] = field(
        # default_factory=lambda: [{'model_name': 'deepseek'}]
        # default_factory=lambda: [{'model_name': 'tictactoe/grpo/game_40', 'port': '4040'}]
        default_factory=lambda: 
            # [{'model_name': 'google/gemini-2.5-flash'}, # {'model_name': 'google/gemini-2.5-flash'},
            #  {'model_name': 'google/gemini-2.5-flash'},{'model_name': 'google/gemini-2.5-flash'},]
            # [{"model_name": "x-ai/grok-4-fast"}, {"model_name": "x-ai/grok-4-fast"},
            #  {"model_name": "x-ai/grok-4-fast"}, {"model_name": "x-ai/grok-4-fast"},]
            [[{'': ''}],[{'model_name': "gemini-2.5-flash-nothinking"}]]
            # [{'model_name': 'Qwen3-14B', 'port': '1414'},
            #  {'model_name': 'Qwen3-14B', 'port': '1414'},
            #  {'model_name': 'Qwen3-14B', 'port': '1414'}]]
    )
    mode: str = 'train'  # train or test for math env.
    seed: int = 123
    k: int = 1024
    render_mode: str = 'text'
    sample_method: str = 'difficulty-square'  # random or sequential