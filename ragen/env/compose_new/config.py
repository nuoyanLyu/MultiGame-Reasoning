from dataclasses import dataclass, field
from typing import Optional, Tuple, List

@dataclass
class ComposeNewConfig:
    base_env_list: List[str] = field(
        default_factory=lambda: ["math_lv3to5", "nash_new"]
    )
    reward_improve: bool = True
    k: int = 100  # 计算成功率使用的最近k个历史信息
    mode: str = 'test'  # train or test for mix env.
    seed: int = 123
    render_mode: str = 'text'