from dataclasses import dataclass, field
from typing import Optional, Tuple, List

@dataclass
class ComposeConfig:
    base_env_list: List[str] = field(
        default_factory=lambda: ["mix_data", "tictactoe"]
    )
    mode: str = 'test'  # train or test for mix env.
    seed: int = 123
    render_mode: str = 'text'
    reward0: float = 0.5  # reward of complete sub-turn task