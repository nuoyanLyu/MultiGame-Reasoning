from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class MixEnvConfig:
    render_mode: str = "text"
    data_path: str = "/root/autodl-tmp"  # "/data1/lvnuoyan"
    seed: int = 123
    mode: str = "test"
    # 数据文件设置，小模型使用abel、大模型使用qwen
    data_files: str = "merge_mmlu_math"
