# nashenv/env.py
from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import nashpy as nash  
import random
import gymnasium as gym
from .config import NashEnvConfig
from ragen.env.base import BaseLanguageBasedEnv, seed_everything, timed


class NashEnv(BaseLanguageBasedEnv, gym.Env):
    """
    完全信息 2x2 矩阵博弈环境（接口与 bandit 对齐）
      - reset(seed) -> str
      - step(action) -> (str, reward, done, info)
      - render() -> str
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, config: Optional[NashEnvConfig] = None):
        BaseLanguageBasedEnv.__init__(self) 
        self.config = config or NashEnvConfig()
        self.render_cache = None
        self.seed = int(self.config.seed)
        self.render_mode = self.config.render_mode
        assert self.render_mode == 'text'
        self.np_random: Optional[np.random.RandomState] = None

        # payoff matrices
        self.A: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None  # P1 payoff
        self.B: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None  # P2 payoff

        # runtime state
        self.role: str = "P1"
        self.variant_idx: int = 0
        self.last_prompt: str = ""
        self._done: bool = False
        self.valid_actions: List[str] = ['1', '2']
        self.templates = self._build_templates()
        self.history = []

        self.reset(self.seed)

    def reset(self, seed: Optional[int] = None, **kwargs) -> str:
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
            seed_everything(seed)
        elif self.np_random is None:
            self.np_random = np.random.RandomState(0)
            seed_everything(0)
        self._done = False

        # payoff matrices 来自 config——随机选取游戏，对应修改config的setting
        games = ["PD", "SH", "MP"]
        game = self.np_random.choice(games)
        self.A, self.B = self.config.get_payoff_matrices(game)

        self.variant_idx = int(self.np_random.randint(0, len(self.templates)))
        if self.config.force_role in ("P1", "P2"):
            self.role = self.config.force_role
        else:
            self.role = "P1" if int(self.np_random.randint(0, 2)) == 0 else "P2"
        self.last_prompt = self._render_prompt()
        return self.last_prompt

    def step(self, action: str) -> Tuple[str, int, bool, Dict[str, Any]]:
        # 测试是否reward全部相同会导致梯度nan 
        self.history.append(action)
        success = False
        reward = 1 if success else 0
        self._done = True
        info: Dict[str, Any] = {
            "action_is_valid": True,
            "action_is_effective": success,
            "success": success,
        }
        return self.last_prompt, reward, True, info

    def render(self) -> str:
        return self.last_prompt

    def close(self):
        pass

    #内部函数
    #版本问题，手写用is_best_response 代替game.pure_nash_equilibria()判断纯策略纳什均衡
    def _pure_nash_equilibria(self) -> List[Tuple[int, int]]:
        game = nash.Game(np.array(self.A), np.array(self.B))
        NE = []
        for i in [0, 1]:   # P1 行
            for j in [0, 1]:  # P2 列
                row_strategy = np.zeros(2); row_strategy[i] = 1
                col_strategy = np.zeros(2); col_strategy[j] = 1
                is_row_best, is_col_best = game.is_best_response(row_strategy, col_strategy)
                if is_row_best and is_col_best:
                    NE.append((i, j))
        return NE

    #构建prompt，12种
    def _build_templates(self) -> List[str]:
        templates: List[str] = []

        # Markdown 双表
        templates.append(
            "{role}.\n\n"
            "Rows = Player 1's actions [{p1a}, {p1b}]; Columns = Player 2's actions [{p2x}, {p2y}].\n\n"
            "### P1's payoff\n"
            "|      | {p2x} | {p2y} |\n"
            "|------|-------|-------|\n"
            "| {p1a} | {A11} | {A12} |\n"
            "| {p1b} | {A21} | {A22} |\n\n"
            "### P2's payoff\n"
            "|      | {p2x} | {p2y} |\n"
            "|------|-------|-------|\n"
            "| {p1a} | {B11} | {B12} |\n"
            "| {p1b} | {B21} | {B22} |\n\n"
            "{instr}\n"
        )

        # 单表(P1,P2)
        templates.append(
            "{role}.\n\n"
            "Each cell shows (P1 payoff, P2 payoff). Rows = [{p1a}, {p1b}], Columns = [{p2x}, {p2y}].\n"
            "|      | {p2x}         | {p2y}         |\n"
            "|------|---------------|---------------|\n"
            "| {p1a} | ({A11},{B11}) | ({A12},{B12}) |\n"
            "| {p1b} | ({A21},{B21}) | ({A22},{B22}) |\n\n"
            "{instr}\n"
        )

        # 逐格文字
        templates.append(
            "{role}.\n\n"
            "The payoffs for each combination of actions are as follows:\n"
            "- If P1={p1a}, P2={p2x} → (P1:{A11}, P2:{B11})\n"
            "- If P1={p1a}, P2={p2y} → (P1:{A12}, P2:{B12})\n"
            "- If P1={p1b}, P2={p2x} → (P1:{A21}, P2:{B21})\n"
            "- If P1={p1b}, P2={p2y} → (P1:{A22}, P2:{B22})\n\n"
            "{instr}\n"
        )

        # 数组（增加说明）
        templates.append(
            "{role}.\n\n"
            "Rows = [{p1a}, {p1b}], Columns = [{p2x}, {p2y}].\n"
            "P1 payoff matrix = [[{A11},{A12}],[{A21},{A22}]]\n"
            "P2 payoff matrix = [[{B11},{B12}],[{B21},{B22}]]\n\n"
            "{instr}\n"
        )

        # 对话体
        templates.append(
            "{role}.\n\n"
            "You (P1) vs Opponent (P2). Rows = P1's actions [{p1a}, {p1b}], Columns = P2's actions [{p2x}, {p2y}].\n"
            "P1 payoff: [[{A11},{A12}],[{A21},{A22}]]\n"
            "P2 payoff: [[{B11},{B12}],[{B21},{B22}]]\n\n"
            "{instr}\n"
        )

        # Alice/Bob
        templates.append(
            "{role}.\n\n"
            "Alice (P1) and Bob (P2) play a game. Rows = Alice's actions [{p1a}, {p1b}], Columns = Bob's actions [{p2x}, {p2y}].\n"
            "|      | {p2x}         | {p2y}         |\n"
            "|------|---------------|---------------|\n"
            "| {p1a} | ({A11},{B11}) | ({A12},{B12}) |\n"
            "| {p1b} | ({A21},{B21}) | ({A22},{B22}) |\n\n"
            "{instr}\n"
        )

        # 公司竞争
        templates.append(
            "{role}.\n\n"
            "Company A (P1) and Company B (P2). Actions for A: [{p1a}, {p1b}], for B: [{p2x}, {p2y}].\n"
            "- ({p1a},{p2x}) = ({A11},{B11})\n"
            "- ({p1a},{p2y}) = ({A12},{B12})\n"
            "- ({p1b},{p2x}) = ({A21},{B21})\n"
            "- ({p1b},{p2y}) = ({A22},{B22})\n\n"
            "{instr}\n"
        )

        # 动物狩猎
        templates.append(
            "{role}.\n\n"
            "Two animals choose strategies. One is P1 (rows: [{p1a}, {p1b}]), the other is P2 (columns: [{p2x}, {p2y}]).\n"
            "|      | {p2x}         | {p2y}         |\n"
            "|------|---------------|---------------|\n"
            "| {p1a} | ({A11},{B11}) | ({A12},{B12}) |\n"
            "| {p1b} | ({A21},{B21}) | ({A22},{B22}) |\n\n"
            "{instr}\n"
        )

        # Q&A 
        templates.append(
            "{role}.\n\n"
            "Q: What are the payoffs (Rows = P1 actions [{p1a}, {p1b}], Columns = P2 actions [{p2x}, {p2y}])?\n"
            "A: ({p1a},{p2x})=({A11},{B11}), "
            "({p1a},{p2y})=({A12},{B12}), "
            "({p1b},{p2x})=({A21},{B21}), "
            "({p1b},{p2y})=({A22},{B22}).\n\n"
            "{instr}\n"
        )

        # 分步骤
        templates.append(
            "{role}.\n\n"
            "Step 1: Observe payoff matrices (Rows = P1 actions [{p1a}, {p1b}], Columns = P2 actions [{p2x}, {p2y}]).\n"
            "- P1 payoff: [[{A11},{A12}],[{A21},{A22}]]\n"
            "- P2 payoff: [[{B11},{B12}],[{B21},{B22}]]\n"
            "Step 2: Decide.\n\n"
            "{instr}\n"
        )

        # 极简
        templates.append(
            "{role}.\n\n"
            "Game (Rows = P1 actions [{p1a}, {p1b}], Columns = P2 actions [{p2x}, {p2y}]):\n"
            "A = [[{A11},{A12}],[{A21},{A22}]]\n"
            "B = [[{B11},{B12}],[{B21},{B22}]]\n\n"
            "{instr}\n"
        )

        # 强调完全信息
        templates.append(
            "{role}.\n\n"
            "This is a complete-information 2x2 game. Rows = [{p1a}, {p1b}], Columns = [{p2x}, {p2y}].\n"
            "|      | {p2x}         | {p2y}         |\n"
            "|------|---------------|---------------|\n"
            "| {p1a} | ({A11},{B11}) | ({A12},{B12}) |\n"
            "| {p1b} | ({A21},{B21}) | ({A22},{B22}) |\n\n"
            "{instr}\n"
        )

        return templates

    def _render_prompt(self) -> str:
        p1a, p1b = self.config.action_labels_p1
        p2x, p2y = self.config.action_labels_p2
        A11, A12 = self.A[0]
        A21, A22 = self.A[1]
        B11, B12 = self.B[0]
        B21, B22 = self.B[1]

        role_sent = "You are Player 1 (P1)" if self.role == "P1" else "You are Player 2 (P2)"

        instr = (
            f"Choose your action:\n"
            f"- Enter 1 to select {p1a if self.role=='P1' else p2x}\n"
            f"- Enter 2 to select {p1b if self.role=='P1' else p2y}"
        )
        template = self.templates[self.variant_idx]
        return template.format(
            role=role_sent,
            p1a=p1a, p1b=p1b, p2x=p2x, p2y=p2y,
            A11=A11, A12=A12, A21=A21, A22=A22,
            B11=B11, B12=B12, B21=B21, B22=B22,
            instr=instr,
        )


if __name__ == "__main__":
    from .config import NashEnvConfig
    cfg = NashEnvConfig()
    for game in ["PD", "SH", "MP"]:
        A, B = cfg.get_payoff_matrices(game)
        print(game)
        game = nash.Game(np.array(A), np.array(B))
        NE = []
        for i in [0, 1]:   # P1 行
            for j in [0, 1]:  # P2 列
                row_strategy = np.zeros(2); row_strategy[i] = 1
                col_strategy = np.zeros(2); col_strategy[j] = 1
                is_row_best, is_col_best = game.is_best_response(row_strategy, col_strategy)
                if is_row_best and is_col_best:
                    NE.append((i, j))
        print(NE)
