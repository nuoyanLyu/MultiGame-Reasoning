# nashenv/env.py
from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import nashpy as nash  
import random
import gymnasium as gym
from .config import NashNewConfig
from ragen.env.base import BaseLanguageBasedEnv, seed_everything, timed


class NashNew(BaseLanguageBasedEnv, gym.Env):
    def __init__(self, config: Optional[NashNewConfig] = None):
        BaseLanguageBasedEnv.__init__(self) 
        self.config = config or NashNewConfig()
        self.render_cache = None
        self.seed = int(self.config.seed)
        self.render_mode = self.config.render_mode
        assert self.render_mode == 'text'
        self.np_random: Optional[np.random.RandomState] = None
        # load payoff matrix
        self.payoff_list = self.config.payoff_list
        # runtime state
        self.role: str = ''
        self.variant_idx: int = 0
        self.last_prompt: str = ""
        self.valid_actions = []
        self.templates = self._build_templates()
        self.history = []

        self.reset(self.seed)

    def reset(self, seed: Optional[int] = None, **kwargs) -> str:
        self.history = []
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
            seed_everything(seed)
        elif self.np_random is None:
            self.np_random = np.random.RandomState(0)
            seed_everything(0)

        # payoff matrices 来自 config——随机选取游戏，对应修改config的setting
        game_id = random.choice(range(len(self.payoff_list)))
        self.A, self.B = self.payoff_list[game_id]
        # 随机*-1后有纳什均衡的进行一下变换，这样最优动作不会都是1
        if self.np_random.randint(0,2) and game_id < 7:
            self.A = (np.array(self.A) * (-1)).tolist()
            self.B = (np.array(self.B) * (-1)).tolist()
        # 随机+1000或-1000 数据多样性
        num0 = random.choice([0, 100, -100])
        if num0:
            self.A = (np.array(self.A) + num0).tolist()
            self.B = (np.array(self.B) + num0).tolist()
        self.variant_idx = int(self.np_random.randint(0, len(self.templates)))
        # self.variant_idx = seed
        if self.config.force_role in ("P1", "P2"):
            self.role = self.config.force_role
        else:
            self.role = "P1" if int(self.np_random.randint(0, 2)) == 0 else "P2"
        # 确定valid_actions个数
        if self.role == "P1":
            self.valid_actions = [str(i + 1) for i in range(len(self.A))]
        else:
            self.valid_actions = [str(i + 1) for i in range(len(self.B[0]))]
        self.last_prompt = self.render()
        return self.last_prompt

    def step(self, action: str) -> Tuple[str, int, bool, Dict[str, Any]]:
        """根据 nashpy计算NE给reward"""
        self.history.append(action)
        is_valid = action in self.valid_actions
        if not is_valid:
            info = dict(
                action_is_valid=False,
                action_is_effective=False,
                success=False,
            )
            return self.last_prompt, 0, True, info
        # game = nash.Game(np.array(self.A), np.array(self.B))
        best_action = self._get_eq()
        action = int(action)
        idx = action - 1
        # print('best action', best_action)
        success = (best_action == idx)
        reward = 1 if success else 0
        info: Dict[str, Any] = {
            "action_is_valid": True,
            "action_is_effective": success,
            "success": success,
        }
        return self.last_prompt, reward, True, info

    def close(self):
        pass

    # 判断是否达到纳什均衡，如果有多个纳什均衡则选择收益最大的动作
    def _get_eq(self):
        game = nash.Game(np.array(self.A), np.array(self.B))
        eqs = list(game.support_enumeration())[-1]
        if self.role == "P1":
            eq = eqs[0]
        else:
            eq = eqs[1]
        # 找到收益最大动作
        action = np.argmax(eq)
        return action

    def _build_templates(self) -> List[str]:
        """
        构建N x M游戏的prompt模板列表。
        使用结构化占位符，而不是单元格占位符。
        """
        templates: List[str] = []

        # 1. Markdown 双表
        templates.append(
            "{role}.\n\n"
            "Rows = Player 1's actions [{p1_actions_list}]; Columns = Player 2's actions [{p2_actions_list}].\n\n"
            "### P1's payoff\n"
            "{p1_payoff_table}\n\n"
            "### P2's payoff\n"
            "{p2_payoff_table}\n\n"
            "{instr}\n"
        )

        # 2. 单表(P1,P2)
        templates.append(
            "{role}.\n\n"
            "Each cell shows (P1 payoff, P2 payoff). Rows = [{p1_actions_list}], Columns = [{p2_actions_list}].\n"
            "{combined_payoff_table}\n\n"
            "{instr}\n"
        )

        # 3. 逐格文字
        templates.append(
            "{role}.\n\n"
            "The payoffs for each combination of actions are as follows:\n"
            "{list_payoffs}\n\n"
            "{instr}\n"
        )

        # 4. 数组（增加说明）
        templates.append(
            "{role}.\n\n"
            "Rows = [{p1_actions_list}], Columns = [{p2_actions_list}].\n"
            "P1 payoff matrix = {matrix_A}\n"
            "P2 payoff matrix = {matrix_B}\n\n"
            "{instr}\n"
        )

        # 5. 对话体
        templates.append(
            "{role}.\n\n"
            "You (P1) vs Opponent (P2). Rows = P1's actions [{p1_actions_list}], Columns = P2's actions [{p2_actions_list}].\n"
            "P1 payoff: {matrix_A}\n"
            "P2 payoff: {matrix_B}\n\n"
            "{instr}\n"
        )

        # 6. Alice/Bob
        templates.append(
            "{role}.\n\n"
            "Alice (P1) and Bob (P2) play a game. Rows = Alice's actions [{p1_actions_list}], Columns = Bob's actions [{p2_actions_list}].\n"
            "{combined_payoff_table}\n\n"
            "{instr}\n"
        )

        # 7. 公司竞争
        templates.append(
            "{role}.\n\n"
            "Company A (P1) and Company B (P2). Actions for A: [{p1_actions_list}], for B: [{p2_actions_list}].\n"
            "{list_payoffs_concise}\n\n" # 使用一种稍微不同的列表格式
            "{instr}\n"
        )

        # 8. 动物狩猎
        templates.append(
            "{role}.\n\n"
            "Two animals choose strategies. One is P1 (rows: [{p1_actions_list}]), the other is P2 (columns: [{p2_actions_list}]).\n"
            "{combined_payoff_table}\n\n"
            "{instr}\n"
        )

        # 9. Q&A 
        templates.append(
            "{role}.\n\n"
            "Q: What are the payoffs (Rows = P1 actions [{p1_actions_list}], Columns = P2 actions [{p2_actions_list}])?\n"
            "A: {qa_payoffs}\n\n"
            "{instr}\n"
        )

        # 10. 分步骤
        templates.append(
            "{role}.\n\n"
            "Step 1: Observe payoff matrices (Rows = P1 actions [{p1_actions_list}], Columns = P2 actions [{p2_actions_list}]).\n"
            "- P1 payoff: {matrix_A}\n"
            "- P2 payoff: {matrix_B}\n"
            "Step 2: Decide.\n\n"
            "{instr}\n"
        )

        # 11. 极简
        templates.append(
            "{role}.\n\n"
            "Game (Rows = P1 actions [{p1_actions_list}], Columns = P2 actions [{p2_actions_list}]):\n"
            "A = {matrix_A}\n"
            "B = {matrix_B}\n\n"
            "{instr}\n"
        )

        # 12. 强调完全信息
        templates.append(
            "{role}.\n\n"
            "This is a complete-information game. Rows = [{p1_actions_list}], Columns = [{p2_actions_list}].\n"
            "{combined_payoff_table}\n\n"
            "{instr}\n"
        )

        return templates

    def render(self) -> str:
        """
        动态生成N x M的prompt内容并填充模板。
        """
        A: List[List[Any]] = self.A
        B: List[List[Any]] = self.B
        N = len(self.A)
        M = len(self.B[0])
        p1_actions = [str(i + 1) for i in range(N)]
        p2_actions = [str(i + 1) for i in range(M)]

        role_sent = "You are Player 1 (P1)" if self.role == "P1" else "You are Player 2 (P2)"

        # --- 2. 定义辅助函数 (用于生成结构化内容) ---
        
        def _build_instr() -> str:
            """动态生成指示。"""
            actions_for_role = p1_actions if self.role == 'P1' else p2_actions
            instr_lines = "Choose your action: " + ' '.join(actions_for_role)
            # for i, action in enumerate(actions_for_role):
            #     instr_lines.append(f"- Enter {i+1} to select {action}")
            # return "\n".join(instr_lines)
            return instr_lines

        def _build_matrix_str(matrix: List[List[Any]]) -> str:
            """将矩阵格式化为 [[...],[...]] 字符串。"""
            rows_str = []
            for row in matrix:
                rows_str.append(f"[{','.join(map(str, row))}]")
            return f"[{','.join(rows_str)}]"

        def _build_separate_table(matrix: List[List[Any]]) -> str:
            """为 P1 或 P2 构建单独的 Markdown 支付表。"""
            header = "|      | " + " | ".join(p2_actions) + " |"
            divider = "|------|" + "-------|" * M
            table_rows = [header, divider]
            for i in range(N):
                row_label = p1_actions[i]
                cells = [str(matrix[i][j]) for j in range(M)]
                table_rows.append(f"| {row_label} | " + " | ".join(cells) + " |")
            return "\n".join(table_rows)

        def _build_combined_table() -> str:
            """构建 (P1, P2) 组合的 Markdown 支付表。"""
            header = "|      | " + " | ".join(p2_actions) + " |"
            # 动态调整分隔符长度，使其更美观（可选，但推荐）
            col_widths = [len(s) for s in p2_actions]
            for j in range(M):
                max_cell_width = max(len(f"({A[i][j]},{B[i][j]})") for i in range(N))
                col_widths[j] = max(col_widths[j], max_cell_width)

            divider = "|------|" + "|".join(["-" * (w + 2) for w in col_widths]) + "|"
            
            table_rows = [header, divider]
            for i in range(N):
                row_label = p1_actions[i]
                cells = []
                for j in range(M):
                    cell_content = f"({A[i][j]},{B[i][j]})"
                    # 居中对齐（可选）
                    # cells.append(f"{cell_content:^{col_widths[j]}}")
                    cells.append(cell_content) # 简单左对齐
                
                table_rows.append(f"| {row_label} | " + " | ".join(cells) + " |")
            return "\n".join(table_rows)

        def _build_list_payoffs() -> str:
            """构建逐格文字列表。"""
            lines = []
            for i in range(N):
                for j in range(M):
                    lines.append(f"- If P1={p1_actions[i]}, P2={p2_actions[j]} → (P1:{A[i][j]}, P2:{B[i][j]})")
            return "\n".join(lines)
        
        def _build_list_payoffs_concise() -> str:
            """为“公司竞争”构建简洁列表。"""
            lines = []
            for i in range(N):
                for j in range(M):
                    lines.append(f"- ({p1_actions[i]},{p2_actions[j]}) = ({A[i][j]},{B[i][j]})")
            return "\n".join(lines)

        def _build_qa_payoffs() -> str:
            """构建 Q&A 格式的单行回报。"""
            items = []
            for i in range(N):
                for j in range(M):
                    items.append(f"({p1_actions[i]},{p2_actions[j]})=({A[i][j]},{B[i][j]})")
            return ", ".join(items)

        # --- 3. 生成所有占位符内容 ---
        
        # 确保 self.templates 已经初始化
        if not hasattr(self, 'templates') or not self.templates:
             self.templates = self._build_templates()
             
        template = self.templates[self.variant_idx]
        
        # 准备一个字典来填充所有可能的占位符
        format_args = {
            "role": role_sent,
            "p1_actions_list": ", ".join(p1_actions),
            "p2_actions_list": ", ".join(p2_actions),
            "instr": _build_instr(),
            
            # 根据需要动态生成内容，避免不必要的计算
            # (如果一个模板不需要某内容，就不生成)
            # 为了简单起见，这里我们全部生成
            
            "p1_payoff_table": _build_separate_table(A),
            "p2_payoff_table": _build_separate_table(B),
            "combined_payoff_table": _build_combined_table(),
            "list_payoffs": _build_list_payoffs(),
            "list_payoffs_concise": _build_list_payoffs_concise(),
            "matrix_A": _build_matrix_str(A),
            "matrix_B": _build_matrix_str(B),
            "qa_payoffs": _build_qa_payoffs(),
        }

        # --- 4. 渲染并返回 ---
        return template.format(**format_args)


if __name__ == '__main__':
    env = NashNew()
    while 1:
        env.reset(random.randint(0, 1000))
        print(env.render())
        action = input('enter action: ')
        if action == 'q':
            break
        p, r, done, info = env.step(action)
        print("reward:", r, "done:", done)
        print(info)
        
