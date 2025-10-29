import numpy as np
from typing import List, Optional, Tuple, Dict
import re
from math_verify import parse, verify

from openai import NotFoundError

from ragen.env.base import BaseLanguageBasedEnv, EnvPlayer, seed_everything, timed
from .config import NashMixEnvConfig
import gymnasium as gym
import datasets
import random
import os
import nashpy as nash  

class NashMixEnv(BaseLanguageBasedEnv, gym.Env):
    """
    A Math game environment.
    Inherits from LLMGameEnv and implements the game-specific logic.
    """
    def __init__(self, config=None):
        BaseLanguageBasedEnv.__init__(self)
        self.config = config if config is not None else NashMixEnvConfig()
        # self.ACTION_SPACE = gym.spaces.discrete.Discrete(2, start=self.config.action_space_start)
        self.render_cache = None
        self.seed = int(self.config.seed)
        self.render_mode = self.config.render_mode
        assert self.render_mode == 'text'
        data_path = self.config.data_path
        data_files = self.config.data_files
        self.data_path = f"{data_path}/reasoning/{data_files}"
        # NOTE：这么多数据集能否支持同时load多个环境的设置？
        # 如果不行可能需要每一次加载环境的时候需要通过stream的方式读取数据
        self.mode = self.config.mode
        self.data_train =  datasets.load_dataset("parquet", 
            data_files=f"{self.data_path}/{self.mode}.parquet")['train']
        self.question0 = None
        self.history = []
        # 记录进行到第几阶段，先做完nash再做mix数据
        self.phase = None
        # 初始化纳什均衡的setting
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

    def reset(self, seed=None, **kwargs):
        """Initializes the environment for a new question"""
        self.seed = seed
        seed_everything(seed)
        # 根据随机数种子初始化抽取一个问题——reset mix
        self.question0 = random.choice(self.data_train)
        # 随机，先nash或先mix
        self.phase = [random.choice(['nash', 'mix'])]
        # reset_nash
        self.reset_nash(seed)
        return self.render()

    def reset_nash(self, seed=None):
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
        self.last_prompt = self._render_nash_prompt()
        return self.last_prompt

    def render(self) -> str:
        assert self.question0 is not None, "question0 is None, please reset the environment first"
        if self.phase[-1] == 'mix':
            prompt = self.question0['question']
        else:
            prompt = self._render_nash_prompt()
        return prompt

    # def render_for_test(self) -> str:
    #     prompt = self.question0['question']
    #     prompt += f'\nAnswer is: {self.question0["answer"]}'
    #     return prompt

    def _extract_choice(self, text: str):
        """
        从文本中提取第一个A-D字母（句首答案），允许格式：
        - A
        - A.
        - A.hello
        - A something
        不匹配：
        - ADHD, BADGE 等单词中嵌入的情况
        """
        text = text.strip()
        match = re.search(r'(?<![A-Za-z])([A-D])(\.|(?:\s|$))', text)
        if match:
            return match.group(1).upper()
        return None

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        # 根据phase阶段判断是nash OR mix环境，返回最终结果OR进入下一步
        self.history.append(action)
        if self.phase[-1] == 'mix':
            prompt, reward, done, info = self.step_mix(action)
        else:
            prompt, reward, done, info = self.step_nash(action)
        # 如果是task2：直接返回结果
        if len(self.phase) == 2:
            return prompt, reward, done, info
        # 如果是task1、不正确，提前终止
        if len(self.phase) == 1 and reward == 0:
            return prompt, reward, done, info
        # task1且正确，进入task2
        if self.phase[-1] == 'mix':
            self.phase.append('nash')
        else:
            self.phase.append('mix')
        prompt = self.render()
        reward = 0 # 这个地方其实后续可以引入step_reward之类的东西
        done = False
        info = dict(
                action_is_valid=True,
                action_is_effective=True,
                success=True,
            )
        return prompt, reward, done, info

    def step_nash(self, action: str) -> Tuple[str, float, bool, Dict]:
        """根据 nashpy计算NE给reward"""
        is_valid = action in self.valid_actions
        # print(type(action), self.valid_actions)
        if not is_valid:
            info = dict(
                action_is_valid=False,
                action_is_effective=False,
                success=False,
            )
            self._done = True
            return '', 0, True, info
        game = nash.Game(np.array(self.A), np.array(self.B))
        NE = self._pure_nash_equilibria()  # [(row, col), ...]
        action = int(action)
        idx = action - 1
        success = False
        if self.role == "P1":
            success = any(r == idx for (r, c) in NE)
        else:  # P2
            success = any(c == idx for (r, c) in NE)

        reward = 1 if success else 0

        self._done = True

        info: Dict[str, Any] = {
            "action_is_valid": True,
            "action_is_effective": success,
            "success": success,
        }
        return '', reward, True, info

    def step_mix(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        Execute one step in the environment.
        In MixEnv, the action is the answer to the question.
        NOTE should also handle predefined invalid action (0)
        Args:
            action: Action to take, must be in action space, or default invalid action
        Returns:
            observation, reward, done, info
            observation: updated game prompt;
            info: dictionary
        """
        # 数学题没必要再问一次，直接设置max_turn数量为1
        # 直接对比action和答案是否相同
        type = self.question0['type']
        if type == 'math':
            ground_truth = parse(self.question0['answer'])
            answer = parse(action)
            valid = not (len(answer) == 0)  # 如果没有匹配到数字，输出的是空列表
            success = verify(answer, ground_truth)
            reward = int(success)
            prompt = 'Answer is correct!' if success else 'Answer is incorrect.'
            done = True
        else:
            # 判定选项是否正确，答案为ABCD其中的一个            
            answer = self._extract_choice(action)
            if answer is None or answer not in ['A', 'B', 'C', 'D']:
                success = False
                valid = False
            else:
                success = answer == self.question0['answer']
                valid = True
            reward = int(success)
            prompt = 'Answer is correct!' if success else 'Answer is incorrect.'
            done = True
        info = {'action_is_effective': success, 'action_is_valid': valid, 'success': success}
        return ('', reward, done, info)

    def close():
        # 清空数据相关的变量信息
        self.data_train = None
        self.question0 = None

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

    def _render_nash_prompt(self) -> str:
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
    # import matplotlib.pyplot as plt
    for key in ["http_proxy", "https_proxy", "all_proxy", 
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
        os.environ.pop(key, None)
    config = NashMixEnvConfig()
    env = NashMixEnv(config)
    # print(env.reset(seed=42))
    r = True
    while r:
        done = False
        env.reset(random.randint(0, 1000))
        while not done:
            print(env.render())
            keyboard = input("Enter answer: ")
            if keyboard == 'q':
                r = False
                break
            obs, reward, done, info = env.step(keyboard)
            for o in [obs, reward, done, info]:
                print(o)
        print('env done! next round')
