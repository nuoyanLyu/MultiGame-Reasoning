import numpy as np
from typing import List, Optional, Tuple, Dict
import re
from openai import NotFoundError
import gymnasium as gym
import random
import json

from ragen.env.base import BaseLanguageBasedEnv, EnvPlayer, seed_everything, timed, MultiGameEnv
from .config import UndercoverEnvConfig



system_prompt = "You are an expert in playing the Game Undercover."

INIT_PROMPT = f"""Game Rule: Who's the Undercover Agent

Player Roles:
- Civilian: UndercoverEnvConfig().player_num - 1 players receive the same keyword.
- Undercover Agent: 1 player receives a different, but thematically related, keyword.

Objective:
- Civilian: Identify and vote out the Undercover Agent.
- Undercover Agent: Avoid being identified and voted out, and cause the civilians to vote out each other.

How to Play:
- At the beginning of each round, a word is secretly assigned to each player. All civilians receive one word, while the undercover agent receives a different one.
- Players take turns describing their assigned word using a single sentence. The key is to be subtle, avoiding words that are too obvious to avoid being found out by the undercover agent, but clear enough for fellow civilians to understand.
- After everyone has spoken, all players vote on who they believe is the undercover agent.
- The player with the most votes is eliminated from the game.

Note:
- **The description must exclude the original word.**
- **The description must be different from others.**

Winning Conditions:
- Civilian Win: The undercover agent is voted out and eliminated.
- Undercover Agent Win: The number of remaining players is reduced to two, with the undercover agent still in the game.
"""


class UndercoverGameEnv(MultiGameEnv):
    """
    Undercover Game Environment.
    Maintain game state and process. Call and process trainer and env_player.
    """
    def __init__(self, config=None):
        MultiGameEnv.__init__(self)
        self.config = config if config is not None else UndercoverEnvConfig()
        self.seed = int(self.config.seed)
        self.render_mode = self.config.render_mode
        assert self.render_mode == 'text'
        self.max_env_try = self.config.max_env_try
        self.env_player = EnvPlayer(self.config.player_num, self.config.player_info, temperature=self.config.temperature)
        self.train_id = None
        self.current_player_id = None
        # game_state应该是一个列表，存储每个玩家历史发言
        self.history_info = ""
        self.conversation = []
        for i in range(self.config.player_num):
            # 第i个列表就是第i个玩家的发言历史——后续可视化的时候要好好设计一下
            self.conversation.append([])
        self.round = 0  # record game round
        # 存储每个人的词以及对应的身份信息
        self.words = []
        self.wordlist = json.load(open("ragen/env/undercover/wordshuffle.json"))
        self.identities = []
        self.reset()

    def reset(self, seed):
        """Initializes the conversation history, choose words and identities"""
        self.seed = seed
        seed_everything(seed)
        self.conversation = []
        for i in range(self.config.player_num):
            # 第i个列表就是第i个玩家的发言历史——后续可视化的时候要好好设计一下
            self.conversation.append([])
        # trainer选择顺序
        self.train_id = random.choice(list(range(self.config.player_num)))
        # 初始化——选择词汇信息
        word_pair = random.choice(self.wordlist, self.config.player_num)
        # 选择undercover的ID信息
        undercover_id = random.choice(list(range(self.config.player_num)))
        # 根据undercoverID记录identity信息以及word信息
        self.words = [word_pair[1]] * self.config.player_num
        # 认为1号是好人词汇、0号是卧底词
        self.words[undercover_id] = word_pair[0]
        self.identities = ["Agents"] * self.config.player_num
        self.identities[undercover_id] = "Undercover"
        # 如果环境是玩家0先手，环境先下一步
        if self.train_id != 0:
            pass

    def history_render(self) -> str:
        """"Generate description for history conversation information."""
        if self.round == 0 and not self.history_info:
            return "There is no history information."
        # 可能需要调用API进行精简或删减
        # 不应该使用循环的方式反复生成，生成后不断叠加、可以反复调用
        # info_str = ""
        # for i in range(self.round):
        #     info_str += f"Round {i + 1}:\n"
        #     for j in range(self.config.player_num):
        #         info_str += f"Player {j + 1}: {self.conversation[j][i]}\n"
        else:
            # 判断是否超出对话长度，如果过长直接删除
            if len(self.history_info.split()) > 250:
                # 调用API进行历史对话的精简
                pass



    def render(self) -> str:
        # 和之前的get_state_prompt相同的作用 生成当前棋局的信息
        # INIT_prompt，state_prompt以及action_prompt
        player = f'Player {self.current_player_id + 1}'
        state_prompt = self._get_state_prompt()
        actions = self.get_all_actions()
        prompt0 = f"""You are {player} playing game Undercover.

## Rules
{INIT_PROMPT}

## History of Conversation
{state_prompt}

## Your Turn
You are {player}.
The available actions are: {actions}.
"""
        if self.current_player_id == self.env_id:
            prompt = prompt0 + f"""Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: 200 words (tokens)."""
        else:
             prompt = prompt0 
        return prompt




class UndercoverEnv(BaseLanguageBasedEnv, gym.Env):
    """
    A Undercover game environment.
    Inherits from LLMGameEnv and implements the game-specific logic.
    """
    def __init__(self, config=None):
        BaseLanguageBasedEnv.__init__(self)
        self.env = UndercoverGameEnv(config)

    def reset(self, seed=None, **kwargs):
        """Initializes the conversation history, choose words and identities"""
        self.env.reset(seed)

    def render(self) -> str:
        # 和之前的get_state_prompt相同的作用 生成当前棋局的信息
        # INIT_prompt，state_prompt以及action_prompt
        return self.env.render()

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        # This step function is same in all two-player game settings.
        Execute one step in the environment.
        NOTE should also handle predefined invalid action (0)
        Args:
            action: Action to take, must be in action space, or default invalid action
        Returns:
            observation, reward, done, info
            observation: updated game prompt;
            info: dictionary
        """
        # print(action)
        # 实际上模型调用的时候会生成format prompt，也会初步提取action信息，这里不需要模板匹配问题
        action = self._parse_action_trainer(action)
        available_actions = self.get_all_actions()
        info = {"action_is_effective": None, "action_is_valid": None, "success": None}
        train_id = 1 - self.env_id
        if action not in available_actions:
            # Handle invalid action - could return an error message, or penalize.
            error_prompt = f"Invalid action: '{action}'. \nGame over."
            info['action_is_effective'] = False
            info['action_is_valid'] = False
            info['success'] = False
            return (error_prompt, -1, True, info)
        
        # Update state with the valid action
        self._update_state(action, train_id)
        # Log the move
        self.history.append({
            "player": train_id,
            "action": action
        })

        # Check if the game is over
        win = self._check_win()
        reward = None
        if win:
            # self.winner = self._get_winner()
            # if self.winner == self.players[self.current_player_index]:
            reward = 1 # Simple reward: 1 for winning, 0 for draw/loss
            done = True
            success_prompt = 'Congratulations! You are the winner!'
            info['action_is_effective'] = True
            info['action_is_valid'] = True
            info['success'] = True
            self.reset()
            return success_prompt, reward, done, info
        # 判断是否为平局
        if len(self.get_all_actions()) == 0:
            reward = 0
            done = True
            draw_prompt = 'Draw! No winner.'
            info['action_is_effective'] = True
            info['action_is_valid'] = True
            info['success'] = False
            self.reset()
            return draw_prompt, reward, done, info
        
        # Switch to the next player if the game is not over
        # 环境agent采取行动
        self.current_player_id = self.env_id
        env_prompt = self.render()
        valid = False
        try_count = 0
        while not valid and try_count < self.max_env_try:
            env_output = self.env_player.act(env_prompt, 0)
            # 同样处理action、更新环境的流程
            # 看一下deepseek输出的是什么东西？是否长篇大论
            # print(env_output)
            # 降低env_output的匹配精确度，环境agent并不需要精确匹配
            action = self._parse_action_env(env_output, strict=False)
            # print(action)
            available_actions = self.get_all_actions()
            # 如果错了环境agent可以多次调用，直到生成合理的solution
            if action in available_actions:
                valid = True
            try_count += 1  
        if not valid:
            # TODO:对手失误，算作agent胜利 OR 平局？感觉都不太合理，给一个中间的奖励？
            # 尽量不要出现这个情况，理论上应该是一直等到环境agent有动作才好——
            # 甚至应该随机选一个作为动作才更加合理
            reward = 0
            done = True
            draw_prompt = 'Your opponent made a mistake! No winner.'
            info['action_is_effective'] = False
            info['action_is_valid'] = True
            # 不算成功吧，要不然会混淆模型训练结果
            info['success'] = False
            self.reset()
            return draw_prompt, reward, done, info
        # 对手正确，更新环境
        # Update state with the valid action
        self._update_state(action, self.env_id)
        # Log the move
        self.history.append({
            "player": self.env_id,
            "action": action
        })
        # 判定是否胜利，注意对手胜利是trainer失败，需要转换过来
        # Check if the game is over
        env_win = self._check_win()
        reward = None
        if env_win:
            # self.winner = self._get_winner()
            # if self.winner == self.players[self.current_player_index]:
            reward = -1 # Simple reward: 1 for winning, 0 for draw/loss
            done = True
            fail_prompt = 'Failed! The opponent wins! Game over. Final state: \n' + self._get_state_prompt()
            info['action_is_effective'] = True
            info['action_is_valid'] = True
            info['success'] = False
            self.reset()
            return fail_prompt, reward, done, info
        # 判断是否为平局
        if len(self.get_all_actions()) == 0:
            reward = 0
            done = True
            draw_prompt = 'Draw! No winner.\n' + self._get_state_prompt()
            info['action_is_effective'] = True
            info['action_is_valid'] = True
            info['success'] = False
            self.reset()
            return draw_prompt, reward, done, info
        
        # 没有失败、没有平局——调用train agent进入下一步
        self.current_player_id = 1 - self.env_id
        train_prompt = self.render()
        reward = 0
        done = False
        info['action_is_effective'] = True
        info['action_is_valid'] = True
        info['success'] = False
        return train_prompt, reward, done, info

    # 需要说明一下这里的四子棋的条件horizontal、vertical、diagonal具体的例子
    def _get_state_prompt(self) -> str:
        """Converts the numpy array into a human-readable board string with row and column numbers."""
        # Header for columns
        header = "   " + " ".join(str(i+1) for i in range(self.cols))  # 前面加3个空格对齐
        # 每行加行号，行号从1开始
        board_str = "\n".join(
            f"{r+1} |" + " ".join(self._get_piece_char(p) for p in row) + "|"
            for r, row in enumerate(self.game_state)
        )
        return f"Board State:\n{header}\n{board_str}\n"

    def _get_piece_char(self, piece_val: int) -> str:
        if piece_val == 1: return "O" # Player 1
        if piece_val == 2: return "X" # Player 2
        return "." # Empty 可能空格也可以？——先用点占位置，看着感觉还行

    def get_all_actions(self) -> List[str]:
        """A column is available if its top cell (row 0) is empty."""
        actions0 = []
        for i in range(self.rows):
            for j in range(self.cols):
                if self.game_state[i, j] == 0:
                    actions0.append((i + 1, j + 1))
        return actions0

    def _parse_action_trainer(self, action: str) -> Optional[Tuple]:
        """Helper to extract action from trainer's raw output."""
        pattern = r"\(\s*(\d+)\s*,\s*(\d+)\s*\)"
        match = re.search(pattern, action)
        if match:
            row = int(match.group(1))
            col = int(match.group(2))
            # print("匹配成功:", row, col)
            action = (row, col)
        else:
            # print("匹配失败")
            action = None
        return action
    
    def _parse_action_env(self, llm_output: str, strict=False) -> Optional[Tuple]:
        """Helper to extract action from env's raw output."""
        pattern = r"<answer>\(\s*(\d+)\s*,\s*(\d+)\s*\)</answer>"
        match = re.search(pattern, llm_output)
        if match:
            row = int(match.group(1))
            col = int(match.group(2))
            # print("匹配成功:", row, col)
            action = (row, col)
        else:
            # 环境player并不需要非常精确的匹配、指令遵循等信息
            if not strict:
                pattern = r"\(\s*(\d+)\s*,\s*(\d+)\s*\)"
                match = re.search(pattern, llm_output)
                if match:
                    row = int(match.group(1))
                    col = int(match.group(2))
                    # print("匹配成功:", row, col)
                    action = (row, col)
                else:
                    action = None
            else:
                action = None
        return action
    
    def _update_state(self, action: str, player_id):
        # 默认这里放下的棋子是合法的——下棋获得action的时候需要判定是否合法——yes
        """Drops a piece into the specified column."""
        # col = int(action) - 1
        row, col = action[0] - 1, action[1] - 1
        player_piece = player_id + 1
        self.game_state[row, col] = player_piece
        self.last_move = (row, col)
        # Find the lowest empty row in that column
        # for r in range(self.rows - 1, -1, -1):
        #     if self.game_state[r, col] == 0:
        #         self.game_state[r, col] = player_piece
        #         self.last_move = (r, col)
        #         return
    
    def _is_over(self) -> bool:
        """Game is over if there's a winner or the board is full (draw)."""
        if self.last_move is None:
            return False
        # A win is only possible if a move has been made
        if self._check_win():
            return True
        # Check for draw (no empty cells left)
        if len(self.get_all_actions()) == 0:
            return True
        return False

    def _get_winner(self) -> Optional[str]:
        """Returns the current player if they just won, otherwise checks for a draw."""
        if self._check_win():
            return self.players[self.current_player_index]
        return None # None indicates a draw or game not over

    def _check_win(self) -> bool:
        """Checks for a 4-in-a-row from the last move position."""
        if self.last_move is None:
            return False
            
        r, c = self.last_move
        player = self.game_state[r, c]
        if player == 0:
            return False

        # Check all four directions (horizontal, vertical, two diagonals)
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            # Check in one direction
            for i in range(1, self.win_condition):
                nr, nc = r + i * dr, c + i * dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.game_state[nr, nc] == player:
                    count += 1
                else:
                    break
            # Check in the opposite direction
            for i in range(1, self.win_condition):
                nr, nc = r - i * dr, c - i * dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.game_state[nr, nc] == player:
                    count += 1
                else:
                    break
            # 两侧统计连线数量是否超过WIN条件
            if count >= self.win_condition:
                return True
        return False

    def _parse_and_update_game_state(self, state_string: str) -> Dict[int, int]:
        """
        Parses a TicTacToe board string, updates the numpy game_state,
        and counts the pieces for each player.
        """
        # Define the character-to-piece-value mapping
        char_map = {'O': 1, 'X': 2, '.': 0}
        piece_counts = {1: 0, 2: 0}
        
        # state_string定位初始位置 直接全局匹配会匹配到之前的终局字符串信息
        board_str0 = state_string.split('Board State:')[1]
        # Use regex to find all board rows, which start with '|' and end with '|\n'
        # 这里是否能够添加换行符？不确定
        board_rows_str = re.findall(r'\|(.*?)\|\n', board_str0)
        
        if len(board_rows_str) != self.rows:
            raise ValueError(f"Invalid board string. Expected {self.rows} rows, but found {len(board_rows_str)}.")
            
        # Re-initialize the game state to ensure it's clean
        self.game_state = np.zeros((self.rows, self.cols), dtype=int)
        
        for r_idx, row_str in enumerate(board_rows_str):
            # Split the row content by spaces, ignoring extra whitespace
            pieces = row_str.strip().split(' ')
            if len(pieces) != self.cols:
                raise ValueError(f"Invalid row format on row {r_idx}. Expected {self.cols} columns.")
                
            for c_idx, piece_char in enumerate(pieces):
                if piece_char in char_map:
                    piece_value = char_map[piece_char]
                    self.game_state[r_idx, c_idx] = piece_value
                    if piece_value in piece_counts:
                        piece_counts[piece_value] += 1
                else:
                    raise ValueError(f"Unknown piece character '{piece_char}' at row {r_idx}, col {c_idx}.")

        return piece_counts

    def _load_current_player(self, state_info):
        """_summary_
        根据state信息更新current_player_index属性
        Args:
            state_info (_type_): _description_
        """
        p1_pieces = piece_counts.get(1, 0)
        p2_pieces = piece_counts.get(2, 0)
        
        if p1_pieces <= p2_pieces:
            self.current_player_index = 0  # Player 1's turn
        else:
            self.current_player_index = 1  # Player 2's turn

    def close():
        return None


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    config = TicTacToeEnvConfig()
    env = TicTacToeEnv(config)
    # print(env.reset(seed=42))
    done = False
    while not done:
        print(env.render())
        keyboard = input("Enter action: ")
        if keyboard == 'q':
            break
        # action = int(keyboard)
        # assert action in env.ACTION_LOOKUP, f"Invalid action: {action}"
        obs, reward, done, info = env.step(keyboard)
        for o in [reward, done, info]:
            print(o)
