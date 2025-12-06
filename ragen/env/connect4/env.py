import numpy as np
from typing import List, Optional, Tuple, Dict
import re

from openai import NotFoundError

from ragen.env.base import BaseDiscreteActionEnv, EnvPlayer, seed_everything, timed, MultiGameEnv, Simplifier
from ragen.env.env_factory import create_env_player_for_config
from .config import Connect4EnvConfig
import gymnasium as gym
import random


system_prompt = "You are an expert in playing the Game Connect4."

INIT_PROMPT = f"""
## Game Rules: Connect Four

**Objective**: Be the first player to connect {Connect4EnvConfig().win_condition} of your pieces in a continuous line.

**Player Pieces**:
- Player 1: 'O'
- Player 2: 'X'
- Empty Slot: '.'

**How to Play**:
1. The game is played on a {Connect4EnvConfig().rows}x{Connect4EnvConfig().cols} vertical grid.
2. Players take turns dropping one of their pieces into any available column from the top.
3. The piece falls to the lowest unoccupied slot in that column.

**Winning Conditions**:
The game ends when a player forms a line of {Connect4EnvConfig().win_condition} of their own pieces. The line can be:

1.  **Horizontal** (side-by-side in a row)
    *Example of a horizontal win for Player 1 ('O'):*
    ```
    . . . . . . .
    . . . . . . .
    . . . . . . .  
    . X . X . . .
    . O O O O . . <-- 4 'O's in a horizontal line
    . O X O X X .
    ```

2.  **Vertical** (stacked on top of each other in a column)
    *Example of a vertical win for Player 2 ('X'):*
    ```
    . . . X . . .
    . . . X . . .
    . . . X . . .  <-- 4 'X's in column 4
    . . . X . . .  
    . O . O . . .
    O X O O . . .
    ```

3.  **Diagonal** (connected at an angle)
    *Example of a diagonal win (bottom-left to top-right) for Player 1:*
    ```
    . . . . . . .
    . . . . . . .
    . . . O . . .  <-- 4 'O's in a diagonal line
    . . O X . . .  
    . O X X . . .
    O X O O X . .
    ```
    *Example of another diagonal win (top-left to bottom-right) for Player 2:*
    ```
    . . . . . . .
    . . . . . . .
    . O X . . . .  <-- 4 'X's in a diagonal line
    . X O X . . .  
    O O X O X . .
    O O X O X X .
    ```

**Draw Condition**:
If the entire grid is filled with pieces and no player has won, the game is a draw.

"""


class Connect4Env(BaseDiscreteActionEnv, gym.Env):
    """
    A Connect4 game environment.
    Inherits from LLMGameEnv and implements the game-specific logic.
    """
    def __init__(self, config=None):
        BaseDiscreteActionEnv.__init__(self)
        self.config = config if config is not None else Connect4EnvConfig()
        # self.ACTION_SPACE = gym.spaces.discrete.Discrete(2, start=self.config.action_space_start)
        self.ACTION_SPACE = [str(c + 1) for c in range(self.config.cols)]
        self.cols = self.config.cols
        self.rows = self.config.rows
        self.seed = int(self.config.seed)
        self.win_condition = self.config.win_condition
        self.render_cache = None
        self.render_mode = self.config.render_mode
        assert self.render_mode == 'text'
        self.max_env_try = self.config.max_env_try
        self.env_player = create_env_player_for_config(self.config)
        self.env_id = None
        self.current_player_id = None
        self.history = []
        self.game_state = np.zeros((self.rows, self.cols), dtype=int)
        self.last_move: Optional[tuple[int, int]] = None
        self.reset()

    def reset(self, seed=None, **kwargs):
        """Initializes the board as a 6x7 grid of zeros."""
        self.seed = seed
        seed_everything(seed)
        self.game_state = np.zeros((self.rows, self.cols), dtype=int)
        # 根据随机数种子初始化一个棋局，打算是根据种子选取下棋步骤以及位置，
        # 从而保证不同的初始状态、不全是空棋盘开始
        # 重置环境Player对应的玩家顺序
        self.env_id = random.choice([0, 1])
        # 初始化——随机进行0 - 2轮下棋作为初始棋局，3局以内肯定不会上来就终止
        pre_action_step = random.choice([0, 1, 2])
        for step in range(pre_action_step):
            # 每一轮内两个玩家分别下棋
            for p in range(2):
                action0 = random.choice(self.get_all_actions())
                self._update_state(action0, p)
                self.history.append({
                    "player": p,
                    "action": action0
                })
        # 如果环境是玩家0先手，环境先下一步
        if self.env_id == 0:
            action = random.choice(self.get_all_actions())
            self._update_state(action, 0)
            self.history.append({
                "player": 0,
                "action": action
            })
        self.current_player_id = 1 - self.env_id

    def render(self) -> str:
        # 和之前的get_state_prompt相同的作用 生成当前棋局的信息
        # INIT_prompt，state_prompt以及action_prompt
        player = f'Player {self.current_player_id + 1}'
        state_prompt = self._get_state_prompt()
        actions = self.get_all_actions()
        prompt = f"""You are {player} playing game Connect 4.

## Rules
{INIT_PROMPT}

## Current Game State
{state_prompt}

## Your Turn
You are {player}.
The available actions are: {actions}.

Please choose your action and provide your response by enclosing your choice in <s></s> tags.
Reply with a SINGLE LINE with <s>{actions[0]}</s> and reasons with no more than 20 words.
For example: <s>{actions[0]}</s> reason: <NO MORE THAN 20 WORDS>
"""
        return prompt

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        Execute one step in the environment.
        NOTE should also handle predefined invalid action (0)
        Args:
            action: Action to take, must be in action space, or default invalid action
        Returns:
            observation, reward, done, info
            observation: updated game prompt;
            info: dictionary
        """
        action = self._parse_action(action)
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
        # print(env_prompt)
        valid = False
        try_count = 0
        while not valid and try_count < self.max_env_try:
            env_output = self.env_player.act(env_prompt, 0)
            # 同样处理action、更新环境的流程
            # 看一下deepseek输出的是什么东西？是否长篇大论
            # print(env_output)
            action = self._parse_action(env_output)
            available_actions = self.get_all_actions()
            # 如果错了环境agent可以多次调用，直到生成合理的solution
            if action in available_actions:
                valid = True
            try_count += 1  
        if not valid:
            # TODO:对手失误，算作agent胜利 OR 平局？感觉都不太合理，给一个中间的奖励？
            reward = 0.5
            done = True
            draw_prompt = 'Your opponent made a mistake! No winner.'
            info['action_is_effective'] = True
            info['action_is_valid'] = True
            info['success'] = True
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
        """Converts the numpy array into a human-readable board string."""
        # Header for columns
        header = " " + " ".join(str(i+1) for i in range(self.cols)) + " "
        # Reverse board rows for display so row 0 is at the bottom
        # 中间格子也应该使用|进行分割这样会更加统一吧——问问大模型看看哪个设置大模型比较好读懂
        board_str = "\n".join(["|" + " ".join(self._get_piece_char(p) for p in row) + "|" for row in self.game_state])
        return f"Board State:\n{header}\n{board_str}\n"

    def _get_piece_char(self, piece_val: int) -> str:
        if piece_val == 1: return "O" # Player 1
        if piece_val == 2: return "X" # Player 2
        return "." # Empty 可能空格也可以？——先用点占位置，看着感觉还行

    def get_all_actions(self) -> List[str]:
        """A column is available if its top cell (row 0) is empty."""
        return [str(c + 1) for c in range(self.cols) if self.game_state[0, c] == 0]

    def _parse_action(self, llm_output: str) -> Optional[str]:
        """Helper to extract action from LLM's raw output."""
        match = re.search(r'<s>(.*?)</s>', llm_output)
        # 这里是提取了第一个匹配到的信息，可能可以改成更加严格的匹配（比如只能匹配到一个）
        return match.group(1).strip() if match else None
    
    def _update_state(self, action: str, player_id):
        # 默认这里放下的棋子是合法的——下棋获得action的时候需要判定是否合法——yes
        """Drops a piece into the specified column."""
        col = int(action) - 1
        player_piece = player_id + 1
        
        # Find the lowest empty row in that column
        for r in range(self.rows - 1, -1, -1):
            if self.game_state[r, col] == 0:
                self.game_state[r, col] = player_piece
                self.last_move = (r, col)
                return
    
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
        Parses a Connect4 board string, updates the numpy game_state,
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
    config = Connect4EnvConfig()
    env = Connect4Env(config)
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
        # for o in [obs, reward, done, info]:
        #     print(o)
