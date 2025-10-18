import numpy as np
from typing import List, Optional, Tuple, Dict
import re
from openai import NotFoundError
import gymnasium as gym
import random
import json

from ragen.env.base import BaseLanguageBasedEnv, EnvPlayer, seed_everything, timed, MultiGameEnv, Simplifier
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
        self.phase = None  # 标志是conversation还是vote轮次
        self.alive_players = None
        self.votes = []
        # 可能需要调用API进行历史对话精简
        self.simplifier = Simplifier('google/gemini-2.5-flash')
        self.simplify_prompt = self.config.simplify_prompt
        self.history = []
        self.reset()

    def reset(self, seed):
        """Initializes the conversation history, choose words and identities"""
        self.seed = seed
        seed_everything(seed)
        self.history = []
        self.conversation = []
        self.current_player_id = 0
        for i in range(self.config.player_num):
            # 第i个列表就是第i个玩家的发言历史，这样组织更方便后续的simplify过程
            # 后续需要设置合理的可视化场景
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
        self.identities = ["Civilian"] * self.config.player_num
        self.identities[undercover_id] = "Undercover"
        self.phase = 'conversation'
        # 记录alive players，如果投票出局则从列表中删除
        # 注意这里记录的index信息都是从0开始，
        # 只有在render以及投票的时候需要注意下标问题
        self.alive_players = list(range(self.config.player_num))
        # 如果环境是玩家0先手，环境先下一步
        while self.current_player_id != self.train_id:
            prompt_for_env_player = self.render()
            action = self.env_player.act(prompt_for_env_player)
            self._process_action(self.current_player_id, action)
            self._move_to_next_player()

    def step(self, action: str) -> Tuple[str, bool, bool, Dict]:
        """
        Processes the trainer's action and advances the game until it is the
        trainer's turn again or the game ends.

        Args:
            action: The action (description or vote) from the trainer.

        Returns:
            A tuple containing:
            - prompt: The next prompt for the trainer.
            - done: Boolean indicating if the game is over.
            - success: Boolean indicating if the trainer won.
            - info: A dictionary with auxiliary information.
        """
        # 1. Process the trainer's action
        is_valid = self._process_action(self.train_id, action)
        if not is_valid:
            info = {'action_is_effective': False, 'action_is_valid': False, 'success': False}
            self.history.append({
                "action": action,
            })
            prompt = 'Invalid action. Please try again. ' + self.render()
            return (prompt, -0.1, False, info)
        
        self._move_to_next_player()

        # 2. Run game loop for env_players until it's trainer's turn again or game ends
        while self.current_player_id != self.train_id and not self.game_over:
            # 确定游戏阶段
            self._check_and_transition_phase()
            if self.game_over: 
                break
            prompt_for_env_player = self.render()
            env_action = self.env_player.act(prompt_for_env_player)
            self._process_action(self.current_player_id, env_action)
            self._move_to_next_player()

        # Final check for phase transition after the last env_player
        self._check_and_transition_phase()

        # 3. Determine success and create info dict
        trainer_identity = self.identities[self.train_id]
        success = (self.winner == trainer_identity)
        
        info = {
            "winner": self.winner,
            "trainer_identity": trainer_identity,
            "elimination_history": self.elimination_history
        }
        
        prompt = self.render() if not self.game_over else "Game Over."
        
        return prompt, self.game_over, success, info

    def render(self) -> str:
        """
        Generates the prompt for the current player based on the game state.

        Returns:
            A formatted string containing all necessary information for the player to act.
        """
        player_id = self.current_player_id
        
        # --- History Information ---
        history_str = ""
        if self.round_num > 1 or self.phase == 'vote':
            history_str += f"### Round {self.round_num} Descriptions\n"
            for i in self.alive_players:
                desc = self.descriptions.get(i, "No description yet.")
                history_str += f"Player {i+1}: {desc}\n"
        
        if self.elimination_history:
            history_str += "\n### Elimination History\n"
            history_str += "\n".join(self.elimination_history) + "\n"

        if not history_str.strip():
            history_str = "No history yet. The first description round is starting."

        # --- Turn-specific Prompt ---
        turn_prompt = f"## Your Turn\n"
        turn_prompt += f"You are Player {player_id + 1}.\n"
        turn_prompt += f"Your identity is: **{self.identities[player_id]}**.\n"
        turn_prompt += f"Your secret word is: **{self.words[player_id]}**.\n\n"

        if self.phase == 'description':
            turn_prompt += "Your task is to describe your word in a single sentence. Remember the rules."
        elif self.phase == 'vote':
            turn_prompt += "Your task is to vote for the undercover agent. Available players to vote for are:\n"
            vote_options = [p for p in self.alive_players if p != player_id]
            for p_id in vote_options:
                turn_prompt += f"- Player {p_id + 1}\n"
            turn_prompt += "\nPlease state your vote clearly, for example: 'I vote for Player 3'."
        
        prompt = f"""{SYSTEM_PROMPT}

{INIT_PROMPT}

## Game State
{history_str}

{turn_prompt}
Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format.
"""
        return prompt.strip()


    def _move_to_next_player(self):
        """Cycles to the next alive player."""
        assert self.current_player_id in self.alive_players
        if self.current_player_id == self.alive_players[-1]:
            # 但如果是这样意味着一轮已经结束——是否需要设置一个结束的标志？
            self.current_player_id = self.alive_players[0]
        id = self.alive_players.index(self.current_player_id)
        self.current_player_id = self.alive_players[id + 1]

    def _process_action(self, player_id: int, action: str) -> bool:
        """
        Updates game state based on a player's action.
        if action is valid, return True; else return False
        """
        # For simplicity, assume `action` is the direct answer.
        if self.phase == 'description':
            # 保存历史信息
            self.descriptions[player_id] = action.strip()
            return True
        elif self.phase == 'vote':
            match = re.search(r'player (\d+)', action.lower())
            if match:
                voted_player_num = int(match.group(1))
                # -1，和真实的index相对应
                voted_id = voted_player_num - 1
                if voted_id in self.alive_players and voted_id != player_id:
                    self.votes.append(voted_id)
                    return True
            return False

    def _check_and_transition_phase(self):
        """Checks if a phase is over and transitions to the next if needed."""
        if self.phase == 'description' and len(self.descriptions) == len(self.alive_players):
            self.phase = 'vote'
            self.current_player_id = self.alive_players[0] # Start voting from the first alive player
        elif self.phase == 'vote' and len(self.votes) == len(self.alive_players):
            self._tally_votes_and_eliminate()
            self._check_game_over()
            if not self.game_over:
                self._start_new_round()

    def _tally_votes_and_eliminate(self):
        """Counts votes, eliminates a player, and records it."""
        if len(self.votes) == 0: 
            return
        vote_counts = {p_id: 0 for p_id in self.alive_players}
        for voted_id in self.votes:
            if voted_id in vote_counts:
                vote_counts[voted_id] += 1
            else:
                print('error! vote not valid! DEBUG!')
                exit(0)
        
        max_votes = max([k for k in vote_counts.values()])
        eliminated_candidates = [e for e in vote_counts.keys() 
                                    if vote_counts[e] == max_votes]
        
        # Handle ties by randomly choosing one to eliminate
        eliminated_id = random.choice(eliminated_candidates)

        self.alive_players.remove(eliminated_id)
        eliminated_identity = self.identities[eliminated_id]
        # 返回被票出的人的身份
        return eliminated_identity

    def _check_game_over(self):
        """Checks for win/loss conditions and sets game_over flag."""
        undercover_alive = any(self.identities[p_id] == "Undercover" for p_id in self.alive_players)
        if not undercover_alive:
            self.game_over = True
            self.winner = "Civilian"
        elif len(self.alive_players) <= 2:
            self.game_over = True
            self.winner = "Undercover"

    def _start_new_round(self):
        """Resets state for a new round of descriptions and votes."""
        self.round_num += 1
        # start from description phase
        self.phase = 'description'
        self.votes = []
        # The starting player for the new round is the first one in the alive list
        self.current_player_id = self.alive_players[0]

    def _history_render(self) -> str:
        """"Generate description for history conversation information."""
        if self.round == 0:
            return "There is no history information."
        info_str = ""
        # 按照Player整理历史发言信息
        for i in range(self.config.player_num):
            info_str += f"Player {i + 1}:"
            for j in self.conversation[i]:
                info_str += j + ' '
            info_str += '\n'
        # 判断是否超出对话长度，如果过长直接删除
        if len(info_str) > 250:
            # 调用API进行历史对话的精简，并缩写历史记录中的玩家发言信息
            info_str = self.simplifier.simplify(info_str, prompt=self.simplify_prompt)
            # 拆分输出结果，重新保存为conversation信息
            # 使用正则匹配所有玩家及其描述
            pattern = r"[Pp]layer\s+(\d+):\s*(.+)"
            matches = re.findall(pattern, info_str)
            for num, info in matches:
                self.conversation[int(num) - 1] = info.strip()
            # TODO：担心调用API会有错误输出，不知道这么搞会不会容易出BUG——记得测试！
        return info_str

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
"""
        if self.current_player_id == self.env_id:
            prompt = prompt0 + f"""Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: 200 words (tokens)."""
        else:
             prompt = prompt0 
        return prompt


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
