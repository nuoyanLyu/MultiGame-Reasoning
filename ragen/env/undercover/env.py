from unittest import TextTestRunner
import numpy as np
from typing import List, Optional, Tuple, Dict
import re
from openai import NotFoundError
import gymnasium as gym
import random
import time
import json

from ragen.env.base import BaseLanguageBasedEnv, EnvPlayer, seed_everything, timed, MultiGameEnv, Simplifier
from ragen.env.env_factory import create_env_player_for_config
from .config import UndercoverEnvConfig


system_prompt = "You are an expert in playing the Game Undercover."

class UndercoverEnv(MultiGameEnv):
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
        self.init_prompts = self.config.init_prompts
        self.env_player = create_env_player_for_config(self.config)
        self.train_id = None
        self.current_player_id = None
        self.conversation = []
        for i in range(self.config.player_num):
            # 第i个列表就是第i个玩家的发言历史——后续可视化的时候要好好设计一下
            self.conversation.append([])
        self.round = 0  # record game round
        # 存储每个人的词以及对应的身份信息
        self.words = []
        # wordlist_easy adapted from
        # https://github.com/xzx34/SocialMaze/tree/main/find_the_spy
        with open(f"ragen/env/undercover/{self.config.wordlist}", "r") as f:
            self.wordlist = json.load(f)
        self.identities = []
        self.phase = None  # 标志是conversation还是vote轮次
        self.alive_players = None
        self.votes = []
        self.winner = None
        # 可能需要调用API进行历史对话精简
        self.simplifier = Simplifier('google/gemini-2.5-flash')
        self.simplify_prompt = self.config.simplify_prompt
        self.history = []
        self.game_over = False
        # self.reset(self.seed)

    def reset(self, seed, **kwargs):
        """Initializes the conversation history, choose words and identities"""
        self.seed = seed
        seed_everything(seed)
        self.history = []
        self.votes = []
        self.conversation = []
        self.game_over = False
        self.current_player_id = 0
        self.winner = None
        self.round = 0  # 初始化发言轮次计数器
        for i in range(self.config.player_num):
            # 第i个列表就是第i个玩家的发言历史，这样组织更方便后续的simplify过程
            # 后续需要设置合理的可视化场景
            self.conversation.append([])
        # trainer选择顺序
        self.train_id = random.choice(list(range(self.config.player_num)))
        # 初始化——选择词汇信息
        word_pair = random.choice(self.wordlist)
        # print(word_pair)
        # 选择undercover的ID信息
        undercover_id = random.choice(list(range(self.config.player_num)))
        # 根据undercoverID记录identity信息以及word信息
        self.words = [word_pair[1]] * self.config.player_num
        # 认为1号是好人词汇、0号是卧底词
        self.words[undercover_id] = word_pair[0]
        self.identities = ["Civilian"] * self.config.player_num
        self.identities[undercover_id] = "Undercover"
        self.undercover_id = undercover_id
        self.phase = 'description'
        # 记录alive players，如果投票出局则从列表中删除
        # 注意这里记录的index信息都是从0开始，
        # 只有在render以及投票的时候需要注意下标问题
        self.alive_players = list(range(self.config.player_num))
        # 如果环境是玩家0先手，环境先下一步
        while self.current_player_id != self.train_id:
            prompt_for_env_player = self.render()
            # print(prompt_for_env_player)
            action = self.env_player.act(prompt_for_env_player, self.current_player_id)
            self._process_env_action(self.current_player_id, action)
            self._move_to_next_player()
            self._check_and_transition_phase()

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
        # 1. Process the trainer's action - check if it is valid
        self.history.append({
                "action": action,
            })
        is_valid = self._process_action(self.train_id, action)
        if not is_valid:
            info = {'action_is_effective': False, 'action_is_valid': False, 'success': False}
            prompt = 'Invalid action. Please try again. ' + self.render()
            return (prompt, -0.1, False, info)
        
        self._move_to_next_player()
        self._check_and_transition_phase()

        # 2. Run game loop for env_players until it's trainer's turn again or game ends
        # 还有一种可能性——Trainer自己被票走了，加上这个判定
        while self.current_player_id != self.train_id and not self.game_over and self.train_id in self.alive_players:
            prompt_for_env_player = self.render()
            env_id = self.current_player_id
            if self.current_player_id > self.train_id:
                env_id -= 1
            valid = False
            for _ in range(self.max_env_try):
                env_action = self.env_player.act(prompt_for_env_player, env_id)
                valid = self._process_env_action(self.current_player_id, env_action)
                if valid:
                    # 如果有效，不用重复提问
                    break
            # 如果始终没有有效动作——env_invalid_output，这一局提前结束
            if not valid:
                print('env invalid output!')
                info = {'action_is_effective': False, 'action_is_valid': True, 'success': False}
                self.history.append({
                    "action": action,
                })
                prompt = 'Invalid env output. End. '
                return (prompt, 0, True, info)
            self._move_to_next_player() # 这一轮结束，确定下一个玩家ID信息，相当于下标+1
            self._check_and_transition_phase() # check游戏阶段

        # 3. 判断是否结束，得到这轮的output信息
        if self.game_over:
            trainer_identity = self.identities[self.train_id]
            success = (self.winner == trainer_identity)
            reward = int(success)
            info = {'action_is_effective': False, 'action_is_valid': True, 'success': success}
            done = True
            prompt = self.render()
        # elif self.train_id not in self.alive_players:
        # 不会进入这个状态——投票之后直接查看最终结果，决定游戏是否结束
        #     # trainer死了，游戏结束
        #     prompt = 'Wrong voting. Game over.'
        #     done = True
        #     reward = 0
        #     info = {'action_is_effective': False, 'action_is_valid': True, 'success': False}
        else:
            # 正常进行下一轮
            prompt = self.render()
            done = False
            reward = 0
            info = {'action_is_effective': False, 'action_is_valid': True, 'success': False}
        
        return prompt, reward, done, info

    def render(self) -> str:
        """
        Generates the prompt for the current player based on the game state.
        Need to check the game phase, discuss or vote.
        Returns:
            A formatted string containing all necessary information for the player to act.
        """
        player_id = self.current_player_id
        # choose init prompt randomly
        init_prompt = random.choice(self.init_prompts)

        # --- History Information ---
        history_str = self._history_render()

        # --- Turn-specific Prompt ---
        turn_prompt = f"## Your Turn\n"
        turn_prompt += f"You are Player {player_id + 1}.\n"
        # 谁是卧底不应该告知玩家身份，删除这个prompt信息
        # turn_prompt += f"Your identity is: **{self.identities[player_id]}**.\n"
        turn_prompt += f"Your secret word is: **{self.words[player_id]}**.\n\n"
        # 添加不同身份游戏策略提示
        if self.phase == 'description':
            turn_prompt += "Description Phase. \n"
            turn_prompt += """### Additional Rules for Description (Very Important)\n- Your description MUST be clearly different from any descriptions you have given in earlier rounds.\n- Do NOT reuse similar words, sentence structures, or ideas. Avoid describing it as a "process" again.\n- Each round, pick a NEW angle of interpretation (e.g., its effect, its form, its symbolism, its usage, etc.)\n- The new description should have LOW semantic similarity with your previous descriptions.\n- Act as if you don't remember your previous answers, but you MUST ensure this answer is not similar to them.\n- Always produce a new, creative, and distinct sentence.\n- If the word has multiple meanings, assume 100% that the basic meaning is intended. Never choose the less common or technical ones."""
            # turn_prompt += "Your description must be plausible but vague. **Do NOT describe your word perfectly**. Try to guess the other word and describe a feature that both words share.\n"
            # turn_prompt += "Describe your word in a single sentence and do not repeat.\n"
        elif self.phase == 'vote':
            if self.identities[player_id] == 'Civilian': 
                turn_prompt += "Vote Phase. Vote for the undercover agent. Listen for descriptions that are too vague, too specific, or contradictory. "
            else:
                turn_prompt += "Vote Phase. Vote strategically to deflect suspicion from yourself. Choose a civilian who seems suspicious based on their descriptions, while blending in with the group's reasoning."
            turn_prompt += "Available players to vote for are:\n"
            # 默认不会投给自己
            vote_options = [p for p in self.alive_players if p != player_id]
            for p_id in vote_options:
                turn_prompt += f"Player {p_id + 1}, "
            turn_prompt += f"\nPlease state your vote clearly: I vote for Player [Player ID]."
        
        prompt = f"""
{init_prompt}

{turn_prompt}

## History Conversation
{history_str}
"""
        if self.current_player_id != self.train_id:
            prompt += "Always output: <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: 50 words (tokens)."
        return prompt.strip()

    def _move_to_next_player(self):
        """Cycles to the next alive player. 
        Only change `self.current_player_id`"""
        assert self.current_player_id in self.alive_players
        # print(self.current_player_id)
        if self.current_player_id == self.alive_players[-1]:
            # 但如果是这样意味着一轮已经结束——是否需要设置一个结束的标志？
            self.current_player_id = self.alive_players[0]
        else:
            # 找到下一个需要访问的Player对应ID信息
            try:
                id = self.alive_players.index(self.current_player_id)
            except ValueError:
                print('[ERROR]Current Player not in alive_player list! DEBUG!!!')
                exit(1)
            self.current_player_id = self.alive_players[id + 1]
        # print(self.current_player_id)

    def _process_action(self, player_id: int, action: str) -> bool:
        """
        Updates game state based on a player's action.
        if action is valid, return True; else return False
        """
        # For simplicity, assume `action` is the direct answer.
        if self.phase == 'description':
            # 保存历史信息
            # 不能没有消息，判定一下action不能为空
            if not action.strip():
                return False
            self.conversation[player_id].append(action.strip())
            return True
        elif self.phase == 'vote':
            match = re.search(r'player (\d+)', action.lower())
            if match:
                voted_player_num = int(match.group(1))
                # -1，和真实的index相对应
                voted_id = voted_player_num - 1
                if voted_id in self.alive_players and voted_id != player_id:
                    # 不能投票给自己
                    self.votes.append(voted_id)
                    return True
            return False

    def _process_env_action(self, player_id: int, action: str) -> bool:
        """
        Updates game state based on the env_player's action.
        If action is valid, return True; else return False.
        
        """
        # For simplicity, assume `action` is the direct answer.
        pattern = r"<answer>(.*)</answer>"
        match = re.search(pattern, action, re.DOTALL)
        if match is None:
            return False
        action = match.group(1).strip()
        # print(f'extract action: {action}')
        if self.phase == 'description':
            self.conversation[player_id].append(action.strip())
            return True
        elif self.phase == 'vote':
            # 投票需要输出玩家ID信息
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
        # 在current_player_id更新后调用——所以只需要确定模型是否完成description或者vote过程
        is_start_of_circle = (self.current_player_id == self.alive_players[0])
        
        if self.phase == 'description' and is_start_of_circle:
            self.round += 1  # 增加发言轮次计数
            if self.round >= 2:  # 两轮发言后进入投票阶段
                self.phase = 'vote'
                # print('change phase to vote after 2 description rounds')
            # 否则继续下一轮发言，不改变phase
        elif self.phase == 'vote' and is_start_of_circle:
            # 所有人都完成投票了
            self._tally_votes_and_eliminate()
            self._check_game_over()
            # 投票后游戏结束，不再开始新一轮

    def _tally_votes_and_eliminate(self):
        """Counts votes, eliminates a player, and records it."""
        if len(self.votes) == 0: 
            print('wrong process! DEBUG!!!')
            exit(1)
        # print('vote info:', self.votes)
        vote_counts = {p_id: 0 for p_id in self.alive_players}
        for voted_id in self.votes:
            if voted_id in vote_counts:
                vote_counts[voted_id] += 1
            else:
                # vote has been checked in `process_action` function
                print('error! vote not valid! DEBUG!')
                exit(1)
        # print('vote counts:', vote_counts)
        max_votes = max([k for k in vote_counts.values()])
        eliminated_candidates = [e for e in vote_counts.keys() 
                                    if vote_counts[e] == max_votes]
        # print('eliminated candidates:', eliminated_candidates)
        # Handle ties by randomly choosing one to eliminate
        eliminated_id = random.choice(eliminated_candidates)
        # print('eliminated id:', eliminated_id)
        self.alive_players.remove(eliminated_id)

    def _check_game_over(self):
        """Checks for win/loss conditions and sets game_over flag."""
        # 因为是最后调用，所以可以直接设定游戏状态为结束状态
        undercover_alive = (self.undercover_id in self.alive_players)
        # print('undercover_id:', self.undercover_id)
        if not undercover_alive:
            self.game_over = True
            self.winner = "Civilian"
            print('Game ends. Civilian win.')
        else:
            self.game_over = True
            self.winner = "Undercover"
            print('Game ends. Undercover win.')

    def _start_new_round(self):
        """Resets state for a new round of descriptions and votes."""
        # 在新流程中不再需要这个方法，因为游戏在投票后直接结束
        # 保留此方法以避免可能的调用错误
        pass

    def _history_render(self) -> str:
        """"Generate description for history conversation information."""
        # 第一轮第一个发言玩家没有历史信息
        if self.round == 0 and self.current_player_id == 0 and self.phase == 'description':
            return "There is no history information."
        info_str = ""
        # 按照Player整理历史发言信息
        # 打印所有历史发言or alive_player的发言？先设置为所有玩家
        for i in range(self.config.player_num):
        # for i in self.alive_players:
            if len(self.conversation[i]) == 0:
                # 如果没有发言历史，跳过
                continue
            info_str += f"Player {i + 1}: "
            for j in self.conversation[i]:
                info_str += j + ' '
            info_str += '\n'
        # 判断是否超出对话长度，如果过长直接删除
        if len(info_str.split()) > 300:
            print('simplify info')
            print(info_str)
            # 调用API进行历史对话的精简，并缩写历史记录中的玩家发言信息
            info_str = self.simplifier.simplify(info_str, prompt=self.simplify_prompt)
            # 拆分输出结果，重新保存为conversation信息
            # 使用正则匹配所有玩家及其描述
            pattern = r"[Pp]layer\s+(\d+):\s*(.+)"
            matches = re.findall(pattern, info_str)
            for num, info in matches:
                self.conversation[int(num) - 1] = [info.strip()]
            # print(self.conversation)
            # TODO：担心调用API会有错误输出，不知道这么搞会不会容易出BUG——记得测试！
        return info_str
    
    def close(self):
        """Close the environment."""
        del self.wordlist
        


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    config = UndercoverEnvConfig()
    env = UndercoverEnv(config)
    # print(env.reset(seed=42))
    r = 1
    while r:
        env.reset(seed=random.randint(0, 100))
        done = False
        while not done:
            print(env.render())
            keyboard = input("Enter action: ")
            if keyboard == 'q':
                r = 0
                break
            # action = int(keyboard)
            # assert action in env.ACTION_LOOKUP, f"Invalid action: {action}"
            obs, reward, done, info = env.step(keyboard)
            for o in [reward, done, info]:
                print(o)
        print('round over. ')
        time.sleep(5)