from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class UndercoverEnvConfig:
    render_mode: str = "text"
    action_lookup: Dict[int, str] = None # defined in env.py\
    # 总玩家数量，env_player数量+1
    player_num: int = 5
    # TODO：后续需要修改，看看self-play需要如何设置
    player_info: List[Dict[str, Any]] = field(
        # default_factory=lambda: [{'model_name': 'deepseek'}]
        # default_factory=lambda: [{'model_name': 'tictactoe/grpo/game_40', 'port': '4040'}]
        default_factory=lambda: 
            [{'model_name': 'deepseek'},{'model_name': 'deepseek'},
             {'model_name': 'deepseek'},{'model_name': 'deepseek'},]
            # [{"model_name": "x-ai/grok-4-fast"}, {"model_name": "x-ai/grok-4-fast"},
            #  {"model_name": "x-ai/grok-4-fast"}, {"model_name": "x-ai/grok-4-fast"},]
    )
    env_max_tokens: int = 600
    simplify_prompt: str = "\nSummarize the conversation history of each player in format: `Player x: [conversation]\n`. Keep the summary ** under 100 words. **"
    init_prompts: List[str] = field(
        default_factory=lambda: [
    # 版本1：简洁总结版
    """Game: Who's the Undercover Agent

Roles:
- 4 Civilians share one word.
- 1 Undercover has a related but different word.

Goal:
- Civilians: Find the undercover.
- Undercover: Stay hidden until only 2 players remain.

How to Play:
Each player describes their word in one sentence (without saying the word itself). 
Be subtle yet clear. After all speak, everyone votes out one player. The one with most votes is eliminated.

Win:
- Civilians win if the undercover is voted out.
- Undercover wins if only Undercover and two Civilians remain.""",

    # 版本2：侧重玩法逻辑版
    """Who's the Undercover Agent

Setup:
5 players, 4 get the same word, 1 gets a similar but different word.

Play:
In turn, each player gives a one-sentence clue about their word. 
Don't say the actual word. Be careful — clues too vague or too specific can expose you.

After all clues, everyone votes for who they think is the undercover. 
The player with most votes leaves the game.

Win:
- Civilians: Eliminate the undercover.
- Undercover: Survive until 3 players remain.""",

    # 版本3：更口语化叙述版
    """Who's the Undercover

There are 5 players: 4 civilians get one word, 1 undercover gets a related one. 
Everyone gives one short clue — can't use the word itself. Try to sound convincing!

Then, all players vote on who seems suspicious. 
Whoever gets the most votes is out.

Victory:
- Civilians win if they kick out the undercover.
- Undercover wins if they survive till the last three players.""",

    # 版本4：强调策略与规则要点版
    """Game Title: Who's the Undercover

Roles:
- 4 Civilians share the same secret word.
- 1 Undercover gets a similar but different word.

Round Rules:
Players give one-sentence hints about their word (no word repetition allowed). 
Hints must be unique and strategic — too clear, you help the undercover; too vague, you look suspicious.

After hints, players vote on who the undercover is. 
Highest votes = eliminated.

End:
- Civilians win if undercover is voted out.
- Undercover wins if they reach the final three.""",
]

    )
    
    temperature: float = 0.5
    seed: int = 123
    max_env_try: int = 3