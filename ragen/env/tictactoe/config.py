from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class TicTacToeEnvConfig:
    rows: int = 3
    cols: int = 3
    win_condition: int = 3
    render_mode: str = "text"
    action_lookup: Dict[int, str] = None # defined in env.py\
    # 另一个玩家的默认设置，后续可能可以更改
    player_num: int = 2
    # TODO：后续需要修改，看看self-play需要如何设置
    player_info: List[Dict[str, Any]] = field(
        # default_factory=lambda: [{'model_name': 'deepseek'}]
        default_factory=lambda: [{'model_name': "gemini-2.5-flash-nothinking"}]
        # default_factory=lambda: [{'model_name': 'Qwen3-14B', 'port': '1414'}]
        # default_factory=lambda: [{'model_name': 'tq100', 'port': '100'}]
    )
    init_prompts: List[str] = field(
        default_factory=lambda: [
            "##Game Rules: TicTacToe\n**Objective**: Be the first player to connect 3 of your pieces in a continuous line.\n**Player Pieces**:\n- Player 1: 'O'\n- Player 2: 'X'\n- Empty Slot: '.'\n**How to Play**:\n1. The game is played on a 3x3 vertical grid.\n2. Players take turns setting one of their pieces into any available slot.\n",
            "Welcome to Tic-Tac-Toe! It's a classic game of wits. Your goal is simple: get three of your marks in a row before your opponent does. Player 1 is 'O' and Player 2 is 'X'. The board is a 3x3 grid where empty slots are represented by '.'. On your turn, just pick an empty spot to place your mark. A winning line can be straight across (horizontal), up and down (vertical), or slanted (diagonal). If the board fills up and no one has won, it's a draw. Good luck!",
            "## Tic-Tac-Toe Quick Rules\n- **Objective**: Form a line of 3 of your symbols (Player 1 is O and Player 2 is X).\n- **Board**: 3x3 grid.\n- **Gameplay**: Players alternate placing their symbol in an empty cell, representing by '.'.\n- **Win Conditions**: A line of 3 identical symbols horizontally, vertically, or diagonally.\n- **Draw**: The game is a draw if the board is full and no player has won.",
            "**About the Game: Tic-Tac-Toe**\n\n*What is the main goal?*\nTo be the first to align three of your pieces (For Player 1 is 'O' and for Player 2 is 'X') in an unbroken line.\n\n*How do I win?*\nA win is achieved by creating a sequence of three of your pieces. This can be in any row, any column, or across either of the two main diagonals.\n\n*How does a turn work?*\nWhen it's your turn, select any unoccupied square on the 3x3 board ('.') to place your piece.\n\n*What if no one wins?*\nIf all nine squares are filled and no winning line has been formed, the game ends in a tie."
    ]
    )
    model_path: str = "/root/autodl-fs"  # "/data1/lvnuoyan/llm_model"
    temperature: float = 0.5
    seed: int = 123
    max_env_try: int = 3
