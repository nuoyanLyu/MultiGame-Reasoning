# from .alfworld.config import AlfredEnvConfig
# from .alfworld.env import AlfredTXTEnv
from .bandit.config import BanditEnvConfig
from .bandit.env import BanditEnv
from .countdown.config import CountdownEnvConfig
from .countdown.env import CountdownEnv
from .sokoban.config import SokobanEnvConfig
from .sokoban.env import SokobanEnv
from .frozen_lake.config import FrozenLakeEnvConfig
from .frozen_lake.env import FrozenLakeEnv
from .metamathqa.env import MetaMathQAEnv
from .metamathqa.config import MetaMathQAEnvConfig
from .connect4.env import Connect4Env
from .connect4.config import Connect4EnvConfig
from .tictactoe.env import TicTacToeEnv
from .tictactoe.config import TicTacToeEnvConfig
from .math_lv3to5.env import MathEnv
from .math_lv3to5.config import MathEnvConfig


REGISTERED_ENVS = {
    'bandit': BanditEnv,
    'countdown': CountdownEnv,
    'sokoban': SokobanEnv,
    'frozen_lake': FrozenLakeEnv,
    # 'alfworld': AlfredTXTEnv,
    'metamathqa': MetaMathQAEnv,
    'connect4': Connect4Env,
    'tictactoe': TicTacToeEnv,
    'math_lv3to5': MathEnv,
}

REGISTERED_ENV_CONFIGS = {
    'bandit': BanditEnvConfig,
    'countdown': CountdownEnvConfig,
    'sokoban': SokobanEnvConfig,
    'frozen_lake': FrozenLakeEnvConfig,
    # 'alfworld': AlfredEnvConfig,
    'metamathqa': MetaMathQAEnvConfig,
    'connect4': Connect4EnvConfig,
    'tictactoe': TicTacToeEnvConfig,
    'math_lv3to5': MathEnvConfig,
}

try:
    from .webshop.env import WebShopEnv
    from .webshop.config import WebShopEnvConfig
    REGISTERED_ENVS['webshop'] = WebShopEnv
    REGISTERED_ENV_CONFIGS['webshop'] = WebShopEnvConfig
except ImportError:
    pass
