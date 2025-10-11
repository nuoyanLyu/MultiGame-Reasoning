# nashenv/config.py
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class NashEnvConfig:
    custom_matrices: Optional[Tuple[Tuple[Tuple[int,int],Tuple[int,int]],
                                    Tuple[Tuple[int,int],Tuple[int,int]]]] = None
    action_labels_p1: Tuple[str, str] = ("A", "B")
    action_labels_p2: Tuple[str, str] = ("X", "Y")
    force_role: Optional[str] = None
    mode: str = "pure_nash_member"  
    seed: int = 123
    render_mode: str = 'text'

    def get_payoff_matrices(self, game="PD"):
        if self.custom_matrices is not None:
            return self.custom_matrices[0], self.custom_matrices[1]

        if game.upper() == "PD":
            A = ((3, 0),
                 (5, 1))  # P1 payoff
            B = ((3, 5),
                 (0, 1))  # P2 payoff
            return A, B

        if game.upper() == "SH":
            A = ((4, 0),
                 (3, 3))
            B = ((4, 3),
                 (0, 3))
            return A, B

        if game.upper() == "MP":
            A = ((1, -1),
                 (-1, 1))
            B = ((-1, 1),
                 (1, -1))
            return A, B

        raise ValueError(f"Unknown game type: {self.game}")