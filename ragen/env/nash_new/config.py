# nashenv/config.py
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

@dataclass
class NashNewConfig:
    force_role: Optional[str] = None
    seed: int = 123
    render_mode: str = 'text'
    # 定义pay_off_matrix数据类型
    PayoffMatrixList = List[List[int]] 
    GameDataList = Tuple[PayoffMatrixList, PayoffMatrixList]
    payoff_list: List[GameDataList] = field(default_factory=lambda:[
        # prison dilemma
        ([[3,0],[5,1]],[[3,5],[0,1]]),
        # battle of sex
        ([[2,0],[0,1]],[[1,0],[0,2]]),
        # game of chicken
        ([[-10,1],[-1,0]],[[-10,-1],[1,0]]),
        # stag hunt
        ([[3,0],[1,1]],[[3,1],[0,1]]),
        # radio station
        ([[25,50,50],[30,15,30],[20,20,10]],[[25,30,20],[50,15,20],[50,30,10]]),
        # IESDS
        ([[13,1,7],[4,3,6],[-1,2,8]],[[3,4,3],[1,3,2],[2,3,-1]]),
        # duo-polistic
        ([[0,0,0,0,0,0],[9,7,5,3,1,-1],[14,10,6,2,-2,-2],[15,9,3,-3,-3,-3],[12,4,-4,-4,-4,-4],[5,-5,-5,-5,-5,-5]],
         [[0,9,14,15,12,5],[0,7,10,9,4,-5],[0,5,6,3,-4,-5],[0,3,2,-3,-4,-5],[0,1,-2,-3,-4,-5],[0,-1,-2,-3,-4,-5]]),
        # GAME
        ([[1,-1,5,1],[2,1,3,5],[1,0,1,0]],[[1,2,0,1],[3,2,0,1],[1,5,7,1]]),
        # weakly dominated game
        ([[5,4],[6,3],[6,4]],[[1,0],[0,1],[4,1]])
    ])