from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple
from src.env.board import Board

@dataclass(frozen=True)
class Constraints:
    max_reach: float = 4.5
    require_monotonic_up: bool = True

def dist(board: Board, a: int, b: int) -> float:
    ar, ac = board.from_id(a)
    br, bc = board.from_id(b)
    return math.hypot(ar - br, ac - bc)

def is_valid_step(board: Board, a: int, b: int, cons: Constraints) -> bool:
    if dist(board, a, b) > cons.max_reach:
        return False
    if cons.require_monotonic_up:
        ar, _ = board.from_id(a)
        br, _ = board.from_id(b)
        if br < ar:  # going down disallowed in MVP
            return False
    return True
