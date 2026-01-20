from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from src.env.board import Board
from src.betamove.constraints import Constraints
from src.betamove.search import bfs_find_sequence

@dataclass
class BetaMoveResult:
    success: bool
    seq: List[int]
    reason: str = ''

def run_betamove(board: Board, holds: list[int], start: int, end: int, cons: Constraints) -> BetaMoveResult:
    holds_set = set(holds)
    holds_set.add(start)
    holds_set.add(end)

    if start not in holds_set or end not in holds_set:
        return BetaMoveResult(False, [], 'start/end not in holds')

    seq = bfs_find_sequence(board, holds_set, start, end, cons)
    if seq is None:
        return BetaMoveResult(False, [], 'no feasible path under constraints')

    return BetaMoveResult(True, seq, '')
