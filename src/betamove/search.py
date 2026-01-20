from __future__ import annotations
from collections import deque
from typing import List, Optional
from src.env.board import Board
from src.betamove.constraints import Constraints, is_valid_step

def bfs_find_sequence(board: Board, holds_set: set[int], start: int, end: int, cons: Constraints) -> Optional[List[int]]:
    # Graph nodes are holds in route; edge if step valid
    q = deque([start])
    prev = {start: None}

    while q:
        cur = q.popleft()
        if cur == end:
            break
        for nxt in holds_set:
            if nxt in prev:
                continue
            if is_valid_step(board, cur, nxt, cons):
                prev[nxt] = cur
                q.append(nxt)

    if end not in prev:
        return None

    # reconstruct
    seq = []
    x = end
    while x is not None:
        seq.append(x)
        x = prev[x]
    seq.reverse()
    return seq
