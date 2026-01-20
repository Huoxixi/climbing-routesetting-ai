from __future__ import annotations
from dataclasses import dataclass
from typing import List

@dataclass
class Route:
    holds: List[int]        # unordered set in raw; ordered seq after BetaMove
    grade: int              # integer grade bucket (toy)
    start: int
    end: int
