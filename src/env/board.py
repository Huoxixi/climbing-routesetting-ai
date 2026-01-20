from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Board:
    rows: int = 18
    cols: int = 11

    def to_id(self, r: int, c: int) -> int:
        return r * self.cols + c

    def from_id(self, hid: int) -> tuple[int, int]:
        r = hid // self.cols
        c = hid % self.cols
        return r, c

    @property
    def n_holds(self) -> int:
        return self.rows * self.cols
