from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import torch
from torch.utils.data import Dataset

@dataclass
class SeqSample:
    x: List[int]     # token ids
    y: int           # grade label

class SeqDataset(Dataset):
    def __init__(self, samples: List[SeqSample], pad_id: int, max_len: int):
        self.samples = samples
        self.pad_id = pad_id
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.samples[i]
        x = s.x[:self.max_len]
        attn = [1] * len(x)
        if len(x) < self.max_len:
            pad_n = self.max_len - len(x)
            x = x + [self.pad_id] * pad_n
            attn = attn + [0] * pad_n
        return torch.tensor(x, dtype=torch.long), torch.tensor(attn, dtype=torch.long), torch.tensor(s.y, dtype=torch.long)
