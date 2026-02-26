from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

SPECIAL = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']

@dataclass
class Tokenizer:
    vocab: Dict[str, int]
    ivocab: Dict[int, str]

    @property
    def pad_id(self) -> int: return self.vocab['<PAD>']
    @property
    def bos_id(self) -> int: return self.vocab['<BOS>']
    @property
    def eos_id(self) -> int: return self.vocab['<EOS>']
    @property
    def unk_id(self) -> int: return self.vocab.get('<UNK>', 0)

    def encode(self, seq: List[Union[int, str]], grade: int | None = None) -> List[int]:
        toks = [self.bos_id]
        if grade is not None:
            toks.append(self.vocab.get(f'<G{grade}>', self.unk_id))
        
        for item in seq:
            toks.append(self.vocab.get(str(item), self.unk_id))
            
        toks.append(self.eos_id)
        return toks

    def decode(self, token_ids: List[int]) -> List[str]:
        out = []
        for tid in token_ids:
            s = self.ivocab.get(tid, '<UNK>')
            if s in SPECIAL or s.startswith('<G'):
                continue
            out.append(s)
        return out

def build_action_tokenizer(rows: int, cols: int, max_grade: int) -> Tokenizer:
    """
    针对【生物力学动作序列】构建全集词表 (暴力全排列，防止 OOV)
    """
    vocab = {}
    idx = 0
    
    for sp in SPECIAL:
        vocab[sp] = idx; idx += 1
    for g in range(max_grade + 1):
        vocab[f'<G{g}>'] = idx; idx += 1
        
    for h in range(rows * cols):
        vocab[f'START_H{h}'] = idx; idx += 1
        
    # 加入左右手前缀
    action_types = ["MOVE", "DYNO", "LOCK", "CROSS"]
    hands = ["LH", "RH"]
    
    for hand in hands:
        for atype in action_types:
            for dr in range(-rows, rows + 1):
                for dc in range(-cols, cols + 1):
                    vocab[f'{hand}_{atype}_R{dr:+d}_C{dc:+d}'] = idx; idx += 1
                
    ivocab = {v: k for k, v in vocab.items()}
    return Tokenizer(vocab=vocab, ivocab=ivocab)

def save_tokenizer(tok: Tokenizer, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(tok.vocab, ensure_ascii=False, indent=2), encoding='utf-8')

def load_tokenizer(path: str) -> Tokenizer:
    vocab = json.loads(Path(path).read_text(encoding='utf-8'))
    ivocab = {v: k for k, v in vocab.items()}
    return Tokenizer(vocab=vocab, ivocab=ivocab)