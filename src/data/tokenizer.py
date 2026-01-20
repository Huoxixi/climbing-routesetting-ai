from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

SPECIAL = ['<PAD>', '<BOS>', '<EOS>']

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

    def encode(self, hold_seq: List[int], grade: int | None = None) -> List[int]:
        toks = [self.bos_id]
        if grade is not None:
            toks.append(self.vocab[f'<G{grade}>'])
        for h in hold_seq:
            toks.append(self.vocab[str(h)])
        toks.append(self.eos_id)
        return toks

    def decode(self, token_ids: List[int]) -> List[int]:
        out = []
        for tid in token_ids:
            s = self.ivocab.get(tid, '')
            if s in SPECIAL or s.startswith('<G'):
                continue
            if s.isdigit():
                out.append(int(s))
        return out

def build_tokenizer(n_holds: int, max_grade: int) -> Tokenizer:
    vocab = {}
    idx = 0
    for sp in SPECIAL:
        vocab[sp] = idx; idx += 1
    for g in range(max_grade + 1):
        vocab[f'<G{g}>'] = idx; idx += 1
    for h in range(n_holds):
        vocab[str(h)] = idx; idx += 1
    ivocab = {v: k for k, v in vocab.items()}
    return Tokenizer(vocab=vocab, ivocab=ivocab)

def save_tokenizer(tok: Tokenizer, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(tok.vocab, ensure_ascii=False, indent=2), encoding='utf-8')

def load_tokenizer(path: str) -> Tokenizer:
    vocab = json.loads(Path(path).read_text(encoding='utf-8'))
    ivocab = {v: k for k, v in vocab.items()}
    return Tokenizer(vocab=vocab, ivocab=ivocab)
