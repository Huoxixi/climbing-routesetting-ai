# ==========================================================
# bootstrap_mvp.ps1
# Fill project with a strict, runnable MVP pipeline:
# MoonBoard-like grid -> BetaMove -> GradeNet -> DeepRouteSet -> Pipeline
# ==========================================================

$ErrorActionPreference = "Stop"
Write-Host "[bootstrap] Writing MVP code into src/ ..."

# ---------- helper ----------
function WriteFile($path, $content) {
    $dir = Split-Path $path -Parent
    if ($dir -and !(Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    Set-Content -Path $path -Value $content -Encoding UTF8
}

# ---------- configs/default.yaml ----------
WriteFile "configs/default.yaml" @"
project:
  name: climbing-routesetting-ai
  seed: 42
  device: cpu

board:
  type: moonboard
  rows: 18
  cols: 11
  n_holds: 142   # keep for compatibility; MVP uses rows*cols mapping

data:
  toy_n_routes: 200
  train_ratio: 0.8
  val_ratio: 0.1
  max_seq_len: 24

betamove:
  max_reach: 4.5      # in grid-distance units
  require_monotonic_up: true

training:
  batch_size: 32
  lr: 0.001
  epochs: 5
  num_workers: 0

model:
  embed_dim: 64
  hidden_dim: 128
  num_layers: 1

generation:
  samples_per_grade: 50
  temperature: 1.0
  top_k: 10
"@

# ---------- src/common/seed.py ----------
WriteFile "src/common/seed.py" @"
from __future__ import annotations
import os
import random
import numpy as np

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
"@

# ---------- src/common/logging.py ----------
WriteFile "src/common/logging.py" @"
from __future__ import annotations
import logging
from pathlib import Path

def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter('[%(asctime)s][%(levelname)s][%(name)s] %(message)s')

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
"@

# ---------- src/common/paths.py ----------
WriteFile "src/common/paths.py" @"
from __future__ import annotations
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

@dataclass(frozen=True)
class RunDir:
    root: Path
    config_snapshot: Path
    meta: Path
    metrics: Path
    artifacts: Path

def _git_commit() -> str:
    try:
        out = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.STDOUT)
        return out.decode().strip()
    except Exception:
        return 'nogit'

def make_run_dir(tag: str) -> RunDir:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    commit = _git_commit()
    root = Path('outputs') / 'runs' / f'{ts}_{tag}_{commit}'
    root.mkdir(parents=True, exist_ok=True)

    artifacts = root / 'artifacts'
    artifacts.mkdir(parents=True, exist_ok=True)

    return RunDir(
        root=root,
        config_snapshot=root / 'config.yaml',
        meta=root / 'meta.json',
        metrics=root / 'metrics.json',
        artifacts=artifacts,
    )

def write_meta(run: RunDir, extra: dict | None = None) -> None:
    import sys
    meta = {
        'python': sys.version,
        'platform': platform.platform(),
        'git_commit': _git_commit(),
    }
    if extra:
        meta.update(extra)
    run.meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
"@

# ---------- src/env/board.py ----------
WriteFile "src/env/board.py" @"
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
"@

# ---------- src/env/route.py ----------
WriteFile "src/env/route.py" @"
from __future__ import annotations
from dataclasses import dataclass
from typing import List

@dataclass
class Route:
    holds: List[int]        # unordered set in raw; ordered seq after BetaMove
    grade: int              # integer grade bucket (toy)
    start: int
    end: int
"@

# ---------- src/betamove/constraints.py ----------
WriteFile "src/betamove/constraints.py" @"
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
"@

# ---------- src/betamove/search.py ----------
WriteFile "src/betamove/search.py" @"
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
"@

# ---------- src/betamove/betamove.py ----------
WriteFile "src/betamove/betamove.py" @"
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
"@

# ---------- src/data/tokenizer.py ----------
WriteFile "src/data/tokenizer.py" @"
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
"@

# ---------- src/data/dataset.py ----------
WriteFile "src/data/dataset.py" @"
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
"@

# ---------- src/data/preprocess.py ----------
WriteFile "src/data/preprocess.py" @"
from __future__ import annotations
import argparse, json, random
from pathlib import Path

import yaml

from src.common.seed import set_seed
from src.env.board import Board
from src.betamove.constraints import Constraints
from src.betamove.betamove import run_betamove
from src.data.tokenizer import build_tokenizer, save_tokenizer

def gen_toy_routes(board: Board, n: int, max_grade: int, max_holds: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    routes = []
    for i in range(n):
        grade = rng.randint(0, max_grade)
        # start near bottom, end near top
        start = board.to_id(rng.randint(0, 2), rng.randint(0, board.cols - 1))
        end = board.to_id(rng.randint(board.rows - 3, board.rows - 1), rng.randint(0, board.cols - 1))
        k = rng.randint(6, max_holds)
        holds = set()
        while len(holds) < k:
            r = rng.randint(0, board.rows - 1)
            c = rng.randint(0, board.cols - 1)
            holds.add(board.to_id(r, c))
        holds = list(holds)
        routes.append({'id': f'toy_{i}', 'grade': grade, 'start': start, 'end': end, 'holds': holds})
    return routes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    seed = int(cfg['project']['seed'])
    set_seed(seed)

    board = Board(rows=int(cfg['board']['rows']), cols=int(cfg['board']['cols']))
    max_seq_len = int(cfg['data']['max_seq_len'])

    raw_dir = Path('data/raw')
    proc_dir = Path('data/processed')
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    max_grade = 9  # toy grade buckets 0..9

    routes = gen_toy_routes(
        board=board,
        n=int(cfg['data']['toy_n_routes']),
        max_grade=max_grade,
        max_holds=max_seq_len,
        seed=seed
    )

    # Save raw
    raw_path = raw_dir / 'toy_routes.jsonl'
    with raw_path.open('w', encoding='utf-8') as f:
        for r in routes:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # BetaMove: convert set -> feasible ordered seq (or mark failed)
    cons = Constraints(
        max_reach=float(cfg['betamove']['max_reach']),
        require_monotonic_up=bool(cfg['betamove']['require_monotonic_up'])
    )

    processed = []
    for r in routes:
        bm = run_betamove(board, r['holds'], r['start'], r['end'], cons)
        if not bm.success:
            continue
        processed.append({
            'id': r['id'],
            'grade': r['grade'],
            'seq': bm.seq
        })

    # Split
    rng = random.Random(seed)
    rng.shuffle(processed)
    n = len(processed)
    n_train = int(n * float(cfg['data']['train_ratio']))
    n_val = int(n * float(cfg['data']['val_ratio']))
    train = processed[:n_train]
    val = processed[n_train:n_train+n_val]
    test = processed[n_train+n_val:]

    for name, arr in [('train', train), ('val', val), ('test', test)]:
        p = proc_dir / f'{name}.jsonl'
        with p.open('w', encoding='utf-8') as f:
            for r in arr:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Tokenizer
    tok = build_tokenizer(n_holds=board.n_holds, max_grade=max_grade)
    save_tokenizer(tok, str(proc_dir / 'tokenizer_vocab.json'))

    # Minimal report
    report = {
        'raw_routes': len(routes),
        'betamove_success': len(processed),
        'train': len(train),
        'val': len(val),
        'test': len(test),
        'board_n_holds': board.n_holds,
        'max_grade': max_grade,
        'max_seq_len': max_seq_len
    }
    (proc_dir / 'preprocess_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[preprocess] Done:', json.dumps(report, ensure_ascii=False))

if __name__ == '__main__':
    main()
"@

# ---------- src/models/gradenet.py ----------
WriteFile "src/models/gradenet.py" @"
from __future__ import annotations
import torch
import torch.nn as nn

class GradeNet(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int, num_classes: int, pad_id: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        # x: [B,L], attn: [B,L]
        emb = self.embed(x)
        out, _ = self.rnn(emb)  # [B,L,2H]
        # masked mean pooling
        mask = attn.unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1.0))
        logits = self.head(pooled)
        return logits
"@

# ---------- src/models/deeprouteset.py ----------
WriteFile "src/models/deeprouteset.py" @"
from __future__ import annotations
import torch
import torch.nn as nn

class DeepRouteSet(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int, pad_id: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        # x_in: [B,L] input tokens
        emb = self.embed(x_in)
        out, _ = self.rnn(emb)
        logits = self.lm_head(out)  # [B,L,V]
        return logits

    @torch.no_grad()
    def generate(self, bos: int, eos: int, prefix: list[int], max_len: int, temperature: float = 1.0, top_k: int = 0) -> list[int]:
        device = next(self.parameters()).device
        seq = prefix[:]  # already includes BOS and maybe grade token
        while len(seq) < max_len:
            x = torch.tensor([seq], dtype=torch.long, device=device)
            logits = self.forward(x)[:, -1, :] / max(temperature, 1e-6)
            if top_k and top_k > 0:
                v, idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                probs = torch.softmax(v, dim=-1)
                next_id = idx[0, torch.multinomial(probs[0], 1).item()].item()
            else:
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs[0], 1).item()
            seq.append(int(next_id))
            if next_id == eos:
                break
        return seq
"@

# ---------- src/train/train_gradenet.py ----------
WriteFile "src/train/train_gradenet.py" @"
from __future__ import annotations
import argparse, json
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

from src.common.seed import set_seed
from src.common.logging import get_logger
from src.common.paths import make_run_dir, write_meta
from src.data.tokenizer import load_tokenizer
from src.data.dataset import SeqDataset, SeqSample
from src.models.gradenet import GradeNet

def load_split(path: str, tok, max_len: int) -> list[SeqSample]:
    out = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        x = tok.encode(r['seq'], grade=None)  # GradeNet does NOT need grade token in input
        out.append(SeqSample(x=x, y=int(r['grade'])))
    return out

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, attn, y in loader:
            x, attn, y = x.to(device), attn.to(device), y.to(device)
            logits = model(x, attn)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    set_seed(int(cfg['project']['seed']))

    run = make_run_dir('gradenet')
    run.config_snapshot.write_text(Path(args.config).read_text(encoding='utf-8'), encoding='utf-8')
    write_meta(run)

    logger = get_logger('train_gradenet', str(run.root / 'stdout.log'))
    device = torch.device(cfg['project']['device'])

    proc = Path('data/processed')
    tok = load_tokenizer(str(proc / 'tokenizer_vocab.json'))

    max_len = int(cfg['data']['max_seq_len'])
    train_samples = load_split(str(proc / 'train.jsonl'), tok, max_len)
    val_samples = load_split(str(proc / 'val.jsonl'), tok, max_len)

    train_ds = SeqDataset(train_samples, pad_id=tok.pad_id, max_len=max_len)
    val_ds = SeqDataset(val_samples, pad_id=tok.pad_id, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=int(cfg['training']['batch_size']), shuffle=True, num_workers=int(cfg['training']['num_workers']))
    val_loader = DataLoader(val_ds, batch_size=int(cfg['training']['batch_size']), shuffle=False, num_workers=int(cfg['training']['num_workers']))

    num_classes = 10  # toy grades 0..9
    model = GradeNet(
        vocab_size=len(tok.vocab),
        embed_dim=int(cfg['model']['embed_dim']),
        hidden_dim=int(cfg['model']['hidden_dim']),
        num_layers=int(cfg['model']['num_layers']),
        num_classes=num_classes,
        pad_id=tok.pad_id
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg['training']['lr']))
    loss_fn = torch.nn.CrossEntropyLoss()

    logger.info(f'train={len(train_ds)} val={len(val_ds)} vocab={len(tok.vocab)} device={device}')

    best_val = 0.0
    for epoch in range(1, int(cfg['training']['epochs']) + 1):
        model.train()
        total_loss = 0.0
        for x, attn, y in train_loader:
            x, attn, y = x.to(device), attn.to(device), y.to(device)
            logits = model(x, attn)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        val_acc = evaluate(model, val_loader, device)
        logger.info(f'epoch={epoch} loss={total_loss/ max(len(train_loader),1):.4f} val_acc={val_acc:.3f}')

        if val_acc >= best_val:
            best_val = val_acc
            ckpt = {'state_dict': model.state_dict(), 'vocab_size': len(tok.vocab), 'pad_id': tok.pad_id}
            torch.save(ckpt, run.root / 'gradenet.pt')

    metrics = {'best_val_acc': best_val}
    run.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    logger.info(f'saved: {run.root / \"gradenet.pt\"}')

if __name__ == '__main__':
    main()
"@

# ---------- src/train/train_deeprouteset.py ----------
WriteFile "src/train/train_deeprouteset.py" @"
from __future__ import annotations
import argparse, json
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader, Dataset

from src.common.seed import set_seed
from src.common.logging import get_logger
from src.common.paths import make_run_dir, write_meta
from src.data.tokenizer import load_tokenizer
from src.models.deeprouteset import DeepRouteSet

class LMDataset(Dataset):
    def __init__(self, items, pad_id: int, max_len: int):
        self.items = items
        self.pad_id = pad_id
        self.max_len = max_len

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        x = self.items[i][:self.max_len]
        # language modeling: predict next token
        x_in = x[:-1]
        x_tg = x[1:]
        # pad
        if len(x_in) < self.max_len - 1:
            pad_n = (self.max_len - 1) - len(x_in)
            x_in = x_in + [self.pad_id]*pad_n
            x_tg = x_tg + [self.pad_id]*pad_n
        return torch.tensor(x_in, dtype=torch.long), torch.tensor(x_tg, dtype=torch.long)

def load_sequences(path: str, tok, with_grade: bool, max_len: int):
    seqs = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        grade = int(r['grade']) if with_grade else None
        seqs.append(tok.encode(r['seq'], grade=grade))
    return seqs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    set_seed(int(cfg['project']['seed']))

    run = make_run_dir('deeprouteset')
    run.config_snapshot.write_text(Path(args.config).read_text(encoding='utf-8'), encoding='utf-8')
    write_meta(run)

    logger = get_logger('train_deeprouteset', str(run.root / 'stdout.log'))
    device = torch.device(cfg['project']['device'])

    proc = Path('data/processed')
    tok = load_tokenizer(str(proc / 'tokenizer_vocab.json'))

    max_len = int(cfg['data']['max_seq_len'])
    train_seqs = load_sequences(str(proc / 'train.jsonl'), tok, with_grade=True, max_len=max_len)

    ds = LMDataset(train_seqs, pad_id=tok.pad_id, max_len=max_len)
    loader = DataLoader(ds, batch_size=int(cfg['training']['batch_size']), shuffle=True, num_workers=int(cfg['training']['num_workers']))

    model = DeepRouteSet(
        vocab_size=len(tok.vocab),
        embed_dim=int(cfg['model']['embed_dim']),
        hidden_dim=int(cfg['model']['hidden_dim']),
        num_layers=int(cfg['model']['num_layers']),
        pad_id=tok.pad_id
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg['training']['lr']))
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tok.pad_id)

    logger.info(f'train_seqs={len(ds)} vocab={len(tok.vocab)} device={device}')

    for epoch in range(1, int(cfg['training']['epochs']) + 1):
        model.train()
        total_loss = 0.0
        for x_in, x_tg in loader:
            x_in, x_tg = x_in.to(device), x_tg.to(device)
            logits = model(x_in)  # [B,L,V]
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), x_tg.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        logger.info(f'epoch={epoch} loss={total_loss / max(len(loader),1):.4f}')

    ckpt = {'state_dict': model.state_dict(), 'vocab_size': len(tok.vocab), 'pad_id': tok.pad_id}
    torch.save(ckpt, run.root / 'deeprouteset.pt')
    run.metrics.write_text(json.dumps({'done': True}, ensure_ascii=False, indent=2), encoding='utf-8')
    logger.info(f'saved: {run.root / \"deeprouteset.pt\"}')

if __name__ == '__main__':
    main()
"@

# ---------- src/pipeline/generate_and_filter.py ----------
WriteFile "src/pipeline/generate_and_filter.py" @"
from __future__ import annotations
import argparse, json
from pathlib import Path

import yaml
import torch

from src.env.board import Board
from src.betamove.constraints import Constraints
from src.betamove.betamove import run_betamove
from src.data.tokenizer import load_tokenizer
from src.models.deeprouteset import DeepRouteSet
from src.models.gradenet import GradeNet

def load_latest_run(tag: str, ckpt_name: str) -> Path:
    runs = sorted((Path('outputs') / 'runs').glob(f'*_{tag}_*'))
    if not runs:
        raise FileNotFoundError(f'No runs found for tag={tag}. Run training first.')
    last = runs[-1]
    ckpt = last / ckpt_name
    if not ckpt.exists():
        raise FileNotFoundError(f'Checkpoint missing: {ckpt}')
    return ckpt

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--target_grade', type=int, required=True)
    ap.add_argument('--out', default='outputs/routes/generated_routes.jsonl')
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    device = torch.device(cfg['project']['device'])

    board = Board(rows=int(cfg['board']['rows']), cols=int(cfg['board']['cols']))
    cons = Constraints(max_reach=float(cfg['betamove']['max_reach']), require_monotonic_up=bool(cfg['betamove']['require_monotonic_up']))

    proc = Path('data/processed')
    tok = load_tokenizer(str(proc / 'tokenizer_vocab.json'))

    # Load checkpoints (latest)
    dr_ckpt_path = load_latest_run('deeprouteset', 'deeprouteset.pt')
    gn_ckpt_path = load_latest_run('gradenet', 'gradenet.pt')

    dr_ckpt = torch.load(dr_ckpt_path, map_location=device)
    gn_ckpt = torch.load(gn_ckpt_path, map_location=device)

    deeprouteset = DeepRouteSet(
        vocab_size=dr_ckpt['vocab_size'],
        embed_dim=int(cfg['model']['embed_dim']),
        hidden_dim=int(cfg['model']['hidden_dim']),
        num_layers=int(cfg['model']['num_layers']),
        pad_id=tok.pad_id
    ).to(device)
    deeprouteset.load_state_dict(dr_ckpt['state_dict'])
    deeprouteset.eval()

    gradenet = GradeNet(
        vocab_size=gn_ckpt['vocab_size'],
        embed_dim=int(cfg['model']['embed_dim']),
        hidden_dim=int(cfg['model']['hidden_dim']),
        num_layers=int(cfg['model']['num_layers']),
        num_classes=10,
        pad_id=tok.pad_id
    ).to(device)
    gradenet.load_state_dict(gn_ckpt['state_dict'])
    gradenet.eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = int(cfg['generation']['samples_per_grade'])
    max_len = int(cfg['data']['max_seq_len'])
    temperature = float(cfg['generation']['temperature'])
    top_k = int(cfg['generation']['top_k'])

    kept = []
    for i in range(samples):
        prefix = [tok.bos_id, tok.vocab[f'<G{args.target_grade}>']]
        gen_ids = deeprouteset.generate(tok.bos_id, tok.eos_id, prefix=prefix, max_len=max_len, temperature=temperature, top_k=top_k)
        seq = tok.decode(gen_ids)

        if len(seq) < 4:
            continue

        start = seq[0]
        end = seq[-1]
        bm = run_betamove(board, holds=seq, start=start, end=end, cons=cons)
        if not bm.success:
            continue

        # GradeNet scoring on BetaMove-ordered seq
        x = tok.encode(bm.seq, grade=None)
        # pad to max_len
        x = x[:max_len] + [tok.pad_id] * max(0, max_len - len(x))
        attn = [1 if t != tok.pad_id else 0 for t in x]
        xt = torch.tensor([x], dtype=torch.long, device=device)
        at = torch.tensor([attn], dtype=torch.long, device=device)
        logits = gradenet(xt, at)
        pred = int(logits.argmax(dim=-1).item())

        kept.append({
            'target_grade': args.target_grade,
            'pred_grade': pred,
            'seq_raw': seq,
            'seq_betamove': bm.seq
        })

    # sort by |pred-target|
    kept.sort(key=lambda r: abs(r['pred_grade'] - r['target_grade']))

    with out_path.open('w', encoding='utf-8') as f:
        for r in kept[:10]:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f'[pipeline] target_grade={args.target_grade} generated={samples} kept={len(kept)} saved_top10={out_path}')

if __name__ == '__main__':
    main()
"@

# ---------- src/viz/plot_board.py ----------
WriteFile "src/viz/plot_board.py" @"
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import matplotlib.pyplot as plt

from src.env.board import Board

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', default='outputs/figures/board.png')
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    board = Board(rows=int(cfg['board']['rows']), cols=int(cfg['board']['cols']))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f'Board Grid: {board.rows}x{board.cols}')
    ax.set_xlim(-0.5, board.cols - 0.5)
    ax.set_ylim(-0.5, board.rows - 0.5)
    ax.set_xticks(range(board.cols))
    ax.set_yticks(range(board.rows))
    ax.grid(True)
    ax.invert_yaxis()  # row 0 at top visually (optional)
    fig.savefig(args.out, dpi=160, bbox_inches='tight')
    print('[viz] saved', args.out)

if __name__ == '__main__':
    main()
"@

# ---------- src/viz/plot_route.py ----------
WriteFile "src/viz/plot_route.py" @"
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
import matplotlib.pyplot as plt

from src.env.board import Board

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--routes', default='outputs/routes/generated_routes.jsonl')
    ap.add_argument('--out', default='outputs/figures/route.png')
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    board = Board(rows=int(cfg['board']['rows']), cols=int(cfg['board']['cols']))

    lines = Path(args.routes).read_text(encoding='utf-8').splitlines()
    if not lines:
        raise RuntimeError('No routes to plot. Run pipeline first.')

    r0 = json.loads(lines[0])
    seq = r0['seq_betamove']

    xs, ys = [], []
    for hid in seq:
        rr, cc = board.from_id(hid)
        xs.append(cc)
        ys.append(rr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f'Route (target={r0[\"target_grade\"]}, pred={r0[\"pred_grade\"]})')
    ax.set_xlim(-0.5, board.cols - 0.5)
    ax.set_ylim(-0.5, board.rows - 0.5)
    ax.set_xticks(range(board.cols))
    ax.set_yticks(range(board.rows))
    ax.grid(True)
    ax.invert_yaxis()

    ax.scatter(xs, ys)
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.text(x + 0.05, y + 0.05, str(i), fontsize=8)

    fig.savefig(args.out, dpi=160, bbox_inches='tight')
    print('[viz] saved', args.out)

if __name__ == '__main__':
    main()
"@

Write-Host "[bootstrap] Done. Next: run preprocess/train/pipeline commands."
