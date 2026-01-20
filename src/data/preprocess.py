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
