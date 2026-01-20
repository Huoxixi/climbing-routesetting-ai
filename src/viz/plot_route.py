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
    ax.set_title(f"Route (target={r0['target_grade']}, pred={r0['pred_grade']})")

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
