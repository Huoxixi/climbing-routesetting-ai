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
