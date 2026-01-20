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
