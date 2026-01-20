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

    out_dir = cfg.get('project', {}).get('out_dir')
    run = make_run_dir('deeprouteset', root=out_dir)
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
    logger.info(f"saved: {run.root / 'deeprouteset.pt'}")


if __name__ == '__main__':
    main()
