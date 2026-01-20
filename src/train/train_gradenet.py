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

    import inspect
    from src.common import paths as paths_mod

    print("[DEBUG] args.config =", args.config)
    print("[DEBUG] project.out_dir =", cfg.get("project", {}).get("out_dir"))
    print("[DEBUG] paths module file =", paths_mod.__file__)
    print("[DEBUG] make_run_dir signature =", inspect.signature(make_run_dir))


    out_dir = cfg.get('project', {}).get('out_dir')
    run = make_run_dir('gradenet', root=out_dir)

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
    logger.info(f"saved: {run.root / 'gradenet.pt'}")

if __name__ == '__main__':
    main()
