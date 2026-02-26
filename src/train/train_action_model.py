from __future__ import annotations
import argparse
import json
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
        # 语言模型任务：用前 N-1 个动作，预测第 N 个动作
        x_in = x[:-1]
        x_tg = x[1:]
        
        # 补齐 (Padding)
        if len(x_in) < self.max_len - 1:
            pad_n = (self.max_len - 1) - len(x_in)
            x_in = x_in + [self.pad_id] * pad_n
            x_tg = x_tg + [self.pad_id] * pad_n
            
        return torch.tensor(x_in, dtype=torch.long), torch.tensor(x_tg, dtype=torch.long)

def load_sequences(jsonl_file, tok, with_grade=True, max_len=50):
    seqs = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            grade = r.get('grade', 3)
            
            # 【核心修复】：对接全新架构的 action_seq 字段
            raw_seq = r.get('action_seq', [])
            if not raw_seq: 
                continue
                
            seqs.append(tok.encode(raw_seq, grade=grade))
    return seqs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default="configs/phase2.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    set_seed(int(cfg['project']['seed']))

    # 创建新的输出目录
    out_dir = cfg.get('project', {}).get('out_dir', 'outputs/phase2')
    run = make_run_dir('action_model', root=out_dir)
    run.config_snapshot.write_text(Path(args.config).read_text(encoding='utf-8'), encoding='utf-8')
    write_meta(run)

    logger = get_logger('train_action_model', str(run.root / 'stdout.log'))
    device = torch.device(cfg['project']['device'])

    # ❗ 核心修改：读取新的 Action 数据集和词表
    proc = Path('data/processed_actions')
    tok = load_tokenizer(str(proc / 'action_tokenizer_vocab.json'))

    max_len = int(cfg['data']['max_seq_len'])
    train_seqs = load_sequences(str(proc / 'train_final.jsonl'), tok, with_grade=True, max_len=max_len)

    ds = LMDataset(train_seqs, pad_id=tok.pad_id, max_len=max_len)
    loader = DataLoader(ds, batch_size=int(cfg['training']['batch_size']), shuffle=True, num_workers=int(cfg['training']['num_workers']))

    # 初始化模型 (模型本身架构不需要改，只要词表变大即可)
    model = DeepRouteSet(
        vocab_size=len(tok.vocab),
        embed_dim=int(cfg['model']['embed_dim']),
        hidden_dim=int(cfg['model']['hidden_dim']),
        num_layers=int(cfg['model']['num_layers']),
        pad_id=tok.pad_id
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg['training']['lr']))
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tok.pad_id)

    logger.info(f'训练启动! 序列数={len(ds)} 词表大小={len(tok.vocab)} 设备={device}')

    epochs = int(cfg['training']['epochs'])
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x_in, x_tg in loader:
            x_in, x_tg = x_in.to(device), x_tg.to(device)
            logits = model(x_in)  # [Batch, Length, VocabSize]
            
            # 计算 Loss
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), x_tg.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            
        logger.info(f'Epoch [{epoch}/{epochs}] | Loss: {total_loss / max(len(loader), 1):.4f}')

    # 保存新的 Action 模型
    ckpt = {'state_dict': model.state_dict(), 'vocab_size': len(tok.vocab), 'pad_id': tok.pad_id}
    torch.save(ckpt, run.root / 'action_model.pt')
    run.metrics.write_text(json.dumps({'done': True}, ensure_ascii=False, indent=2), encoding='utf-8')
    logger.info(f"🎉 训练完成! 模型已保存至: {run.root / 'action_model.pt'}")


if __name__ == '__main__':
    main()