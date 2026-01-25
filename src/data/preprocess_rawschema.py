from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import yaml


def hold_token(r: int, c: int, cols: int) -> str:
    hid = r * cols + c
    return f"H{hid}"


def route_to_seq(rec: dict, cols: int) -> list[str]:
    holds = rec["holds"]
    seq: list[str] = []
    # 按照 BetaMove 要求的逻辑，或者是原始顺序？
    # 这里我们保持 raw schema 的顺序，或者按 id 排序
    # 为了保证一致性，建议按 id (r*cols+c) 排序，但 raw 数据可能已经是乱序
    # 这里暂时保持 list 顺序，假设 raw schema 已经合理
    # 但通常为了 Transformer 学习，最好是底->顶排序
    # 这里简化处理，直接转换
    for h in holds:
        tok = hold_token(int(h["r"]), int(h["c"]), cols)
        role = str(h.get("role", "M")).upper()
        if role not in {"S", "M", "F"}:
            role = "M"
        seq.append(f"{role}_{tok}")
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase2.yaml")
    ap.add_argument("--raw", default=None)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rows = int(cfg["board"]["rows"])
    cols = int(cfg["board"]["cols"])
    seed = int(cfg["project"]["seed"])

    raw_path = Path(args.raw or cfg["data"]["raw_path"])
    outdir = Path(args.outdir or cfg["data"]["processed_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    # 读取数据
    routes = []
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if "holds" not in rec or not rec["holds"]:
                continue
            routes.append(rec)
    else:
        print(f"[warn] Raw path not found: {raw_path}")

    random.seed(seed)
    random.shuffle(routes)

    n = len(routes)
    tr = float(cfg["data"]["train_ratio"])
    vr = float(cfg["data"]["val_ratio"])
    n_train = int(n * tr)
    n_val = int(n * vr)

    train_recs = routes[:n_train]
    val_recs = routes[n_train : n_train + n_val]
    test_recs = routes[n_train + n_val :]

    # -------------------------------------------------------
    # ✅ 关键修复：构建全集词表 (Full Vocabulary)
    # 不依赖数据统计，而是遍历所有可能的岩点位置
    # -------------------------------------------------------
    
    # 1. 特殊 Token
    specials = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
    
    # 2. 等级 Token (0..20, 预留足够多)
    # 虽然 toy 数据只到 9，但预留多一点没坏处
    grade_tokens = [f"<G{i}>" for i in range(21)]

    # 3. 岩点 Token (全排列: S/M/F * 所有坐标)
    hold_tokens = []
    n_holds = rows * cols
    for h in range(n_holds):
        # 每个岩点可能有 3 种状态
        hold_tokens.append(f"S_H{h}")
        hold_tokens.append(f"M_H{h}")
        hold_tokens.append(f"F_H{h}")

    # 合并 (顺序: 特殊 -> 等级 -> 岩点)
    full_vocab_list = specials + grade_tokens + hold_tokens
    
    # 建立映射字典
    tok2id = {t: i for i, t in enumerate(full_vocab_list)}

    # -------------------------------------------------------

    def dump_split(recs, path: Path):
        with path.open("w", encoding="utf-8") as f:
            for rec in recs:
                seq = route_to_seq(rec, cols)
                # 过滤掉不在词表里的 token (理论上现在不会有了)
                seq = [t if t in tok2id else "<UNK>" for t in seq]
                f.write(json.dumps({"id": rec.get("id"), "seq": seq, "grade": rec.get("grade")}, ensure_ascii=False) + "\n")

    dump_split(train_recs, outdir / "train.jsonl")
    dump_split(val_recs, outdir / "val.jsonl")
    dump_split(test_recs, outdir / "test.jsonl")

    # 保存词表
    (outdir / "tokenizer_vocab.json").write_text(
        json.dumps(tok2id, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[preprocess] total={n} train={len(train_recs)} val={len(val_recs)} vocab_size={len(tok2id)} (FULL BOARD) -> {outdir}")


if __name__ == "__main__":
    main()