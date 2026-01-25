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
    ap.add_argument("--raw", default=None)      # optional override
    ap.add_argument("--outdir", default=None)   # optional override
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rows = int(cfg["board"]["rows"])
    cols = int(cfg["board"]["cols"])

    raw_path = Path(args.raw or cfg["data"]["raw_path"])
    outdir = Path(args.outdir or cfg["data"]["processed_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    routes = []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "holds" not in rec or not rec["holds"]:
            continue
        routes.append(rec)

    seed = int(cfg["project"]["seed"])
    random.seed(seed)
    random.shuffle(routes)

    n = len(routes)
    tr = float(cfg["data"]["train_ratio"])
    vr = float(cfg["data"]["val_ratio"])
    n_train = int(n * tr)
    n_val = int(n * vr)

    train_recs = routes[:n_train]
    val_recs = routes[n_train:n_train + n_val]
    test_recs = routes[n_train + n_val :]

    vocab = Counter()
    for rec in train_recs:
        vocab.update(route_to_seq(rec, cols))

    PAD, BOS, EOS, UNK = "<PAD>", "<BOS>", "<EOS>", "<UNK>"

    # ---- infer grade range from data (assumes int grades in toy/rawschema stage) ----
    all_grades = []
    for rec in routes:
        g = rec.get("grade", 0)
        if isinstance(g, int):
            all_grades.append(g)
        else:
            # fallback for non-int grades; keep minimal (toy stage should be int)
            try:
                all_grades.append(int(g))
            except Exception:
                all_grades.append(0)

    g_min = min(all_grades) if all_grades else 0
    g_max = max(all_grades) if all_grades else 0

    # We want tokens <G0>..<Gmax> (DeepRouteSet expects this exact format)
    grade_tokens = [f"<G{i}>" for i in range(0, g_max + 1)]

    # ---- final vocab order: specials -> grade tokens -> hold/role tokens ----
    hold_tokens = [t for t, _ in sorted(vocab.items(), key=lambda x: (-x[1], x[0]))]
    tokens = [PAD, BOS, EOS, UNK] + grade_tokens + hold_tokens
    tok2id = {t: i for i, t in enumerate(tokens)}



    def dump_split(recs, path: Path):
        with path.open("w", encoding="utf-8") as f:
            for rec in recs:
                seq = route_to_seq(rec, cols)
                f.write(json.dumps({"id": rec.get("id"), "seq": seq, "grade": rec.get("grade")}, ensure_ascii=False) + "\n")

    dump_split(train_recs, outdir / "train.jsonl")
    dump_split(val_recs, outdir / "val.jsonl")
    dump_split(test_recs, outdir / "test.jsonl")

    report = {
        "raw_path": str(raw_path),
        "n_total": n,
        "n_train": len(train_recs),
        "n_val": len(val_recs),
        "n_test": len(test_recs),
        "rows": rows,
        "cols": cols,
        "vocab_size": len(tokens),
        "seed": seed,
    }
    # ---- FORCE add grade tokens required by DeepRouteSet: <G0>..<Gmax> ----
    all_grades = []
    for rec in routes:
        try:
            all_grades.append(int(rec.get("grade", 0)))
        except Exception:
            all_grades.append(0)
    g_max = max(all_grades) if all_grades else 0

    # Ensure grade tokens exist in tok2id (insert after specials if possible)
    grade_tokens = [f"<G{i}>" for i in range(0, g_max + 1)]

    # Rebuild a stable ordered dict: specials -> grade tokens -> rest
    specials = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
    ordered_tokens = []
    for s in specials:
        if s in tok2id:
            ordered_tokens.append(s)
    for gt in grade_tokens:
        ordered_tokens.append(gt)
    # append remaining tokens in existing id order
    for t, _id in sorted(tok2id.items(), key=lambda kv: kv[1]):
        if t in specials:
            continue
        if t in grade_tokens:
            continue
        ordered_tokens.append(t)

    tok2id = {t: i for i, t in enumerate(ordered_tokens)}

    (outdir / "tokenizer_vocab.json").write_text(
        json.dumps(tok2id, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


    print(f"[preprocess] total={n} train={len(train_recs)} val={len(val_recs)} test={len(test_recs)} vocab={len(tokens)} -> {outdir}")


if __name__ == "__main__":
    main()
