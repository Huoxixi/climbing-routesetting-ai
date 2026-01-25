from __future__ import annotations

import argparse
import json
from pathlib import Path


def _infer_id_base(holds: list[int], rows: int, cols: int) -> int:
    """
    Infer whether hold ids are 0-based [0..N-1] or 1-based [1..N].
    Returns base (0 or 1).
    """
    n = rows * cols
    if not holds:
        return 0
    mn = min(holds)
    mx = max(holds)

    if mn == 0:
        return 0
    if mx == n:
        return 1
    if 1 <= mn and mx <= n:
        return 1
    return 0


def _id_to_rc(hid: int, base: int, rows: int, cols: int) -> tuple[int, int]:
    idx = hid - base
    r = idx // cols
    c = idx % cols
    return int(r), int(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/raw/toy_routes.jsonl")
    ap.add_argument("--out", default="data/raw/moonboard_routes.jsonl")
    ap.add_argument("--board", default="moonboard")
    ap.add_argument("--rows", type=int, default=18)
    ap.add_argument("--cols", type=int, default=11)
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        print(f"[convert] Input file not found: {inp}")
        return

    lines = inp.read_text(encoding="utf-8").splitlines()

    n_in = 0
    n_out = 0

    out.write_text("", encoding="utf-8")

    for i, line in enumerate(lines):
        if not line.strip():
            continue
        n_in += 1
        r = json.loads(line)

        holds_ids = r.get("holds", [])
        start_id = r.get("start", None)
        end_id = r.get("end", None)

        # ✅ 关键修复：合并 start, end 和 holds，确保它们都在处理列表中
        all_ids = set()
        if holds_ids:
            all_ids.update([int(x) for x in holds_ids])
        
        # 确保 start 和 end 被加入集合
        if start_id is not None:
            all_ids.add(int(start_id))
        if end_id is not None:
            all_ids.add(int(end_id))

        if not all_ids:
            continue

        base = _infer_id_base(list(all_ids), args.rows, args.cols)

        holds_out = []
        # 转为列表排序，保持稳定性
        for hid in sorted(list(all_ids)):
            rr, cc = _id_to_rc(hid, base, args.rows, args.cols)

            if rr < 0 or rr >= args.rows or cc < 0 or cc >= args.cols:
                continue

            role = "M"
            # ✅ 这里的逻辑现在能生效了，因为 start_id 肯定在循环里
            if start_id is not None and hid == int(start_id):
                role = "S"
            elif end_id is not None and hid == int(end_id):
                role = "F"
            
            holds_out.append({"r": rr, "c": cc, "role": role})

        if not holds_out:
            continue

        rec = {
            "id": str(r.get("id", f"toy_{i}")),
            "board": args.board,
            "grade": r.get("grade", 0),
            "holds": holds_out,
            "source": "toy_routes.jsonl",
        }

        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_out += 1

    print(f"[convert] in={n_in} out={n_out} -> {out}")


if __name__ == "__main__":
    main()