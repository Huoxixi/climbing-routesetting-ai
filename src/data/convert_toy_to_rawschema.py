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
    mn = min(holds)
    mx = max(holds)

    # If any 0 appears, it's 0-based.
    if mn == 0:
        return 0
    # If max exceeds n-1 but <= n, likely 1-based.
    if mx == n:
        return 1
    # If all within 1..n and none is 0, prefer 1-based.
    if 1 <= mn and mx <= n:
        return 1
    # Fallback: assume 0-based.
    return 0


def _id_to_rc(hid: int, base: int, rows: int, cols: int) -> tuple[int, int]:
    """
    Map hold id to (r,c) with 0-based r,c.
    Assumes row-major order: idx = r*cols + c (+ base).
    """
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

    lines = inp.read_text(encoding="utf-8").splitlines()

    n_in = 0
    n_out = 0

    # Overwrite output file at start
    out.write_text("", encoding="utf-8")

    for i, line in enumerate(lines):
        if not line.strip():
            continue
        n_in += 1
        r = json.loads(line)

        holds_ids = r.get("holds", None)
        if not isinstance(holds_ids, list) or not holds_ids:
            continue

        # ensure int list
        try:
            holds_ids_int = [int(x) for x in holds_ids]
        except Exception:
            continue

        base = _infer_id_base(holds_ids_int, args.rows, args.cols)

        start_id = r.get("start", None)
        end_id = r.get("end", None)
        start_id = int(start_id) if start_id is not None else None
        end_id = int(end_id) if end_id is not None else None

        holds = []
        for hid in holds_ids_int:
            rr, cc = _id_to_rc(hid, base, args.rows, args.cols)

            # bound check: skip invalid
            if rr < 0 or rr >= args.rows or cc < 0 or cc >= args.cols:
                continue

            role = "M"
            if start_id is not None and hid == start_id:
                role = "S"
            if end_id is not None and hid == end_id:
                role = "F"
            holds.append({"r": rr, "c": cc, "role": role})

        if not holds:
            continue

        rec = {
            "id": str(r.get("id", f"toy_{i}")),
            "board": args.board,
            "grade": r.get("grade", 0),
            "holds": holds,
            "source": "toy_routes.jsonl",
        }

        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_out += 1

    print(f"[convert] in={n_in} out={n_out} -> {out}")


if __name__ == "__main__":
    main()
