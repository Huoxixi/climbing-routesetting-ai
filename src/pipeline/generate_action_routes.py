from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
import yaml

from src.common.seed import set_seed
from src.common.logging import get_logger
from src.common.paths import make_run_dir, write_meta
from src.data.tokenizer import load_tokenizer
from src.env.board import Board
from src.models.deeprouteset import DeepRouteSet

def get_latest_action_ckpt(out_dir: str) -> Path:
    base = Path(out_dir)
    runs = sorted(base.glob("*_action_model_*"))
    if not runs:
        raise FileNotFoundError("没有找到 action_model 的运行记录！")
    ckpt = runs[-1] / "action_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"找不到权重文件: {ckpt}")
    return ckpt

def decode_actions_to_holds(action_tokens: list[str], board: Board) -> list[int] | None:
    """
    生物力学解码：把左右手分离的动作，翻译回物理岩板上的绝对坐标。
    """
    holds = []
    lh_r, lh_c = -1, -1
    rh_r, rh_c = -1, -1

    for tok in action_tokens:
        if tok.startswith("START_H"):
            hid_str = tok.split("H")[-1]
            if not hid_str.isdigit(): return None
            hid = int(hid_str)
            r, c = board.from_id(hid)
            
            # 初始化左右手位置
            if lh_r == -1 and rh_r == -1:
                lh_r, lh_c = r, c
                rh_r, rh_c = r, c
            elif r != lh_r or c != lh_c:
                if c > lh_c:
                    rh_r, rh_c = r, c
                else:
                    rh_r, rh_c = lh_r, lh_c
                    lh_r, lh_c = r, c
            holds.append(hid)
            
        elif "_R" in tok and "_C" in tok:
            try:
                parts = tok.split("_")
                hand = parts[0]  # "LH" 或 "RH"
                r_part = [p for p in parts if p.startswith('R') and (p[1]=='+' or p[1]=='-')][0]
                c_part = [p for p in parts if p.startswith('C') and (p[1]=='+' or p[1]=='-')][0]
                dr = int(r_part[1:])
                dc = int(c_part[1:])
            except Exception:
                return None
            
            # 分别对左手和右手应用位移计算
            if hand == "LH":
                if lh_r == -1: return None
                lh_r += dr
                lh_c += dc
                if 0 <= lh_r < board.rows and 0 <= lh_c < board.cols:
                    holds.append(board.to_id(lh_r, lh_c))
                else: return None
            elif hand == "RH":
                if rh_r == -1: return None
                rh_r += dr
                rh_c += dc
                if 0 <= rh_r < board.rows and 0 <= rh_c < board.cols:
                    holds.append(board.to_id(rh_r, rh_c))
                else: return None
            else:
                return None
    return holds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase2.yaml")
    ap.add_argument("--grades", default="3,4,5,6")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(cfg["project"]["seed"]))

    out_root = cfg["project"].get("out_dir", "outputs/phase2")
    run = make_run_dir("generate_action", root=out_root)
    write_meta(run)

    logger = get_logger("generate_action", str(run.root / "stdout.log"))
    device = torch.device(cfg["project"]["device"])
    board = Board(rows=int(cfg["board"]["rows"]), cols=int(cfg["board"]["cols"]))

    proc = Path("data/processed_actions")
    tok = load_tokenizer(str(proc / "action_tokenizer_vocab.json"))
    
    ckpt_path = get_latest_action_ckpt(out_root)
    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model = DeepRouteSet(
        vocab_size=ckpt["vocab_size"],
        embed_dim=int(cfg["model"]["embed_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_layers=int(cfg["model"]["num_layers"]),
        pad_id=ckpt["pad_id"]
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    grades = [int(x.strip()) for x in args.grades.split(",") if x.strip() != ""]
    samples_per_grade = int(cfg["generation"]["samples_per_grade"])
    max_len = int(cfg["data"]["max_seq_len"])
    temperature = float(cfg["generation"]["temperature"])
    top_k = int(cfg["generation"]["top_k"])

    out_file = run.artifacts / "action_generated_routes.jsonl"
    n_total = 0
    n_ok = 0

    with out_file.open("w", encoding="utf-8") as f_out:
        for g in grades:
            prefix = [tok.bos_id, tok.vocab.get(f"<G{g}>", tok.unk_id)]
            for _ in range(samples_per_grade):
                ids = model.generate(
                    bos=tok.bos_id, eos=tok.eos_id,
                    prefix=prefix, max_len=max_len,
                    temperature=temperature, top_k=top_k
                )
                action_tokens = tok.decode(ids)
                n_total += 1

                holds = decode_actions_to_holds(action_tokens, board)
                if holds is not None and len(holds) >= 3:
                    rec = {
                        "grade": g,
                        "action_tokens": action_tokens,
                        "seq_betamove": holds,
                        "board": "moonboard"
                    }
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_ok += 1

    pass_rate = (n_ok / n_total) if n_total > 0 else 0
    logger.info(f"生物力学生成完毕! 尝试生成 {n_total} 条, 物理合法 {n_ok} 条 (合法率 {pass_rate:.1%})")

if __name__ == "__main__":
    main()