from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
import yaml

from src.common.seed import set_seed
from src.common.logging import get_logger
from src.common.paths import make_run_dir, write_meta
from src.data.tokenizer import load_tokenizer

from src.models.deeprouteset import DeepRouteSet
from src.betamove.betamove import run_betamove
from src.betamove.constraints import Constraints
from src.env.board import Board


def parse_tokens_to_holds(tokens: List[str]) -> Tuple[Optional[int], Optional[int], List[int]]:
    """
    解析 Token 序列。
    策略调整：如果模型没有显式生成 S_ 或 F_ 标签，
    则默认使用序列的第一个点作为 Start，最后一个点作为 End。
    """
    explicit_start = None
    explicit_end = None
    holds = []
    
    for t in tokens:
        # 提取 ID：无论是 S_H12, M_H12, 还是 F_H12，都提取出 12
        if "_H" not in t:
            continue
        
        parts = t.split("_H")
        if len(parts) != 2:
            continue
        
        role_part = parts[0]  # S, M, F
        id_part = parts[1]    # 123
        
        if not id_part.isdigit():
            continue
            
        hid = int(id_part)
        holds.append(hid)
        
        if role_part == "S":
            explicit_start = hid
        elif role_part == "F":
            explicit_end = hid

    # ---- 鲁棒性修复 (Robustness Fix) ----
    # 如果没有找到点，直接返回空
    if not holds:
        return None, None, []

    # 如果模型忘了标记 Start，就用第一个点
    final_start = explicit_start if explicit_start is not None else holds[0]
    
    # 如果模型忘了标记 End，就用最后一个点
    final_end = explicit_end if explicit_end is not None else holds[-1]

    # 如果只有一个点，Start 和 End 都是它，这在物理上可能导致距离为0，但在逻辑上是通的
    
    return final_start, final_end, holds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="path to deeprouteset.pt")
    ap.add_argument("--out_root", default=None, help="override project.out_dir")
    ap.add_argument("--grades", default="0,1,2,3,4,5,6,7,8,9")
    args = ap.parse_args()

    # 1. 加载配置
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(cfg["project"]["seed"]))

    out_root = args.out_root or cfg["project"].get("out_dir", None)
    run = make_run_dir("generate", root=out_root)
    write_meta(run)

    logger = get_logger("generate_routes", str(run.root / "stdout.log"))
    device = torch.device(cfg["project"]["device"])

    # 2. 准备数据处理器
    proc = Path(cfg["data"]["processed_dir"])
    tok = load_tokenizer(str(proc / "tokenizer_vocab.json"))

    # 3. 加载模型
    logger.info(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    
    model = DeepRouteSet(
        vocab_size=len(tok.vocab),
        embed_dim=int(cfg["model"]["embed_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_layers=int(cfg["model"]["num_layers"]),
        pad_id=tok.pad_id,
    )
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    model.to(device).eval()

    # 4. 准备 BetaMove 环境
    board = Board(
        rows=int(cfg["board"]["rows"]), 
        cols=int(cfg["board"]["cols"])
    )
    cons = Constraints(
        max_reach=float(cfg["betamove"]["max_reach"]),
        require_monotonic_up=bool(cfg["betamove"]["require_monotonic_up"])
    )
    
    # 5. 生成参数
    grades = [int(x.strip()) for x in args.grades.split(",") if x.strip() != ""]
    samples_per_grade = int(cfg["generation"]["samples_per_grade"])
    max_len = int(cfg["data"]["max_seq_len"])
    temperature = float(cfg["generation"]["temperature"])
    top_k = int(cfg["generation"]["top_k"])

    artifacts_dir = run.root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    out_all = artifacts_dir / "generated_routes.jsonl"
    out_ok = artifacts_dir / "generated_routes_filtered.jsonl"

    n_total = 0
    n_ok = 0

    logger.info(f"Start generation: grades={grades}, count/grade={samples_per_grade}")

    with out_all.open("w", encoding="utf-8") as f_all, out_ok.open("w", encoding="utf-8") as f_ok:
        for g in grades:
            # 构造 Prefix
            prefix = [tok.bos_id, tok.vocab[f"<G{g}>"]]
            
            for i in range(samples_per_grade):
                ids = model.generate(
                    bos=tok.bos_id,
                    eos=tok.eos_id,
                    prefix=prefix,
                    max_len=max_len,
                    temperature=temperature,
                    top_k=top_k
                )
                
                # 解码
                raw_tokens = [tok.ivocab.get(tid, "") for tid in ids]
                route_tokens = [
                    t for t in raw_tokens 
                    if t not in ["<BOS>", "<EOS>", "<PAD>"] and not t.startswith("<G")
                ]

                # 记录原始生成
                rec = {"grade": g, "tokens": route_tokens}
                f_all.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_total += 1

                # --- BetaMove Filter (With Robust Parsing) ---
                start_id, end_id, hold_ids = parse_tokens_to_holds(route_tokens)
                
                is_valid = False
                reason = "unknown"

                if start_id is not None and end_id is not None and len(hold_ids) >= 1:
                    bm_res = run_betamove(board, hold_ids, start_id, end_id, cons)
                    if bm_res.success:
                        is_valid = True
                        rec["seq_betamove"] = bm_res.seq
                    else:
                        reason = bm_res.reason
                else:
                    reason = "empty_route"

                if is_valid:
                    f_ok.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_ok += 1
    
    pass_rate = (n_ok / n_total) if n_total > 0 else 0.0
    metrics = {
        "n_total": n_total,
        "n_ok": n_ok,
        "pass_rate": pass_rate,
        "samples_per_grade": samples_per_grade,
        "grades": grades,
    }
    (run.root / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"Done. Generated: {n_total}, Climabable: {n_ok} ({pass_rate:.1%})")
    logger.info(f"Artifacts in: {run.root}")

if __name__ == "__main__":
    main()