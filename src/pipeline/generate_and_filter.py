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

# ✅ 引入模型
from src.models.deeprouteset import DeepRouteSet
# ✅ 引入 BetaMove 相关组件 (修复逻辑断层)
from src.betamove.betamove import run_betamove
from src.betamove.constraints import Constraints
from src.env.board import Board


def parse_tokens_to_holds(tokens: List[str]) -> Tuple[Optional[int], Optional[int], List[int]]:
    """
    将 token 序列 (e.g. ['S_H4', 'M_H10', 'F_H140']) 解析为 BetaMove 需要的 ID。
    返回: (start_id, end_id, all_hold_ids)
    """
    start = None
    end = None
    holds = []
    
    for t in tokens:
        # 简单正则提取: S_H123 -> 123
        # 格式应该是 {ROLE}_H{ID}
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
            start = hid
        elif role_part == "F":
            end = hid

    return start, end, holds


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
    ckpt = torch.load(args.ckpt, map_location="cpu") # 先加载到 cpu 避免显存碎片
    
    # 确保从 checkpoint 或 config 中获取正确的参数
    # 如果 checkpoint 中保存了 args 最好，否则从 cfg 读取
    model = DeepRouteSet(
        vocab_size=len(tok.vocab),
        embed_dim=int(cfg["model"]["embed_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_layers=int(cfg["model"]["num_layers"]),
        pad_id=tok.pad_id,
    )
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    model.to(device).eval()

    # 4. 准备 BetaMove 环境 (Board & Constraints)
    # ✅ 必须初始化 Board 才能计算距离
    board = Board(
        rows=int(cfg["board"]["rows"]), 
        cols=int(cfg["board"]["cols"])
    )
    # ✅ 必须初始化约束条件
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

    # ✅ 修复: 修正变量名错误 (as f_ok)
    with out_all.open("w", encoding="utf-8") as f_all, out_ok.open("w", encoding="utf-8") as f_ok:
        for g in grades:
            # 构造 Prefix: <BOS> + <Gk>
            prefix = [tok.bos_id, tok.vocab[f"<G{g}>"]]
            
            for i in range(samples_per_grade):
                # ✅ 修复: 直接调用 Model 内部的 generate 方法
                ids = model.generate(
                    bos=tok.bos_id,
                    eos=tok.eos_id,
                    prefix=prefix,
                    max_len=max_len,
                    temperature=temperature,
                    top_k=top_k
                )
                
                toks = tok.decode(ids) # 注意: tokenizer.decode 返回的是 int list 还是 token str list? 
                # 里的 decode 返回的是 List[int] (hold ids) 还是 List[str]?
                # 检查你的 tokenizer.py: decode 返回 List[int] (hold ids)，但是 logic 是过滤掉 special token。
                # ‼️ 重要修正: 
                # 你的 Tokenizer.decode 实现会将 token ID 转回原始 Hold ID (int)。
                # 但是 DeepRouteSet 也是作为 Token ID 输出的。
                # 在 Phase 2 的 preprocess_rawschema.py 中，我们把 hold 变成了 "M_H123" 这种 STRING token。
                # 所以 tok.ivocab[id] 拿到的是 "M_H123"。
                
                # 手动 decode 以获取 raw tokens (因为我们要解析 S_ / F_ 结构)
                raw_tokens = [tok.ivocab.get(tid, "") for tid in ids]
                
                # 过滤掉 <BOS>, <EOS>, <Gk>, <PAD>
                route_tokens = [
                    t for t in raw_tokens 
                    if t not in ["<BOS>", "<EOS>", "<PAD>"] and not t.startswith("<G")
                ]

                rec = {"grade": g, "tokens": route_tokens}
                f_all.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_total += 1

                # --- BetaMove Filter ---
                # 1. 解析 token 为 ID
                start_id, end_id, hold_ids = parse_tokens_to_holds(route_tokens)
                
                is_valid = False
                reason = "parse_fail"

                if start_id is not None and end_id is not None and len(hold_ids) >= 2:
                    # 2. 运行 BetaMove
                    # ✅ 修复: 使用正确的 run_betamove
                    bm_res = run_betamove(board, hold_ids, start_id, end_id, cons)
                    if bm_res.success:
                        is_valid = True
                        rec["seq_betamove"] = bm_res.seq # 保存物理可行序列
                    else:
                        reason = bm_res.reason
                else:
                    reason = "missing_start_or_end"

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