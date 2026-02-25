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
    """自动寻找最新训练出来的 action_model.pt"""
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
    核心逻辑：把动作“翻译”回物理岩点
    例如: ['START_H37', 'MOVE_R+3_C-1'] -> [37, 70]
    """
    holds = []
    curr_r, curr_c = -1, -1

    for tok in action_tokens:
        if tok.startswith("START_H"):
            hid_str = tok.split("H")[-1]
            if not hid_str.isdigit():
                return None
            hid = int(hid_str)
            curr_r, curr_c = board.from_id(hid)
            holds.append(hid)
            
        elif "_R" in tok and "_C" in tok:
            # 解析相对位移，例如 MOVE_R+3_C-1
            try:
                parts = tok.split("_")
                dr_str = parts[1][1:]  # 去掉 'R'，留下 '+3'
                dc_str = parts[2][1:]  # 去掉 'C'，留下 '-1'
                
                dr = int(dr_str)
                dc = int(dc_str)
            except Exception:
                return None  # 格式不对

            # 只有当有了起步点之后才能应用相对位移
            if curr_r == -1:
                return None
                
            curr_r += dr
            curr_c += dc
            
            # 物理越界检查 (墙外面的点不算)
            if 0 <= curr_r < board.rows and 0 <= curr_c < board.cols:
                holds.append(board.to_id(curr_r, curr_c))
            else:
                # AI 想象力太丰富，飞出岩板了
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

    # 1. 加载最新的词表和模型
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

    # 2. 准备输出
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
                # AI 生成动作 Token IDs
                ids = model.generate(
                    bos=tok.bos_id,
                    eos=tok.eos_id,
                    prefix=prefix,
                    max_len=max_len,
                    temperature=temperature,
                    top_k=top_k
                )
                
                # 解码为动作词汇
                action_tokens = tok.decode(ids)
                n_total += 1

                # 将动作翻译为物理坐标点
                holds = decode_actions_to_holds(action_tokens, board)
                
                # 过滤：必须合法在板子上，且至少 3 个点
                if holds is not None and len(holds) >= 3:
                    rec = {
                        "grade": g,
                        "action_tokens": action_tokens,
                        "seq_betamove": holds, # 为了兼容现有的画图脚本
                        "board": "moonboard"
                    }
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_ok += 1

    pass_rate = (n_ok / n_total) if n_total > 0 else 0
    logger.info(f"生成完毕! 尝试生成 {n_total} 条, 物理合法 {n_ok} 条 (合法率 {pass_rate:.1%})")
    logger.info(f"已保存至: {out_file}")

if __name__ == "__main__":
    main()