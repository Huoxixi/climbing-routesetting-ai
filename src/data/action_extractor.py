from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from src.env.board import Board


def determine_action(dr: int, dc: int) -> str:
    """
    根据物理位移向量 (dr, dc) 推断动作类型 (参考 Wei 2014)
    """
    dist = math.hypot(dr, dc)
    
    # 启发式规则 (Heuristic Rules)
    if dist >= 4.5:
        # 距离超过 4.5 格，通常需要动态发力或大跳
        return "DYNO"
    elif dc == 0 and dr >= 3:
        # 横向不位移，纵向大幅拉伸，需要强锁臂能力
        return "LOCK"
    elif abs(dc) >= 3 and dr <= 1:
        # 横向大幅度移动，通常是交叉手或大侧拉
        return "CROSS"
    else:
        # 默认普通移动
        return "MOVE"


def parse_hid(token) -> int:
    """
    鲁棒地从 Token 中提取数字 ID。
    支持: 45, "45", "H45", "S_H45", "M_H45"
    """
    if isinstance(token, int):
        return token
    
    token = str(token)
    if "_H" in token:
        return int(token.split("_H")[-1])
    elif token.startswith("H"):
        return int(token[1:])
    else:
        return int(token)


def extract_action_sequence(seq: list, board: Board) -> list[str]:
    """
    将包含绝对 ID 的序列转化为相对动作序列。
    例如: ["S_H10", "M_H45"] -> ["START_H10", "MOVE_R+3_C+2"]
    """
    if not seq:
        return []

    # 第一步：把所有 Token 转化成纯数字 ID
    parsed_ids = []
    for t in seq:
        try:
            parsed_ids.append(parse_hid(t))
        except ValueError:
            continue
            
    if not parsed_ids:
        return []

    # 第二步：必须有一个绝对位置作为起步的“锚点” (Anchor)
    action_seq = [f"START_H{parsed_ids[0]}"]

    # 第三步：将后续的移动转为“动作向量”
    for i in range(1, len(parsed_ids)):
        r1, c1 = board.from_id(parsed_ids[i - 1])
        r2, c2 = board.from_id(parsed_ids[i])

        dr = r2 - r1
        dc = c2 - c1

        action_type = determine_action(dr, dc)
        
        # 格式化输出，例如 MOVE_R+3_C-2
        action_token = f"{action_type}_R{dr:+d}_C{dc:+d}"
        action_seq.append(action_token)

    return action_seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp_dir", default="data/processed")
    ap.add_argument("--out_dir", default="data/processed_actions")
    args = ap.parse_args()

    inp_dir = Path(args.inp_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    board = Board()  # 默认 18x11

    # 遍历 train, val, test 数据集
    splits = ["train", "val", "test"]
    
    for split in splits:
        inp_file = inp_dir / f"{split}.jsonl"
        out_file = out_dir / f"{split}_actions.jsonl"
        
        if not inp_file.exists():
            continue

        lines = inp_file.read_text(encoding="utf-8").splitlines()
        processed_count = 0

        with out_file.open("w", encoding="utf-8") as f_out:
            for line in lines:
                if not line.strip():
                    continue
                rec = json.loads(line)
                
                # 获取排序后的原始序列 (可能是 S_H10 这种格式)
                raw_seq = rec.get("seq", [])
                
                # 转化为动作序列
                action_seq = extract_action_sequence(raw_seq, board)
                
                # 保存新格式
                new_rec = {
                    "id": rec.get("id", f"unknown_{processed_count}"),
                    "grade": rec.get("grade", 0),
                    "raw_seq": raw_seq,
                    "action_seq": action_seq
                }
                f_out.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
                processed_count += 1
                
        print(f"[Action Extractor] {split:5s} : 转换了 {processed_count} 条路线 -> {out_file}")

        # 打印一条 Demo 看看长什么样
        if processed_count > 0:
            demo = json.loads(out_file.read_text(encoding="utf-8").splitlines()[0])
            print(f"  └─ Demo 原始: {demo['raw_seq']}")
            print(f"  └─ Demo 动作: {demo['action_seq']}")
            print("-" * 50)


if __name__ == "__main__":
    main()