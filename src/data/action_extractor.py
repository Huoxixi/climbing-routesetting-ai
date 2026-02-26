from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from src.env.board import Board


class ClimberState:
    """生物力学状态机，记录攀岩者肢体位置"""
    def __init__(self, board: Board):
        self.board = board
        self.lh_r, self.lh_c = -1, -1  # 左手坐标 (Left Hand)
        self.rh_r, self.rh_c = -1, -1  # 右手坐标 (Right Hand)

    def set_start(self, holds: list[int]):
        """根据起步点初始化左右手"""
        if len(holds) == 1:
            # 单点起步 (Match Start)
            r, c = self.board.from_id(holds[0])
            self.lh_r, self.lh_c = r, c
            self.rh_r, self.rh_c = r, c
        else:
            # 双点起步：靠左的是左手，靠右的是右手
            r1, c1 = self.board.from_id(holds[0])
            r2, c2 = self.board.from_id(holds[1])
            if c1 <= c2:
                self.lh_r, self.lh_c = r1, c1
                self.rh_r, self.rh_c = r2, c2
            else:
                self.lh_r, self.lh_c = r2, c2
                self.rh_r, self.rh_c = r1, c1

    def infer_next_move(self, target_id: int) -> str:
        """
        核心生物力学推断：根据目标点位置，决定是动左手还是右手，以及做什么动作。
        """
        tr, tc = self.board.from_id(target_id)
        
        # 计算左右手到目标点的距离
        dist_l = math.hypot(tr - self.lh_r, tc - self.lh_c) if self.lh_r != -1 else 999
        dist_r = math.hypot(tr - self.rh_r, tc - self.rh_c) if self.rh_r != -1 else 999
        
        # 计算身体重心 (简化为双手中点)
        center_c = (self.lh_c + self.rh_c) / 2 if self.lh_r != -1 and self.rh_r != -1 else tc

        # 1. 决定用哪只手 (Heuristic Decision)
        moving_hand = "UNKNOWN"
        if tc < center_c - 1:
            # 目标在身体极度偏左 -> 通常用左手 (除非刻意交叉)
            moving_hand = "LH"
        elif tc > center_c + 1:
            # 目标在身体极度偏右 -> 通常用右手
            moving_hand = "RH"
        else:
            # 在中间，谁离得近谁上，或者交替出手
            moving_hand = "LH" if dist_l <= dist_r else "RH"

        # 2. 判定具体动作类型
        if moving_hand == "LH":
            dr, dc = tr - self.lh_r, tc - self.lh_c
            dist = dist_l
        else:
            dr, dc = tr - self.rh_r, tc - self.rh_c
            dist = dist_r

        action_type = "MOVE"
        if dist >= 4.5:
            action_type = "DYNO"
        elif moving_hand == "LH" and dc > 2:
            action_type = "CROSS" # 左手向右大跨度
        elif moving_hand == "RH" and dc < -2:
            action_type = "CROSS" # 右手向左大跨度
        elif dc == 0 and dr >= 3:
            action_type = "LOCK"
            
        # 3. 更新状态机
        if moving_hand == "LH":
            self.lh_r, self.lh_c = tr, tc
        else:
            self.rh_r, self.rh_c = tr, tc

        # 组合新 Token: 例 RH_CROSS_R+2_C-3
        return f"{moving_hand}_{action_type}_R{dr:+d}_C{dc:+d}"


def parse_hid(token) -> int:
    if isinstance(token, int): return token
    token = str(token)
    if "_H" in token: return int(token.split("_H")[-1])
    elif token.startswith("H"): return int(token[1:])
    else: return int(token)


def extract_action_sequence(seq: list, board: Board) -> list[str]:
    """带有左右手推断的新版提取逻辑"""
    parsed_ids = []
    for t in seq:
        try:
            parsed_ids.append(parse_hid(t))
        except ValueError:
            continue
            
    if not parsed_ids: return []

    # 初始化状态机
    state = ClimberState(board)
    
    # 假设前 1 个或 2 个点是起步点 (MoonBoard 通常底下起步点很密集)
    start_holds = [parsed_ids[0]]
    if len(parsed_ids) > 1 and board.from_id(parsed_ids[1])[0] <= 4:
        # 如果第二个点也很低，认为是双点起步
        start_holds.append(parsed_ids[1])
        
    state.set_start(start_holds)

    action_seq = []
    # 记录起步点 Token
    for hid in start_holds:
        action_seq.append(f"START_H{hid}")

    # 从起步点之后开始推断后续动作
    start_idx = len(start_holds)
    for i in range(start_idx, len(parsed_ids)):
        target_hid = parsed_ids[i]
        action_token = state.infer_next_move(target_hid)
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
    board = Board()

    for split in ["train", "val", "test"]:
        inp_file = inp_dir / f"{split}.jsonl"
        out_file = out_dir / f"{split}_actions.jsonl"
        
        if not inp_file.exists(): continue

        lines = inp_file.read_text(encoding="utf-8").splitlines()
        processed_count = 0

        with out_file.open("w", encoding="utf-8") as f_out:
            for line in lines:
                if not line.strip(): continue
                rec = json.loads(line)
                raw_seq = rec.get("seq", [])
                
                # 提取带左右手的动作序列！
                action_seq = extract_action_sequence(raw_seq, board)
                
                new_rec = {
                    "id": rec.get("id", f"unknown_{processed_count}"),
                    "grade": rec.get("grade", 0),
                    "raw_seq": raw_seq,
                    "action_seq": action_seq
                }
                f_out.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
                processed_count += 1
                
        print(f"[Biomechanical Extractor] {split:5s} : 转换了 {processed_count} 条路线 -> {out_file}")

        if processed_count > 0:
            demo = json.loads(out_file.read_text(encoding="utf-8").splitlines()[0])
            print(f"  └─ 原始: {demo['raw_seq']}")
            print(f"  └─ 生物力学动作: {demo['action_seq']}")
            print("-" * 60)

if __name__ == "__main__":
    main()