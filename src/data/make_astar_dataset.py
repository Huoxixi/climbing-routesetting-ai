import argparse
import json
import math
import random
from pathlib import Path
from src.env.board import Board

# 向量叉乘，判断点 C 是否在直线 AB 的逆时针方向
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

# 判断线段 AB 和 CD 是否相交
def intersect(A, B, C, D):
    # 如果两条线段有共同端点（比如同一只手的连续两步），不算交叉
    if A == C or A == D or B == C or B == D:
        return False
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

class BiomechSimulator:
    def __init__(self, board: Board):
        self.board = board

    def generate_route(self) -> dict | None:
        grade = random.randint(3, 6)
        if grade <= 4:
            finish_r, step_max_reach = random.randint(13, 15), random.uniform(3.0, 4.0)
        else:
            finish_r, step_max_reach = random.randint(16, 17), random.uniform(4.5, 6.0)

        for _ in range(20):
            start_r = random.choice([2, 3])
            lh_c = random.randint(3, 6)
            rh_c = lh_c + random.choice([1, 2])
            
            lh_hid = self.board.to_id(start_r, lh_c)
            rh_hid = self.board.to_id(start_r, rh_c)
            
            end_r, end_c = finish_r, random.randint(2, 8)
            
            action_seq = [f"START_LH_H{lh_hid}", f"START_RH_H{rh_hid}"]
            holds_seq = [lh_hid, rh_hid]
            lh_r, lh_c, rh_r, rh_c = start_r, lh_c, start_r, rh_c
            
            # 记录历史移动线段：[((r1, c1), (r2, c2)), ...]
            segments = []

            stuck = False
            while max(lh_r, rh_r) < finish_r:
                if lh_r < rh_r: moving_hand = 'LH'
                elif rh_r < lh_r: moving_hand = 'RH'
                else: moving_hand = random.choice(['LH', 'RH'])

                cur_r, cur_c = (lh_r, lh_c) if moving_hand == 'LH' else (rh_r, rh_c)
                static_r, static_c = (rh_r, rh_c) if moving_hand == 'LH' else (lh_r, lh_c)

                candidates = []
                for nr in range(cur_r + 1, min(cur_r + 4, end_r + 1)):
                    for nc in range(max(0, static_c - 4), min(self.board.cols, static_c + 5)):
                        if nr == static_r and nc == static_c: continue 
                        
                        dist_hands = math.hypot(nr - static_r, nc - static_c)
                        if dist_hands > step_max_reach: continue
                        
                        # 常规防交叉
                        if moving_hand == 'LH' and nc > static_c + 1: continue
                        if moving_hand == 'RH' and nc < static_c - 1: continue
                            
                        # 【核心防线：绝对线段防交叉】
                        new_seg = ((cur_r, cur_c), (nr, nc))
                        is_crossing = False
                        for old_seg in segments:
                            if intersect(new_seg[0], new_seg[1], old_seg[0], old_seg[1]):
                                is_crossing = True
                                break
                        if is_crossing: continue

                        dist_to_end = math.hypot(end_r - nr, end_c - nc)
                        candidates.append((nr, nc, dist_to_end))

                if not candidates:
                    stuck = True
                    break

                weights = [1.0 / ((d_end + 1.0) ** 2) for _, _, d_end in candidates]
                nr, nc, _ = random.choices(candidates, weights=weights, k=1)[0]
                
                target_hid = self.board.to_id(nr, nc)
                action_seq.append(f"{moving_hand}_H{target_hid}")
                holds_seq.append(target_hid)
                segments.append(((cur_r, cur_c), (nr, nc)))
                
                if moving_hand == 'LH': lh_r, lh_c = nr, nc
                else: rh_r, rh_c = nr, nc

            if not stuck and 5 <= len(action_seq) <= 24:
                return {"action_seq": action_seq, "seq_betamove": holds_seq, "grade": grade}
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/processed_actions")
    ap.add_argument("--num_train", type=int, default=1500)
    ap.add_argument("--num_val", type=int, default=150)
    ap.add_argument("--num_test", type=int, default=150)
    args = ap.parse_args()

    out_dir, board = Path(args.out_dir), Board()
    out_dir.mkdir(parents=True, exist_ok=True)
    simulator = BiomechSimulator(board)

    print("🚀 开始生成【绝对不交叉版】数据...")
    for split_name, target_num in {"train": args.num_train, "val": args.num_val, "test": args.num_test}.items():
        success_count = 0
        with (out_dir / f"{split_name}_actions.jsonl").open("w", encoding="utf-8") as f:
            while success_count < target_num:
                route_data = simulator.generate_route()
                if route_data:
                    rec = {"id": f"sim_{split_name}_{success_count}", "grade": route_data["grade"], "action_seq": route_data["action_seq"], "seq_betamove": route_data["seq_betamove"]}
                    f.write(json.dumps(rec) + "\n")
                    success_count += 1
        print(f"✅ {split_name} 生成完毕: {success_count} 条。")

if __name__ == "__main__":
    main()