import argparse
import json
import math
import random
from pathlib import Path
from src.env.board import Board

class BiomechSimulator:
    def __init__(self, board: Board):
        self.board = board
        self.max_reach = 6.0 

    def generate_route(self, finish_r: int) -> dict | None:
        for _ in range(20):
            r1 = random.randint(0, 1)
            c1 = random.randint(3, 7)
            hid1 = self.board.to_id(r1, c1)
            lh_r, lh_c, rh_r, rh_c = r1, c1, r1, c1
            action_seq, holds_seq = [], []
            
            if random.random() > 0.5:
                c2 = c1 + random.choice([-1, 1])
                hid2 = self.board.to_id(r1, c2)
                if c1 < c2:
                    lh_r, lh_c, rh_r, rh_c = r1, c1, r1, c2
                    action_seq.extend([f"START_LH_H{hid1}", f"START_RH_H{hid2}"])
                else:
                    lh_r, lh_c, rh_r, rh_c = r1, c2, r1, c1
                    action_seq.extend([f"START_LH_H{hid2}", f"START_RH_H{hid1}"])
                holds_seq.extend([hid1, hid2])
            else:
                action_seq.extend([f"START_LH_H{hid1}", f"START_RH_H{hid1}"])
                holds_seq.append(hid1)

            stuck = False
            while max(lh_r, rh_r) < finish_r:
                if lh_r < rh_r: moving_hand = 'LH'
                elif rh_r < lh_r: moving_hand = 'RH'
                else: moving_hand = random.choice(['LH', 'RH'])

                if random.random() < 0.2: moving_hand = 'LH' if moving_hand == 'RH' else 'RH'
                cur_r, cur_c = (lh_r, lh_c) if moving_hand == 'LH' else (rh_r, rh_c)
                static_r, static_c = (rh_r, rh_c) if moving_hand == 'LH' else (lh_r, lh_c)

                candidates = []
                for nr in range(cur_r + 1, cur_r + 4):
                    if nr > finish_r: continue
                    for nc in range(max(0, static_c - 4), min(self.board.cols, static_c + 5)):
                        if nr == static_r and nc == static_c: continue
                        if math.hypot(nr - static_r, nc - static_c) <= self.max_reach:
                            candidates.append((nr, nc))

                if not candidates:
                    stuck = True
                    break

                weights = [0.1 if (moving_hand == 'LH' and nc > static_c) or (moving_hand == 'RH' and nc < static_c) else (2.0 if nr - cur_r == 2 else 1.0) for nr, nc in candidates]
                nr, nc = random.choices(candidates, weights=weights, k=1)[0]
                target_hid = self.board.to_id(nr, nc)
                
                action_seq.append(f"{moving_hand}_H{target_hid}")
                holds_seq.append(target_hid)
                
                if moving_hand == 'LH': lh_r, lh_c = nr, nc
                else: rh_r, rh_c = nr, nc

            if not stuck and 5 <= len(action_seq) <= 24:
                return {"action_seq": action_seq, "seq_betamove": holds_seq}
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

    print("🚀 开始生成【绝对坐标 Token】数据...")
    for split_name, target_num in {"train": args.num_train, "val": args.num_val, "test": args.num_test}.items():
        success_count = 0
        with (out_dir / f"{split_name}_actions.jsonl").open("w", encoding="utf-8") as f:
            while success_count < target_num:
                route_data = simulator.generate_route(board.rows - 1)
                if route_data:
                    rec = {"id": f"sim_{split_name}_{success_count}", "grade": random.randint(3, 6), "action_seq": route_data["action_seq"], "seq_betamove": route_data["seq_betamove"]}
                    f.write(json.dumps(rec) + "\n")
                    success_count += 1
        print(f"✅ {split_name} 生成完毕: {success_count} 条。")

if __name__ == "__main__":
    main()