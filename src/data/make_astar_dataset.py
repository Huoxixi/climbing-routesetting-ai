import argparse
import json
import math
import random
from pathlib import Path
from src.env.board import Board

class BiomechSimulator:
    def __init__(self, board: Board):
        self.board = board

    def generate_route(self) -> dict | None:
        grade = random.randint(3, 6)
        if grade <= 4:
            finish_r, step_max_reach = random.randint(14, 15), random.uniform(3.0, 4.0)
        else:
            finish_r, step_max_reach = random.randint(16, 17), random.uniform(4.5, 6.0)

        for _ in range(50):
            # ==============================================================================
            # 🚀 极其精密的“三角起步”物理约束
            # ==============================================================================
            # 1. 随机定 B 点的横坐标（留出左右余量）
            base_c = random.randint(2, self.board.cols - 3) 
            base_hid = self.board.to_id(0, base_c)
            
            # 2. 控制 1、2 点的纵坐标差距在 0-1 之间
            lh_r = random.choice([2, 3])
            rh_r = random.choice([2, 3])  # 从 [2,3] 里随机抽，两者的差距天然 <= 1
            
            # 3. 控制 1、2 点的横坐标，满足：
            # (a) lh_c <= base_c <= rh_c (B夹在中间或某一手正下方)
            # (b) rh_c - lh_c < 3 (跨度差距为 1 或 2)
            configs = [
                (base_c, base_c + 1),       # 左手在B正上方，差距 1
                (base_c - 1, base_c),       # 右手在B正上方，差距 1
                (base_c - 1, base_c + 1),   # B在两者正中央，差距 2
                (base_c, base_c + 2),       # 左手在B正上方，差距 2
                (base_c - 2, base_c)        # 右手在B正上方，差距 2
            ]
            lh_c, rh_c = random.choice(configs)

            lh_init_hid = self.board.to_id(lh_r, lh_c)
            rh_init_hid = self.board.to_id(rh_r, rh_c)
            # ==============================================================================

            action_seq = [f"START_LH_H{lh_init_hid}", f"START_RH_H{rh_init_hid}"]
            holds_seq = [lh_init_hid, rh_init_hid]
            
            cur_lh_r, cur_lh_c, cur_rh_r, cur_rh_c = lh_r, lh_c, rh_r, rh_c
            stuck = False

            while max(cur_lh_r, cur_rh_r) < finish_r:
                if cur_lh_r < cur_rh_r: moving_hand = 'LH'
                elif cur_rh_r < cur_lh_r: moving_hand = 'RH'
                else: moving_hand = random.choice(['LH', 'RH'])

                cur_r, cur_c = (cur_lh_r, cur_lh_c) if moving_hand == 'LH' else (cur_rh_r, cur_rh_c)
                static_r, static_c = (cur_rh_r, cur_rh_c) if moving_hand == 'LH' else (cur_lh_r, cur_lh_c)

                candidates = []
                for nr in range(cur_r + 1, min(cur_r + 4, finish_r + 1)):
                    for nc in range(max(0, static_c - 5), min(self.board.cols, static_c + 6)):
                        dist_hands = math.hypot(nr - static_r, nc - static_c)
                        if dist_hands > step_max_reach: continue
                        
                        # 铁律：左手呆左边，右手呆右边，防麻花
                        if moving_hand == 'LH' and nc > static_c + 1.0: continue
                        if moving_hand == 'RH' and nc < static_c - 1.0: continue

                        dist_to_top = finish_r - nr
                        candidates.append((nr, nc, dist_to_top))

                if not candidates:
                    stuck = True; break

                weights = [10.0 / ((d_top + 1.0)**2) for nr, _, d_top in candidates]
                nr, nc, _ = random.choices(candidates, weights=weights, k=1)[0]
                
                target_hid = self.board.to_id(nr, nc)
                action_seq.append(f"{moving_hand}_H{target_hid}")
                holds_seq.append(target_hid)
                
                if moving_hand == 'LH': cur_lh_r, cur_lh_c = nr, nc
                else: cur_rh_r, cur_rh_c = nr, nc

            if not stuck and 5 <= len(action_seq) <= 24:
                return {
                    "action_seq": action_seq, 
                    "seq_betamove": holds_seq, 
                    "grade": grade,
                    "base_holds": [base_hid] 
                }
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

    print("🚀 开始生成【绝佳重心·三角起步法】数据...")
    for split_name, target_num in {"train": args.num_train, "val": args.num_val, "test": args.num_test}.items():
        success_count = 0
        with (out_dir / f"{split_name}_actions.jsonl").open("w", encoding="utf-8") as f:
            while success_count < target_num:
                route_data = simulator.generate_route()
                if route_data:
                    rec = {
                        "id": f"sim_{split_name}_{success_count}", 
                        "grade": route_data["grade"], 
                        "base_holds": route_data["base_holds"],     
                        "action_seq": route_data["action_seq"], 
                        "seq_betamove": route_data["seq_betamove"]
                    }
                    f.write(json.dumps(rec) + "\n")
                    success_count += 1
        print(f"✅ {split_name} 生成完毕: {success_count} 条。")

if __name__ == "__main__":
    main()