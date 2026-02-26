from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from src.env.board import Board

class BiomechSimulator:
    def __init__(self, board: Board):
        self.board = board
        self.max_reach = 5.5  # 人类极限臂展 5.5格

    def generate_route(self, finish_r: int) -> list[int] | None:
        # 每条路线最多重试 20 次，如果走到死胡同就推翻重来
        for _ in range(20):
            # 1. 随机起步点 (最底部 0 或 1 行)
            r1 = random.randint(0, 1)
            c1 = random.randint(3, 7)
            start_holds = [(r1, c1)]
            lh = (r1, c1)
            rh = (r1, c1)
            
            # 50%概率双点起步
            if random.random() > 0.5:
                c2 = c1 + random.choice([-1, 1])
                start_holds.append((r1, c2))
                if c1 < c2:
                    lh, rh = (r1, c1), (r1, c2)
                else:
                    lh, rh = (r1, c2), (r1, c1)

            path = list(start_holds)
            stuck = False

            # 2. 左右手交替向上攀爬，直到摸到顶点 17 行
            while max(lh[0], rh[0]) < finish_r:
                # 优先动位置较低的那只手
                if lh[0] < rh[0]: moving_hand = 'LH'
                elif rh[0] < lh[0]: moving_hand = 'RH'
                else: moving_hand = random.choice(['LH', 'RH'])

                # 20%概率打破死板交替，模拟连续出同一只手
                if random.random() < 0.2:
                    moving_hand = 'LH' if moving_hand == 'RH' else 'RH'

                if moving_hand == 'LH':
                    cur, static = lh, rh
                else:
                    cur, static = rh, lh

                candidates = []
                # 寻找向上的岩点 (向上1到3格)
                for nr in range(cur[0] + 1, cur[0] + 4):
                    if nr > finish_r: continue
                    
                    # 左右搜索范围：基于静止手左右4格
                    for nc in range(max(0, static[1] - 4), min(self.board.cols, static[1] + 5)):
                        if (nr, nc) == static: continue
                        
                        # 判断是否超过臂展
                        dist_to_static = math.hypot(nr - static[0], nc - static[1])
                        if dist_to_static <= self.max_reach:
                            candidates.append((nr, nc))

                # 如果上面没有点了，说明走到了死胡同，放弃这一轮
                if not candidates:
                    stuck = True
                    break

                # 3. 评估每个可行点位的“舒服程度”
                weights = []
                for (nr, nc) in candidates:
                    w = 1.0
                    # 惩罚别扭的交叉手
                    if moving_hand == 'LH' and nc > static[1]: w *= 0.1 
                    if moving_hand == 'RH' and nc < static[1]: w *= 0.1
                    # 偏好跨度稍大一点的点(2行)，让路线更干脆
                    if nr - cur[0] == 2: w *= 2.0 
                    weights.append(w)

                # 掷骰子，选择下一个岩点
                nxt = random.choices(candidates, weights=weights, k=1)[0]
                path.append(nxt)
                
                # 更新手的当前位置
                if moving_hand == 'LH': lh = nxt
                else: rh = nxt

            # 如果没有卡死，且攀爬步数合理(5到22步)，直接返回！
            if not stuck and 5 <= len(path) <= 22:
                return [self.board.to_id(r, c) for r, c in path]

        # 20次都死胡同则返回失败（概率极小）
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--num_train", type=int, default=1500)
    ap.add_argument("--num_val", type=int, default=150)
    ap.add_argument("--num_test", type=int, default=150)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    board = Board()
    simulator = BiomechSimulator(board)

    splits = {"train": args.num_train, "val": args.num_val, "test": args.num_test}
    print("🚀 开始使用马尔可夫物理引擎极速生成数据...")

    for split_name, target_num in splits.items():
        out_file = out_dir / f"{split_name}.jsonl"
        success_count = 0
        
        with out_file.open("w", encoding="utf-8") as f:
            while success_count < target_num:
                finish_r = board.rows - 1
                path_ids = simulator.generate_route(finish_r)
                
                if path_ids:
                    grade = random.randint(3, 6)
                    rec = {"id": f"sim_{split_name}_{success_count}", "grade": grade, "seq": path_ids}
                    f.write(json.dumps(rec) + "\n")
                    success_count += 1
                    
                    # 每成功生成 100 条就打印一次进度
                    if success_count % 100 == 0:
                        print(f"  [{split_name}] 已生成 {success_count}/{target_num} 条...")

        print(f"✅ {split_name} 生成完毕: {success_count} 条。")

if __name__ == "__main__":
    main()