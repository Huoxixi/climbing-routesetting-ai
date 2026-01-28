import json
import random
from pathlib import Path
from src.env.board import Board
from src.betamove.constraints import Constraints, dist

def generate_synthetic_data(n_routes=1000):
    # 1. 初始化环境
    board = Board()
    # 稍微放宽一点限制 (max_reach=5.0)，让生成器更容易跑通
    cons = Constraints(max_reach=5.0) 
    
    routes = []
    print(f"[Synthetic] Start generating {n_routes} plausible routes...")
    
    generated_count = 0
    attempts = 0
    
    while generated_count < n_routes:
        attempts += 1
        if attempts > n_routes * 10: # 防止死循环
            print(f"[Warn] Too many attempts. Stopped at {generated_count}.")
            break

        # --- A. 随机选起点 (底座: Row 0-3) ---
        start_c = random.randint(0, 10)
        start_r = random.randint(0, 3)
        current_id = board.to_id(start_r, start_c)
        
        route_holds = [current_id]
        
        # --- B. 随机向上游走 (Random Walk with Physics) ---
        # 模拟一个人向上爬，每次在可行范围内选一个点
        failed = False
        while True:
            cur_r, cur_c = board.from_id(current_id)
            
            # 如果够高了(>=16)，就视为完攀
            if cur_r >= 16:
                break
                
            # 寻找所有“合规”的下一步：
            # 1. 比当前点高 (r > cur_r)
            # 2. 距离在合理范围内 (1.0 < d <= 5.0)
            candidates = []
            # 搜索范围：向上找 1~5 排
            for r in range(cur_r + 1, min(cur_r + 6, 18)): 
                for c in range(0, 11):
                    nid = board.to_id(r, c)
                    d = dist(board, current_id, nid)
                    if 1.0 < d <= cons.max_reach: 
                        candidates.append(nid)
            
            if not candidates:
                failed = True # 走进死胡同了
                break 
                
            # 随机选一个作为下一步
            next_id = random.choice(candidates)
            route_holds.append(next_id)
            current_id = next_id
            
        if failed:
            continue

        # --- C. 质量筛选 ---
        # 只有长度合适 (4-15步) 的路线才保留，太短太长都不要
        if 4 <= len(route_holds) <= 15:
            # 简单粗暴的难度模拟：先随机分配 V3-V6
            # (真正的难度要等以后有了 GradeNet 才能准，现在先占位)
            grade = random.randint(3, 6)
            
            # 转换为标准详细格式 (role: S/M/F)
            formatted_holds = []
            for j, hid in enumerate(route_holds):
                r, c = board.from_id(hid)
                role = "M"
                if j == 0: role = "S"
                elif j == len(route_holds)-1: role = "F"
                formatted_holds.append({"r": r, "c": c, "role": role})
                
            routes.append({
                "id": f"syn_{generated_count}",
                "grade": grade,
                "holds": formatted_holds,
                "board": "moonboard"
            })
            generated_count += 1
            
    # 保存结果
    out_path = Path("data/raw/synthetic_1k.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with out_path.open("w", encoding="utf-8") as f:
        for r in routes:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"[Synthetic] Done! Saved {len(routes)} routes to {out_path}")

if __name__ == "__main__":
    generate_synthetic_data()