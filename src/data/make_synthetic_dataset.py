import json
import random
import math
from pathlib import Path
from src.env.board import Board
from src.betamove.constraints import Constraints, dist

def get_dist(board, h1, h2):
    r1, c1 = board.from_id(h1)
    r2, c2 = board.from_id(h2)
    # 欧几里得距离
    return math.hypot(r1 - r2, c1 - c2)

def calculate_physics_grade(board, hold_ids):
    """
    根据物理特征计算“伪难度” (Heuristic Grading)
    """
    if len(hold_ids) < 2: return 0
    
    total_dist = 0
    max_move = 0
    horizontal_accum = 0
    
    for i in range(len(hold_ids) - 1):
        u, v = hold_ids[i], hold_ids[i+1]
        d = get_dist(board, u, v)
        
        # 1. 记录最大单步跨度 (Crux)
        if d > max_move:
            max_move = d
            
        # 2. 累积总距离
        total_dist += d
        
        # 3. 累积横向移动 (模拟核心力量消耗)
        r1, c1 = board.from_id(u)
        r2, c2 = board.from_id(v)
        horizontal_accum += abs(c1 - c2)

    # --- 评分公式 (Heuristic Formula) ---
    # 基础分：总长度
    # 难点加成：最大跨度的平方 (大跨度极难)
    # 消耗加成：横移
    score = (total_dist * 0.5) + (max_move ** 2.5) + (horizontal_accum * 0.8)
    
    # --- 映射到 V-Grade (V3 - V8) ---
    # 这些阈值是根据 MoonBoard 尺寸经验调整的
    if score < 15: return 3   # V3 (简单热身)
    elif score < 25: return 4 # V4
    elif score < 40: return 5 # V5
    elif score < 60: return 6 # V6
    elif score < 85: return 7 # V7
    else: return 8            # V8+
    
def generate_synthetic_data(n_routes=1000):
    board = Board()
    cons = Constraints(max_reach=6.0) # 稍微放宽，允许产生 V6/V7 的大动作
    
    routes = []
    print(f"[Synthetic] Generating {n_routes} physics-graded routes...")
    
    generated_count = 0
    
    while generated_count < n_routes:
        # A. 随机起点
        start_c = random.randint(0, 10)
        start_r = random.randint(0, 3)
        current_id = board.to_id(start_r, start_c)
        route_holds = [current_id]
        
        # B. 随机游走
        while True:
            cur_r, cur_c = board.from_id(current_id)
            if cur_r >= 17: break # 到顶
            
            candidates = []
            # 搜索范围
            for r in range(cur_r + 1, min(cur_r + 7, 18)): 
                for c in range(0, 11):
                    nid = board.to_id(r, c)
                    d = dist(board, current_id, nid)
                    if 1.0 < d <= cons.max_reach:
                        candidates.append(nid)
            
            if not candidates: break
            
            # 策略：稍微倾向于选远一点的点，不然全是碎步
            # 随机选一个，但给远点加点权重
            next_id = random.choice(candidates)
            route_holds.append(next_id)
            current_id = next_id
            
        # C. 物理定级与筛选
        if 4 <= len(route_holds) <= 15:
            # 关键：根据物理特征计算 Grade，不再是随机数！
            grade = calculate_physics_grade(board, route_holds)
            
            # 转换为详细格式
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
            
            if generated_count % 200 == 0:
                print(f"  ... {generated_count} routes generated.")

    out_path = Path("data/raw/synthetic_1k.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for r in routes:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"[Synthetic] Done! Saved to {out_path}")

if __name__ == "__main__":
    generate_synthetic_data()