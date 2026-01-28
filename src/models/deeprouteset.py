import json
import random
import math
from pathlib import Path
from src.env.board import Board
from src.betamove.constraints import Constraints, dist

def get_dist(board, h1, h2):
    r1, c1 = board.from_id(h1)
    r2, c2 = board.from_id(h2)
    return math.hypot(r1 - r2, c1 - c2)

def simplify_route(board, holds, cons):
    """
    【核心升级】路径剪枝算法 (Path Pruning)
    模拟攀岩者的“省力策略”：如果能跳过中间点直接去下一个，就跳过。
    """
    if len(holds) < 3:
        return holds
        
    simplified = [holds[0]] #以此为起点
    current_idx = 0
    
    while current_idx < len(holds) - 1:
        # 尝试“贪婪地”向前看：能不能直接跳到更远的点？
        # 我们往后看最多 2 步 (比如跳过中间 1 个点)
        # 如果步子太大容易导致线路太稀疏，所以这里保守一点
        next_hop = current_idx + 1
        
        # 检查是否可以跳过 current_idx + 1，直接去 current_idx + 2
        if current_idx + 2 < len(holds):
            target = holds[current_idx + 2]
            source = holds[current_idx]
            d = get_dist(board, source, target)
            
            # 如果距离允许 (小于最大臂展)，且确实是向上的 (避免跳回下面)
            r_src, _ = board.from_id(source)
            r_tgt, _ = board.from_id(target)
            
            if d <= cons.max_reach and r_tgt > r_src:
                # 成功跳点！
                next_hop = current_idx + 2
        
        # 加入确定的下一点
        simplified.append(holds[next_hop])
        current_idx = next_hop
        
    return simplified

def calculate_physics_grade(board, hold_ids):
    """(保持不变) 物理定级"""
    if len(hold_ids) < 2: return 0
    total_dist = 0
    max_move = 0
    horizontal_accum = 0
    for i in range(len(hold_ids) - 1):
        u, v = hold_ids[i], hold_ids[i+1]
        d = get_dist(board, u, v)
        if d > max_move: max_move = d
        total_dist += d
        r1, c1 = board.from_id(u)
        r2, c2 = board.from_id(v)
        horizontal_accum += abs(c1 - c2)
    score = (total_dist * 0.5) + (max_move ** 2.5) + (horizontal_accum * 0.8)
    if score < 15: return 3
    elif score < 25: return 4
    elif score < 40: return 5
    elif score < 60: return 6
    elif score < 85: return 7
    else: return 8

def generate_synthetic_data(n_routes=1000):
    board = Board()
    cons = Constraints(max_reach=5.5) # 臂展限制
    
    routes = []
    print(f"[Synthetic] Generating {n_routes} optimized routes (with Pruning)...")
    
    generated_count = 0
    
    while generated_count < n_routes:
        # A. 随机起点
        start_c = random.randint(0, 10)
        start_r = random.randint(0, 3)
        current_id = board.to_id(start_r, start_c)
        route_holds = [current_id]
        
        # B. 随机游走 (生成原始素材)
        # 这里我们故意生成得稍微密集一点，然后靠剪枝来优化
        while True:
            cur_r, cur_c = board.from_id(current_id)
            if cur_r >= 17: break 
            
            candidates = []
            # 搜索范围：向上找 1~4 排 (步子小一点，方便后续剪枝)
            for r in range(cur_r + 1, min(cur_r + 5, 18)): 
                for c in range(0, 11):
                    nid = board.to_id(r, c)
                    d = dist(board, current_id, nid)
                    if 1.0 < d <= cons.max_reach:
                        candidates.append(nid)
            
            if not candidates: break
            next_id = random.choice(candidates)
            route_holds.append(next_id)
            current_id = next_id
            
        # C. 【关键步骤】路径剪枝
        # 把啰嗦的随机路线变成精简的路线
        optimized_holds = simplify_route(board, route_holds, cons)
        
        # D. 筛选与定级
        # 剪枝后，点数可能会变少，我们只保留依然足够长的
        if 4 <= len(optimized_holds) <= 12:
            grade = calculate_physics_grade(board, optimized_holds)
            
            formatted_holds = []
            for j, hid in enumerate(optimized_holds):
                r, c = board.from_id(hid)
                role = "M"
                if j == 0: role = "S"
                elif j == len(optimized_holds)-1: role = "F"
                formatted_holds.append({"r": r, "c": c, "role": role})
                
            routes.append({
                "id": f"syn_{generated_count}",
                "grade": grade,
                "holds": formatted_holds,
                "board": "moonboard"
            })
            generated_count += 1
            
            if generated_count % 200 == 0:
                print(f"  ... {generated_count} optimized routes generated.")

    out_path = Path("data/raw/synthetic_1k.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for r in routes:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"[Synthetic] Done! Saved to {out_path}")

if __name__ == "__main__":
    generate_synthetic_data()