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
    """(保持不变) 路径剪枝"""
    if len(holds) < 3: return holds
    simplified = [holds[0]]
    current_idx = 0
    while current_idx < len(holds) - 1:
        next_hop = current_idx + 1
        if current_idx + 2 < len(holds):
            target = holds[current_idx + 2]
            source = holds[current_idx]
            d = get_dist(board, source, target)
            r_src, _ = board.from_id(source)
            r_tgt, _ = board.from_id(target)
            if d <= cons.max_reach and r_tgt > r_src:
                next_hop = current_idx + 2
        simplified.append(holds[next_hop])
        current_idx = next_hop
    return simplified

def calculate_physics_grade(board, hold_ids):
    """
    【严格版】物理定级公式
    不妥协标准。如果分数高就是难，分数低就是简单。
    """
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

    # 评分核心：距离权重大，大跨度权重极大
    # 这个公式是基于 MoonBoard 40度板的物理直觉设计的
    score = (total_dist * 0.6) + (max_move ** 2.2) + (horizontal_accum * 0.6)
    
    # 严格的阈值 (基于大量样本观察微调)
    if score < 20: return 3   # 只有非常紧凑的线才能进 V3
    elif score < 30: return 4 
    elif score < 45: return 5 
    elif score < 65: return 6 
    elif score < 90: return 7 
    else: return 8

def generate_route_with_strategy(board, target_grade):
    """
    【针对性生成策略】
    根据目标难度，采用不同的生成逻辑，而不是撞大运。
    """
    route_holds = []
    
    # --- 1. 参数设定 ---
    if target_grade == 3:
        # V3 策略：小碎步，直上，短途
        cons = Constraints(max_reach=3.5) 
        len_limit = random.randint(6, 9)
        crux_prob = 0.0 # 绝不制造难点
    elif target_grade == 4:
        # V4 策略：正常步幅
        cons = Constraints(max_reach=4.5)
        len_limit = random.randint(7, 11)
        crux_prob = 0.1 
    elif target_grade == 5:
        # V5 策略：允许大动作，偶尔制造 Crux
        cons = Constraints(max_reach=5.5)
        len_limit = random.randint(8, 12)
        crux_prob = 0.3
    else: # Grade 6+
        # V6 策略：大开大合，强制制造 Crux
        cons = Constraints(max_reach=6.5)
        len_limit = random.randint(8, 14)
        crux_prob = 0.6 # 高概率尝试大跨度

    # --- 2. 生成过程 ---
    # 随机起点
    start_c = random.randint(0, 10)
    start_r = random.randint(0, 3)
    current_id = board.to_id(start_r, start_c)
    route_holds = [current_id]
    
    while True:
        cur_r, cur_c = board.from_id(current_id)
        if cur_r >= 17: break 
        if len(route_holds) >= len_limit: break

        candidates = []
        # 搜索范围
        search_radius = 6 if target_grade >= 5 else 4
        
        for r in range(cur_r + 1, min(cur_r + search_radius, 18)): 
            for c in range(0, 11):
                nid = board.to_id(r, c)
                d = dist(board, current_id, nid)
                if 1.0 < d <= cons.max_reach:
                    candidates.append({"id": nid, "dist": d})
        
        if not candidates: break
        
        # --- 3. 智能选点逻辑 ---
        if random.random() < crux_prob:
            # 【制造难点】：倾向于选最远的那个点 (Crux Move)
            # 按距离降序排列，取前 20%
            candidates.sort(key=lambda x: x["dist"], reverse=True)
            top_n = max(1, len(candidates) // 4)
            choice = random.choice(candidates[:top_n])
        else:
            # 【正常攀爬】：倾向于选距离适中的点 (Flow Move)
            # 或者是 V3 的时候，倾向于选近点
            if target_grade == 3:
                 candidates.sort(key=lambda x: x["dist"]) # 升序，选近的
                 top_n = max(1, len(candidates) // 3)
                 choice = random.choice(candidates[:top_n])
            else:
                 choice = random.choice(candidates)
                 
        next_id = choice["id"]
        route_holds.append(next_id)
        current_id = next_id

    # --- 4. 剪枝与定级 ---
    optimized_holds = simplify_route(board, route_holds, cons)
    
    if 4 <= len(optimized_holds) <= 15:
        grade = calculate_physics_grade(board, optimized_holds)
        return optimized_holds, grade
    return None, 0

def generate_synthetic_data(n_routes=1200):
    board = Board()
    target_per_grade = n_routes // 4
    counts = {3: 0, 4: 0, 5: 0, 6: 0}
    routes = []
    
    print(f"[Synthetic] Generating {n_routes} STRICT balanced routes...")
    print(f"            Target: {target_per_grade} per grade (No Compromise)")

    attempts = 0
    while len(routes) < n_routes:
        attempts += 1
        if attempts > n_routes * 500: # 即使跑 50万次也要跑出来
            print(f"[Warn] Loop limit reached. Final counts: {counts}")
            break
            
        # 1. 决定我们要生成哪个难度的 (缺啥补啥)
        # 找出当前数量最少的难度
        min_count = min(counts.values())
        target_grade = random.choice([g for g in [3,4,5,6] if counts[g] == min_count])
        
        # 如果都已经满了，就随便选一个防止死循环 (最后会截断)
        if counts[target_grade] >= target_per_grade:
             if len(routes) >= n_routes: break
             target_grade = random.choice([3,4,5,6])

        # 2. 尝试生成
        holds, grade = generate_route_with_strategy(board, target_grade)
        
        # 3. 【严格验收】
        # 只有当 生成出来的实际难度 == 我们想要的难度 时，才收货！
        # 绝不把 V4 标成 V3，绝不把 V8 标成 V6 (除非是兜底逻辑)
        # 这里我们允许 V6 兜底 (>=6 都算 6)，但 V3/V4/V5 必须精准
        
        valid = False
        if target_grade == grade:
            valid = True
        elif target_grade == 6 and grade >= 6: # V6 以上都归入 V6 桶
            grade = 6
            valid = True
            
        if valid and counts.get(grade, 0) < target_per_grade:
            formatted_holds = []
            for j, hid in enumerate(holds):
                r, c = board.from_id(hid)
                role = "M"
                if j == 0: role = "S"
                elif j == len(holds)-1: role = "F"
                formatted_holds.append({"r": r, "c": c, "role": role})
                
            routes.append({
                "id": f"syn_{len(routes)}",
                "grade": grade,
                "holds": formatted_holds,
                "board": "moonboard"
            })
            counts[grade] += 1
            
            if sum(counts.values()) % 50 == 0:
                print(f"  Progress: {counts}")

    out_path = Path("data/raw/synthetic_1k.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for r in routes:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"[Synthetic] Done! Final distribution: {counts}")

if __name__ == "__main__":
    generate_synthetic_data()