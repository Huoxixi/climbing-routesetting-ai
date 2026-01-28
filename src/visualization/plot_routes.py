import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np

# MoonBoard 2016 布局
ROWS = 18
COLS = 11

def get_rc_from_id(hid):
    r = hid // COLS
    c = hid % COLS
    return r, c

def enforce_physics_rules(tokens):
    """物理规则清洗逻辑 (保持不变)"""
    unique_holds = set()
    for t in tokens:
        if isinstance(t, str) and "_H" in t:
            try:
                parts = t.split("_H")
                hid = int(parts[-1]) 
                unique_holds.add(hid)
            except: continue
        elif isinstance(t, int): # 兼容整数 ID
             unique_holds.add(t)

    if not unique_holds: return []

    points = []
    for hid in unique_holds:
        r, c = get_rc_from_id(hid)
        points.append({"r": r, "c": c, "hid": hid})
    
    points.sort(key=lambda x: (x['r'], x['c']))
    if not points: return []

    final_holds = []
    min_r = points[0]['r']
    max_r = points[-1]['r']

    for p in points:
        role = "M"
        if p['r'] == min_r: role = "S"
        elif p['r'] == max_r and max_r > min_r: role = "F"
        final_holds.append({"r": p['r'], "c": p['c'], "role": role})
    return final_holds

def plot_route(route_data, save_path, bg_path=None):
    # 调整画布比例
    fig, ax = plt.subplots(figsize=(6, 9))
    
    # --- 1. 绘制底图 ---
    if bg_path and Path(bg_path).exists():
        try:
            img = mpimg.imread(bg_path)
            ax.imshow(img, extent=[-0.5, COLS-0.5, -0.5, ROWS-0.5], alpha=0.9)
        except Exception:
            draw_grid_fallback(ax)
    else:
        draw_grid_fallback(ax)

    # --- 2. 绘制岩点 ---
    # 兼容 hold 对象列表或 token 列表
    if "holds" in route_data and isinstance(route_data["holds"], list) and isinstance(route_data["holds"][0], dict):
        holds = route_data["holds"] # 原始数据格式
    else:
        holds = enforce_physics_rules(route_data.get("tokens", [])) # 生成数据格式
        
    grade = route_data.get("grade", "?")
    
    if not holds:
        plt.close(); return

    for h in holds:
        r, c = h['r'], h['c']
        role = h.get('role', 'M')
        
        # 颜色配置 (LED 风格)
        if role == 'S':
            color = '#00FF00' # 绿
            edge = 'white'
            lw = 2
        elif role == 'F':
            color = '#FF0000' # 红
            edge = 'white'
            lw = 2
        else:
            color = '#00BFFF' # 蓝
            edge = 'none'
            lw = 0
            
        ax.add_patch(patches.Circle((c, r), 0.45, color=color, alpha=0.4))
        ax.add_patch(patches.Circle((c, r), 0.25, color=color, alpha=0.9, ec=edge, lw=lw))

    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(-0.5, ROWS - 0.5)
    ax.axis('off')
    
    plt.title(f"AI Route | V{grade}", fontsize=15, color='white', fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#222222')
    plt.close()

def draw_grid_fallback(ax):
    ax.set_facecolor('#222222')
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(-0.5, ROWS - 0.5)
    for r in range(ROWS):
        for c in range(COLS):
            ax.add_patch(patches.Circle((c, r), 0.1, color='#444444'))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bg", default="assets/moonboard_bg.jpg")
    # 【改动】不再使用全局 limit，而是每种难度限制几张
    ap.add_argument("--limit_per_grade", type=int, default=6, help="每个难度生成几张图")
    args = ap.parse_args()

    inp = Path(args.file)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lines = inp.read_text(encoding='utf-8').strip().splitlines()
    print(f"[Visualizer] Scanning {len(lines)} routes in {inp}...")
    
    # 【核心改动】分难度计数
    grade_counts = {}
    total_saved = 0

    for i, line in enumerate(lines):
        if not line.strip(): continue
        rec = json.loads(line)
        
        # 获取难度 (兼容整数或字符串)
        g = rec.get('grade', '?')
        
        # 初始化计数器
        if g not in grade_counts: grade_counts[g] = 0
        
        # 如果这个难度的图已经够了，就跳过，去找别的难度的
        if grade_counts[g] >= args.limit_per_grade:
            continue
            
        # 画图
        fname = f"viz_{i:03d}_V{g}.png"
        plot_route(rec, out / fname, args.bg)
        
        grade_counts[g] += 1
        total_saved += 1
        
    print(f"[Visualizer] Done. Saved {total_saved} images. Distribution: {grade_counts}")

if __name__ == "__main__":
    main()