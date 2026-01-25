import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# MoonBoard 标准布局 (18行 x 11列)
ROWS = 18
COLS = 11

# 难度颜色映射
GRADE_COLORS = {
    3: "#2ecc71", # V3 Green
    4: "#f1c40f", # V4 Yellow
    5: "#e67e22", # V5 Orange
    "default": "#3498db"
}

def get_rc_from_id(hid):
    """根据 ID 计算行列 (假设 ID = r * 11 + c)"""
    r = hid // COLS
    c = hid % COLS
    return r, c

def parse_tokens(tokens):
    """解析生成的 Token，并进行视觉修正"""
    holds = []
    parsed_items = []
    
    for t in tokens:
        if "_H" not in t: continue
        parts = t.split("_H")
        if len(parts) != 2: continue
        role = parts[0] # S, M, F
        hid = int(parts[1])
        parsed_items.append({"role": role, "hid": hid})

    if not parsed_items:
        return []

    # 逻辑修正：如果全是 M，强制把首尾标记为 S 和 F
    has_start = any(x['role'] == 'S' for x in parsed_items)
    has_end = any(x['role'] == 'F' for x in parsed_items)
    
    for i, item in enumerate(parsed_items):
        role = item['role']
        hid = item['hid']
        
        if i == 0 and not has_start: role = 'S'
        if i == len(parsed_items) - 1 and not has_end: role = 'F'
            
        r, c = get_rc_from_id(hid)
        holds.append({"r": r, "c": c, "role": role})
        
    return holds

def draw_board(ax):
    """绘制 MoonBoard 背景"""
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(-0.5, ROWS - 0.5)
    ax.set_aspect('equal')
    ax.grid(True, color='#e0e0e0', linestyle='-', linewidth=0.5, zorder=0)
    
    # 边框
    rect = patches.Rectangle((-0.5, -0.5), COLS, ROWS, linewidth=3, edgecolor='#333333', facecolor='#f9f9f9', zorder=0)
    ax.add_patch(rect)
    
    # 坐标轴
    ax.set_xticks(range(COLS))
    ax.set_xticklabels(['A','B','C','D','E','F','G','H','I','J','K'], fontsize=10, fontweight='bold', color='#555')
    ax.set_yticks(range(ROWS))
    ax.set_yticklabels(range(1, ROWS+1), fontsize=10, fontweight='bold', color='#555')
    ax.tick_params(left=False, bottom=False)

def plot_route(route, save_path):
    fig, ax = plt.subplots(figsize=(6, 9))
    draw_board(ax)
    
    holds = parse_tokens(route.get("tokens", []))
    grade = route.get("grade", "?")
    theme_color = GRADE_COLORS.get(grade, GRADE_COLORS["default"])
    
    # 1. 先画连接线 (模拟手臂移动)
    for i in range(len(holds) - 1):
        h1 = holds[i]
        h2 = holds[i+1]
        ax.plot([h1['c'], h2['c']], [h1['r'], h2['r']], color=theme_color, linewidth=2, alpha=0.3, zorder=5)

    # 2. 再画岩点
    for h in holds:
        r, c, role = h['r'], h['c'], h['role']
        
        if role == 'S':
            color = '#27ae60' # 起步绿
            radius = 0.4
            text = 'S'
            edge = 'black'
        elif role == 'F':
            color = '#c0392b' # 完攀红
            radius = 0.4
            text = 'F'
            edge = 'black'
        else: # Middle
            color = theme_color
            radius = 0.25
            text = ''
            edge = 'none'
            
        circ = patches.Circle((c, r), radius=radius, facecolor=color, edgecolor=edge, linewidth=1.5, alpha=0.9, zorder=10)
        ax.add_patch(circ)
        
        if text:
            ax.text(c, r, text, color='white', ha='center', va='center', fontweight='bold', fontsize=10, zorder=20)

    plt.title(f"AI Route | Grade: V{grade}", fontsize=15, fontweight='bold', pad=12, color='#333')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    inp = Path(args.file)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lines = inp.read_text(encoding='utf-8').strip().splitlines()
    print(f"[plot] Found {len(lines)} routes. Plotting first {args.limit}...")
    
    count = 0
    for i, line in enumerate(lines):
        if count >= args.limit: break
        if not line.strip(): continue
        
        rec = json.loads(line)
        fname = f"route_{i:03d}_V{rec.get('grade','?')}.png"
        plot_route(rec, out / fname)
        count += 1
        
    print(f"[plot] Saved {count} images to {out}")

if __name__ == "__main__":
    main()