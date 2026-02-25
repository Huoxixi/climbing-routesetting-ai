import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from pathlib import Path

# MoonBoard 2016 布局
ROWS = 18
COLS = 11

def get_rc_from_id(hid):
    r = hid // COLS
    c = hid % COLS
    return r, c

def extract_holds(route_data):
    """提取岩点坐标的终极兼容逻辑"""
    final_holds = []
    
    # 1. 如果有 seq_betamove (我们新一代Action模型生成的物理坐标)
    if "seq_betamove" in route_data:
        hids = route_data["seq_betamove"]
        for i, hid in enumerate(hids):
            r, c = get_rc_from_id(hid)
            role = "M"
            if i == 0: role = "S"
            elif i == len(hids) - 1: role = "F"
            final_holds.append({"r": r, "c": c, "role": role})
        return final_holds
        
    # 2. 原始数据格式
    if "holds" in route_data and isinstance(route_data["holds"], list) and isinstance(route_data["holds"][0], dict):
        return route_data["holds"]
        
    # 3. 旧版 tokens 格式 fallback
    tokens = route_data.get("tokens", [])
    if not tokens: return []
    
    unique_holds = set()
    for t in tokens:
        if isinstance(t, str) and "_H" in t:
            try:
                unique_holds.add(int(t.split("_H")[-1]))
            except: continue
        elif isinstance(t, int):
             unique_holds.add(t)

    points = []
    for hid in unique_holds:
        r, c = get_rc_from_id(hid)
        points.append({"r": r, "c": c, "hid": hid})
    
    points.sort(key=lambda x: (x['r'], x['c']))
    if not points: return []

    min_r = points[0]['r']
    max_r = points[-1]['r']
    for p in points:
        role = "M"
        if p['r'] == min_r: role = "S"
        elif p['r'] == max_r and max_r > min_r: role = "F"
        final_holds.append({"r": p['r'], "c": p['c'], "role": role})
        
    return final_holds

def plot_route(route_data, save_path, bg_path=None):
    fig, ax = plt.subplots(figsize=(6, 9))
    
    if bg_path and Path(bg_path).exists():
        try:
            img = mpimg.imread(bg_path)
            ax.imshow(img, extent=[-0.5, COLS-0.5, -0.5, ROWS-0.5], alpha=0.9)
        except Exception:
            draw_grid_fallback(ax)
    else:
        draw_grid_fallback(ax)

    holds = extract_holds(route_data)
    grade = route_data.get("grade", "?")
    
    if not holds:
        plt.close()
        return False

    for h in holds:
        r, c = h['r'], h['c']
        role = h.get('role', 'M')
        
        if role == 'S':
            color = '#00FF00'
            edge = 'white'
            lw = 2
        elif role == 'F':
            color = '#FF0000'
            edge = 'white'
            lw = 2
        else:
            color = '#00BFFF'
            edge = 'none'
            lw = 0
            
        ax.add_patch(patches.Circle((c, r), 0.45, color=color, alpha=0.4))
        ax.add_patch(patches.Circle((c, r), 0.25, color=color, alpha=0.9, ec=edge, lw=lw))

    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(-0.5, ROWS - 0.5)
    ax.axis('off')
    
    plt.title(f"AI Action Route | V{grade}", fontsize=15, color='white', fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#222222')
    plt.close()
    return True

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
    ap.add_argument("--limit_per_grade", type=int, default=6, help="每个难度生成几张图")
    args = ap.parse_args()

    inp = Path(args.file)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lines = inp.read_text(encoding='utf-8').strip().splitlines()
    print(f"[Visualizer] 正在扫描 {len(lines)} 条路线 -> {inp}...")
    
    grade_counts = {}
    total_saved = 0

    for i, line in enumerate(lines):
        if not line.strip(): continue
        rec = json.loads(line)
        
        g = rec.get('grade', '?')
        if g not in grade_counts: grade_counts[g] = 0
        if grade_counts[g] >= args.limit_per_grade:
            continue
            
        fname = f"viz_{i:03d}_V{g}.png"
        
        if plot_route(rec, out / fname, args.bg):
            grade_counts[g] += 1
            total_saved += 1
        
    print(f"[Visualizer] 任务完成! 实际保存了 {total_saved} 张图片。分布: {grade_counts}")

if __name__ == "__main__":
    main()