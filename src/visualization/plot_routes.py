import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# MoonBoard 2016 物理布局参数
ROWS = 18
COLS = 11

def get_rc_from_id(hid):
    """将 ID 转换为 (行, 列)"""
    r = hid // COLS
    c = hid % COLS
    return r, c

def enforce_physics_rules(tokens):
    """
    【核心重构】物理规则引擎
    不依赖 AI 的 S/F 标签，而是根据几何位置强制分配角色。
    规则：
    1. 收集所有被选中的点。
    2. 按高度 (Row) 从低到高排序。
    3. 最底部的点 -> 强制为 Start (绿色)。
    4. 最顶部的点 -> 强制为 Finish (红色)。
    5. 中间的所有点 -> 强制为 Middle (蓝色)。
    """
    unique_holds = set()
    
    # 1. 提取所有有效坐标
    for t in tokens:
        if "_H" not in t: continue
        try:
            # 兼容 S_H15, M_H15, F_H15, 甚至纯 H15
            parts = t.split("_H")
            hid = int(parts[-1]) 
            unique_holds.add(hid)
        except:
            continue
            
    if not unique_holds:
        return []

    # 2. 转换为对象并排序
    points = []
    for hid in unique_holds:
        r, c = get_rc_from_id(hid)
        points.append({"r": r, "c": c, "hid": hid})
    
    # 按行(r)从小到大排序，如果行相同按列(c)排
    points.sort(key=lambda x: (x['r'], x['c']))

    # 3. 强制分配角色
    final_holds = []
    
    min_r = points[0]['r']
    max_r = points[-1]['r']

    for p in points:
        role = "M" # 默认中间点
        
        # 规则：最低的一层（或多层，如果最低点有多个）是起点
        # 这里简化：绝对最低的点肯定是起点
        if p['r'] == min_r:
            role = "S"
            
        # 规则：绝对最高的点肯定是终点
        # 只有当它比起点高的时候才设为终点 (避免单点线路既是S又是F)
        elif p['r'] == max_r and max_r > min_r:
            role = "F"
            
        final_holds.append({"r": p['r'], "c": p['c'], "role": role})

    return final_holds

def draw_moonboard_grid(ax):
    """绘制极简且专业的 MoonBoard 网格"""
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(-0.5, ROWS - 0.5)
    ax.set_aspect('equal')
    ax.set_facecolor('#FAFAFA') # 干净的灰白底
    
    # 细网格
    for x in range(COLS + 1):
        ax.axvline(x - 0.5, color='#E0E0E0', lw=0.5, zorder=0)
    for y in range(ROWS + 1):
        ax.axhline(y - 0.5, color='#E0E0E0', lw=0.5, zorder=0)

    # 粗边框
    rect = patches.Rectangle((-0.5, -0.5), COLS, ROWS, linewidth=2, edgecolor='#333333', facecolor='none', zorder=1)
    ax.add_patch(rect)
    
    # 刻度
    ax.set_xticks(range(COLS))
    ax.set_xticklabels([chr(ord('A')+i) for i in range(COLS)], fontweight='bold', color='#444')
    ax.set_yticks(range(ROWS))
    ax.set_yticklabels(range(1, ROWS+1), fontweight='bold', color='#444')
    ax.tick_params(length=0) 

def plot_route(route_data, save_path):
    fig, ax = plt.subplots(figsize=(6, 9))
    draw_moonboard_grid(ax)
    
    holds = enforce_physics_rules(route_data.get("tokens", []))
    grade = route_data.get("grade", "?")
    
    # 如果没有点，直接跳过
    if not holds:
        plt.close()
        return

    # 绘制
    for h in holds:
        r, c, role = h['r'], h['c'], h['role']
        
        if role == 'S':
            color = '#00C853' # 鲜艳的绿
            edge = '#005020'
            text = 'S'
            radius = 0.4
            z = 20
        elif role == 'F':
            color = '#D50000' # 鲜艳的红
            edge = '#500000'
            text = 'F'
            radius = 0.4
            z = 20
        else:
            color = '#2962FF' # 鲜艳的蓝
            edge = 'none'
            text = ''
            radius = 0.25 # 中间点稍微小一点，突出起终点
            z = 10
            
        # 1. 绘制光晕 (Glow)
        glow = patches.Circle((c, r), radius=radius+0.15, facecolor=color, alpha=0.2, zorder=z-5)
        ax.add_patch(glow)
        
        # 2. 绘制实体点
        circ = patches.Circle((c, r), radius=radius, facecolor=color, edgecolor=edge, linewidth=1.5, zorder=z)
        ax.add_patch(circ)
        
        # 3. 绘制文字
        if text:
            ax.text(c, r, text, color='white', ha='center', va='center', fontweight='bold', fontsize=11, zorder=z+1)

    plt.title(f"Generated Route | Grade V{grade}", fontsize=14, fontweight='bold', pad=12, color='#333')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=50) # 多画几张看看
    args = ap.parse_args()

    inp = Path(args.file)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lines = inp.read_text(encoding='utf-8').strip().splitlines()
    print(f"[Visualizer] Found {len(lines)} routes. Processing with Physics Rules...")
    
    count = 0
    for i, line in enumerate(lines):
        if count >= args.limit: break
        if not line.strip(): continue
        
        rec = json.loads(line)
        # 过滤掉点数过少的废线
        if len(rec.get("tokens", [])) < 3: 
            continue
            
        fname = f"clean_route_{i:03d}_V{rec.get('grade','?')}.png"
        plot_route(rec, out / fname)
        count += 1
        
    print(f"[Visualizer] Saved {count} standardized images to {out}")

if __name__ == "__main__":
    main()