import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from pathlib import Path
import matplotlib.lines as mlines

ROWS = 18
COLS = 11

def get_rc_from_id(hid):
    return hid // COLS, hid % COLS

def parse_trace(action_tokens):
    """
    解析动作序列，生成左右手独立轨迹，以及全局严格递增的序号。
    """
    lh, rh = None, None
    lh_history = []
    rh_history = []
    all_holds = set()
    global_steps = [] # 记录全局顺序: (hand_type, (r, c), step_idx)
    
    step_idx = 1
    
    for tok in action_tokens:
        if tok.startswith("START_H"):
            hid_str = tok.split("H")[-1]
            if not hid_str.isdigit(): continue
            hid = int(hid_str)
            r, c = get_rc_from_id(hid)
            
            # 过滤异常坐标
            if not (0 <= r < ROWS and 0 <= c < COLS): continue
            
            all_holds.add((r, c))
            
            if lh is None and rh is None:
                lh = rh = (r, c)
            else:
                if c > lh[1]: rh = (r, c)
                else: rh, lh = lh, (r, c)
                
            if lh is not None and (not lh_history or lh_history[-1] != lh):
                lh_history.append(lh)
                global_steps.append(('LH', lh, 0)) # 起步点序号为 0
            if rh is not None and (not rh_history or rh_history[-1] != rh):
                rh_history.append(rh)
                global_steps.append(('RH', rh, 0))
                
        elif tok.startswith("LH_") or tok.startswith("RH_"):
            try:
                parts = tok.split("_")
                hand = parts[0]
                dr = int([p for p in parts if p.startswith('R')][0][1:])
                dc = int([p for p in parts if p.startswith('C')][0][1:])
            except: continue
            
            if hand == "LH" and lh:
                new_lh = (lh[0] + dr, lh[1] + dc)
                if not (0 <= new_lh[0] < ROWS and 0 <= new_lh[1] < COLS): continue
                lh = new_lh
                all_holds.add(lh)
                lh_history.append(lh)
                # 分配全局序号
                global_steps.append(('LH', lh, step_idx))
                step_idx += 1
            elif hand == "RH" and rh:
                new_rh = (rh[0] + dr, rh[1] + dc)
                if not (0 <= new_rh[0] < ROWS and 0 <= new_rh[1] < COLS): continue
                rh = new_rh
                all_holds.add(rh)
                rh_history.append(rh)
                # 分配全局序号
                global_steps.append(('RH', rh, step_idx))
                step_idx += 1
                
    return lh_history, rh_history, list(all_holds), global_steps

def plot_route(route_data, save_path, bg_path=None):
    tokens = route_data.get("action_tokens", [])
    grade = route_data.get("grade", "?")
    if not tokens: return False
    
    lh_history, rh_history, holds, global_steps = parse_trace(tokens)
    if not holds: return False

    fig, ax = plt.subplots(figsize=(6, 9))
    
    if bg_path and Path(bg_path).exists():
        try:
            img = mpimg.imread(bg_path)
            ax.imshow(img, extent=[-0.5, COLS-0.5, -0.5, ROWS-0.5], alpha=0.9)
        except: draw_grid_fallback(ax)
    else: draw_grid_fallback(ax)

    # 1. 垫底岩点背景
    for r, c in holds:
        ax.add_patch(patches.Circle((c, r), 0.45, color='#aaaaaa', alpha=0.3, zorder=1))
        ax.add_patch(patches.Circle((c, r), 0.20, color='#ffffff', alpha=0.5, zorder=2))

    # 2. 画出左右手的独立连线 (图层在下面)
    if lh_history and len(lh_history) > 1:
        hx, hy = [p[1] for p in lh_history], [p[0] for p in lh_history]
        ax.plot(hx, hy, color='cyan', linestyle='-', linewidth=3, alpha=0.7, zorder=3)
    if rh_history and len(rh_history) > 1:
        hx, hy = [p[1] for p in rh_history], [p[0] for p in rh_history]
        ax.plot(hx, hy, color='magenta', linestyle='-', linewidth=3, alpha=0.7, zorder=3)

    # 3. 基于全局时间线 (global_steps) 画节点和共用序号
    # 使用字典记录每个坐标上最后一个到达的手的序号，避免文本重叠
    text_records = {} 
    
    for hand, (r, c), step_idx in global_steps:
        # 区分颜色
        color = 'cyan' if hand == 'LH' else 'magenta'
        
        # 画圆圈
        ax.scatter(c, r, s=250, color=color, edgecolors='white', linewidth=1.5, alpha=0.9, zorder=4)
        
        # 记录序号（相同位置覆盖为最新的步骤）
        text_records[(r, c)] = ('S' if step_idx == 0 else str(step_idx), hand)

    # 统一渲染文字
    for (r, c), (text_str, hand) in text_records.items():
        text_color = 'black' if hand == 'LH' else 'white'
        ax.text(c, r, text_str, color=text_color, fontsize=10, fontweight='bold', ha='center', va='center', zorder=5)

    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(-0.5, ROWS - 0.5)
    ax.axis('off')
    
    plt.title(f"Biomechanics Trace | V{grade}", fontsize=15, color='white', fontweight='bold', pad=10)
    
    lh_legend = mlines.Line2D([], [], color='cyan', marker='o', markersize=10, label='Left Hand (L)')
    rh_legend = mlines.Line2D([], [], color='magenta', marker='o', markersize=10, label='Right Hand (R)')
    ax.legend(handles=[lh_legend, rh_legend], loc='upper left', facecolor='#222222', edgecolor='none', labelcolor='white')

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
    ap.add_argument("--limit_per_grade", type=int, default=10)
    args = ap.parse_args()

    inp = Path(args.file)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lines = inp.read_text(encoding='utf-8').strip().splitlines()
    print(f"[Visualizer] 正在扫描 {len(lines)} 条路线...")
    
    grade_counts = {}
    total_saved = 0

    for i, line in enumerate(lines):
        if not line.strip(): continue
        rec = json.loads(line)
        g = rec.get('grade', '?')
        if g not in grade_counts: grade_counts[g] = 0
        if grade_counts[g] >= args.limit_per_grade: continue
            
        fname = f"viz_{i:03d}_V{g}.png"
        
        # 加上异常捕获，防止单张图的错误中断整个流程
        try:
            if plot_route(rec, out / fname, args.bg):
                grade_counts[g] += 1
                total_saved += 1
        except Exception as e:
            print(f"绘制第 {i} 条路线时出错: {e}")
        
    print(f"[Visualizer] 任务完成! 保存了 {total_saved} 张全局序号轨迹图。")

if __name__ == "__main__":
    main()