import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.image as mpimg
from pathlib import Path

ROWS = 18
COLS = 11

def get_rc_from_id(hid):
    return hid // COLS, hid % COLS

def parse_states(action_tokens):
    """
    解析动作序列，还原每一帧左右手绝对坐标。
    """
    states = []
    lh, rh = None, None
    current_holds = []
    
    for tok in action_tokens:
        if tok.startswith("START_H"):
            hid = int(tok.split("H")[-1])
            r, c = get_rc_from_id(hid)
            current_holds.append((r, c))
            if lh is None and rh is None:
                lh = rh = (r, c)
            else:
                if c > lh[1]: rh = (r, c)
                else: rh, lh = lh, (r, c)
            
            states.append({'lh': lh, 'rh': rh, 'action': tok, 'holds': list(current_holds)})
            
        elif tok.startswith("LH_") or tok.startswith("RH_"):
            try:
                parts = tok.split("_")
                hand = parts[0]
                action_type = parts[1]
                dr = int([p for p in parts if p.startswith('R')][0][1:])
                dc = int([p for p in parts if p.startswith('C')][0][1:])
            except: continue
            
            if hand == "LH" and lh:
                lh = (lh[0] + dr, lh[1] + dc)
                current_holds.append(lh)
            elif hand == "RH" and rh:
                rh = (rh[0] + dr, rh[1] + dc)
                current_holds.append(rh)
                
            states.append({'lh': lh, 'rh': rh, 'action': tok, 'holds': list(current_holds)})
            
    return states

def draw_board(ax, bg_path=None):
    if bg_path and Path(bg_path).exists():
        try:
            img = mpimg.imread(bg_path)
            ax.imshow(img, extent=[-0.5, COLS-0.5, -0.5, ROWS-0.5], alpha=0.9)
            return
        except: pass
    
    ax.set_facecolor('#222222')
    for r in range(ROWS):
        for c in range(COLS):
            ax.add_patch(patches.Circle((c, r), 0.1, color='#444444'))
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(-0.5, ROWS - 0.5)

def create_animation(route_data, save_path, bg_path="assets/moonboard_bg.jpg"):
    tokens = route_data.get("action_tokens", [])
    grade = route_data.get("grade", "?")
    states = parse_states(tokens)
    if not states: return False

    fig, ax = plt.subplots(figsize=(6, 9))
    draw_board(ax, bg_path)
    ax.axis('off')
    
    # 初始化动画元素
    title = ax.text(COLS/2 - 0.5, ROWS, "", color='white', fontsize=14, fontweight='bold', ha='center')
    route_holds_scatter, = ax.plot([], [], 'o', color='#444444', markersize=15, alpha=0.5, zorder=2) 
    
    # 核心修改：左右手的历史轨迹线 (zorder控制图层，让它在点下面)
    lh_path_line, = ax.plot([], [], '-', color='cyan', linewidth=3, alpha=0.6, zorder=3)
    rh_path_line, = ax.plot([], [], '-', color='magenta', linewidth=3, alpha=0.6, zorder=3)
    
    # 左右手当前位置的光标点
    lh_marker, = ax.plot([], [], 'o', color='cyan', markersize=14, markeredgecolor='white', markeredgewidth=2, zorder=4)
    rh_marker, = ax.plot([], [], 'o', color='magenta', markersize=14, markeredgecolor='white', markeredgewidth=2, zorder=4)
    lh_text = ax.text(0, 0, 'L', color='black', fontsize=9, fontweight='bold', ha='center', va='center', zorder=5)
    rh_text = ax.text(0, 0, 'R', color='black', fontsize=9, fontweight='bold', ha='center', va='center', zorder=5)

    def init():
        return lh_path_line, rh_path_line, lh_marker, rh_marker, lh_text, rh_text, title, route_holds_scatter

    def update(frame):
        state = states[frame]
        lh, rh = state['lh'], state['rh']
        holds = state['holds']
        
        # 1. 更新背景被点亮的岩点
        hx = [c for r, c in holds]
        hy = [r for r, c in holds]
        route_holds_scatter.set_data(hx, hy)
        
        # 2. 核心修改：追溯从第 0 帧到当前帧的所有左/右手历史位置，画出轨迹线
        lh_history = [s['lh'] for s in states[:frame+1] if s['lh'] is not None]
        rh_history = [s['rh'] for s in states[:frame+1] if s['rh'] is not None]
        
        if lh_history:
            lh_path_line.set_data([p[1] for p in lh_history], [p[0] for p in lh_history])
        if rh_history:
            rh_path_line.set_data([p[1] for p in rh_history], [p[0] for p in rh_history])
            
        # 3. 更新当前手的标记点位置
        if lh:
            lh_marker.set_data([lh[1]], [lh[0]])
            lh_text.set_position((lh[1], lh[0]))
            
        if rh:
            rh_marker.set_data([rh[1]], [rh[0]])
            rh_text.set_position((rh[1], rh[0]))
            
        title.set_text(f"V{grade} | Step {frame+1}/{len(states)}: {state['action']}")
        return lh_path_line, rh_path_line, lh_marker, rh_marker, lh_text, rh_text, title, route_holds_scatter

    # 制作动画 (fps=1.5 比较容易看清动作流)
    ani = animation.FuncAnimation(fig, update, frames=len(states), init_func=init, blit=True, repeat_delay=2000)
    
    print(f"正在渲染动画轨迹 -> {save_path.name} ...")
    ani.save(save_path, writer='pillow', fps=1.5)
    plt.close()
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=5, help="生成几个动画")
    args = ap.parse_args()

    inp = Path(args.file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = inp.read_text(encoding='utf-8').strip().splitlines()
    count = 0
    
    for i, line in enumerate(lines):
        if not line.strip(): continue
        rec = json.loads(line)
        if "action_tokens" not in rec: continue
        
        grade = rec.get("grade", "?")
        save_path = out_dir / f"climb_trace_V{grade}_{i:03d}.gif"
        
        if create_animation(rec, save_path):
            count += 1
        
        if count >= args.limit: break
        
    print(f"🎉 动画制作完成！共生成了 {count} 个带轨迹的 GIF 动图。")

if __name__ == "__main__":
    main()